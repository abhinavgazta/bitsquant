import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
import ta
import matplotlib.pyplot as plt
import shap

# ==========================================
# 1. DATA ENGINEERING LAYER
# ==========================================
def fetch_and_process_data(tickers, start_date="2015-01-01", end_date="2025-01-01"):
    """
    Fetches NIFTY 50 component data and processes technical/macro features.
    """
    print(f"Fetching data for {len(tickers)} assets...")
    # 'group_by' ensures we get a MultiIndex (Ticker, OHLCV)
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    
    processed_frames = []
    
    for ticker in tickers:
        # Handle cases where yfinance returns different structures
        if len(tickers) == 1:
            df = data.copy()
        else:
            try:
                df = data[ticker].copy()
            except KeyError:
                print(f"Skipping missing ticker: {ticker}")
                continue
                
        if df.empty: continue
        
        # --- Feature Engineering (as per Dissertation Chapter 3) ---
        
        # 1. Log Returns (Stationarity)
        # Add 1e-8 to avoid log(0) errors
        df['log_ret'] = np.log(df['Close'] / (df['Close'].shift(1) + 1e-8))
        
        # 2. Technical Indicators
        # RSI (Momentum)
        df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi() / 100.0
        
        # MACD (Trend)
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd_diff()
        
        # Bollinger Width (Volatility)
        bb = ta.volatility.BollingerBands(df['Close'])
        df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / (df['Close'] + 1e-8)
        
        # ATR (For Dynamic Slippage Calculation)
        df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        
        # 3. Volume / Liquidity
        # ADV_20: Average Daily Volume * Price (Approx Trade Value)
        df['adv_20'] = df['Close'] * df['Volume'].rolling(window=20).mean()
        df['vol_ratio'] = (df['Close'] * df['Volume']) / (df['adv_20'] + 1e-8)
        
        # Drop initial NaNs from rolling windows
        df.dropna(inplace=True)
        
        # 4. Normalization (Rolling Z-Score for Neural Net Stability)
        cols_to_norm = ['log_ret', 'rsi', 'macd', 'bb_width', 'vol_ratio']
        for col in cols_to_norm:
            rolling_mean = df[col].rolling(60).mean()
            rolling_std = df[col].rolling(60).std()
            df[col] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
            
        df['ticker'] = ticker
        processed_frames.append(df.dropna())

    # Combine and Pivot to align dates
    if not processed_frames:
        raise ValueError("No data downloaded. Check your internet or ticker list.")
        
    combined = pd.concat(processed_frames)
    # Pivot to shape: Index=Date, Columns=(Feature, Ticker)
    pivot_df = combined.pivot_table(index='Date', columns='ticker', values=['Close', 'Open', 'log_ret', 'rsi', 'macd', 'bb_width', 'vol_ratio', 'atr', 'adv_20'])
    pivot_df.dropna(inplace=True)
    
    return pivot_df

# ==========================================
# 2. CUSTOM MARKET SIMULATOR (GYMNASIUM)
# ==========================================
class IndianEquityEnv(gym.Env):
    """
    Simulates Indian Equity Market with Linear & Quadratic Costs.
    Matches Dissertation Chapter 2.2 requirements.
    """
    def __init__(self, df, lookback_window=30, initial_balance=1_000_000):
        super(IndianEquityEnv, self).__init__()
        self.df = df
        # Extract tickers from the columns
        self.tickers = df['Close'].columns.tolist()
        self.n_assets = len(self.tickers)
        self.lookback = lookback_window
        self.initial_balance = initial_balance
        
        # Action Space: Weights for [Cash, Asset_1, ..., Asset_N]
        # Using Softmax normalization in step() to ensure sum=1
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_assets + 1,), dtype=np.float32)
        
        # Observation Space: (Lookback, Total Features)
        # Features used: log_ret, rsi, macd, bb_width, vol_ratio
        self.n_features_per_asset = 5 
        self.total_features = self.n_assets * self.n_features_per_asset
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(lookback_window, self.total_features), 
            dtype=np.float32
        )
        
        # Cost Parameters (Chapter 2.1.2)
        self.stt_brokerage = 0.0015 # 15 bps Linear
        self.slippage_coeff = 0.1   # Quadratic Beta
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.lookback
        self.balance = self.initial_balance
        self.weights = np.zeros(self.n_assets + 1)
        self.weights[0] = 1.0 # Start with 100% Cash
        self.portfolio_value = self.initial_balance
        
        return self._get_obs(), {}

    def _get_obs(self):
        # Slice the data window
        idx = self.current_step
        start_idx = idx - self.lookback
        
        # Stack features for all assets
        # Shape: [Lookback, N_Assets] for each feature
        obs_list = []
        for feat in ['log_ret', 'rsi', 'macd', 'bb_width', 'vol_ratio']:
            data_slice = self.df[feat].iloc[start_idx:idx].values
            obs_list.append(data_slice)
            
        # Concatenate along the feature axis -> (Lookback, Total_Features)
        return np.concatenate(obs_list, axis=1).astype(np.float32)

    def step(self, action):
        # 1. Action Normalization (Softmax) -> Portfolio Weights
        exp_action = np.exp(action)
        weights = exp_action / np.sum(exp_action)
        
        # 2. Market Movement (Close-to-Close for simplicity in daily data)
        # Note: A more precise simulator uses Open prices as requested in Ch 2, 
        # but Close-Close is standard for vectorization unless Next-Open is explicitly in df.
        current_prices = self.df['Close'].iloc[self.current_step].values
        next_prices = self.df['Close'].iloc[self.current_step + 1].values
        
        # Asset vector return
        asset_returns = (next_prices - current_prices) / current_prices
        
        # 3. Gross Portfolio Return
        # weights[0] is cash (0 return), weights[1:] are assets
        weighted_returns = np.dot(weights[1:], asset_returns)
        
        # 4. Transaction Cost Calculation
        # Turnover: sum of absolute weight changes
        delta_weights = np.abs(weights - self.weights)
        turnover = np.sum(delta_weights)
        
        # Linear Cost (STT + Brokerage)
        cost_linear = turnover * self.stt_brokerage
        
        # Quadratic Slippage (Impact Cost)
        # Trade Value per asset approx = delta_weight * portfolio_value
        advs = self.df['adv_20'].iloc[self.current_step].values
        trade_vals = delta_weights[1:] * self.portfolio_value
        
        # Avoid division by zero with +1e-5
        cost_slippage = np.sum(self.slippage_coeff * (trade_vals / (advs + 1e-5))**2)
        
        # Total cost as a percentage of portfolio
        total_cost_pct = cost_linear + (cost_slippage / self.portfolio_value)
        
        # 5. Net Return & Reward
        net_ret = weighted_returns - total_cost_pct
        
        # Downside Risk Penalty (from Dissertation Eq)
        downside_penalty = min(0, net_ret)**2 * 10
        reward = net_ret - downside_penalty
        
        # 6. Update State
        self.portfolio_value *= (1 + net_ret)
        self.weights = weights
        self.current_step += 1
        
        # Check Termination
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        info = {
            'portfolio_value': self.portfolio_value,
            'net_return': net_ret,
            'cost_linear': cost_linear,
            'cost_slippage': cost_slippage
        }
        
        return self._get_obs(), reward, terminated, truncated, info

# ==========================================
# 3. HYBRID TRANSFORMER ARCHITECTURE
# ==========================================
class TransformerExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor for Stable-Baselines3.
    Inputs: (Batch, Lookback, Features)
    Outputs: (Batch, 128) - Latent Representation
    """
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        
        # Observation shape is (Lookback, Total_Features)
        self.seq_len = observation_space.shape[0]
        self.input_dim = observation_space.shape[1]
        self.d_model = 128
        
        # 1. Input Projection
        self.linear_in = nn.Linear(self.input_dim, self.d_model)
        
        # 2. Transformer Encoder
        # batch_first=True is CRITICAL here because SB3 sends (Batch, Seq, Feat)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=4, 
            dim_feedforward=512, 
            dropout=0.1, 
            batch_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 3. Output Head
        self.linear_out = nn.Linear(self.seq_len * self.d_model, features_dim)
        
        # Activation
        self.act = nn.Tanh()

    def forward(self, observations):
        # SB3 automatically batches the input: (Batch_Size, Lookback, Features)
        # We do NOT need unsqueeze() because it's already 3D.
        
        # Project -> (Batch, Seq, d_model)
        x = self.linear_in(observations)
        
        # Encode -> (Batch, Seq, d_model)
        x = self.transformer(x)
        
        # Flatten -> (Batch, Seq * d_model)
        x = x.flatten(start_dim=1)
        
        # Project -> (Batch, features_dim)
        return self.act(self.linear_out(x))


# ==========================================
# 4. TRAINING & EVALUATION PIPELINE
# ==========================================

def run_dissertation_experiment():
    # 1. Define Universe (Expand this list for full dissertation)
    TICKERS = ['RELIANCE.NS', 'INFY.NS', 'HDFCBANK.NS', 'TCS.NS', 'SBIN.NS']
    
    print("Step 1: Fetching Data...")
    try:
        df_gym = fetch_and_process_data(TICKERS)
        print(f"Data Loaded successfully. Shape: {df_gym.shape}")
    except Exception as e:
        print(f"Data Error: {e}")
        return

    # 2. Split Data (Train / Test)
    train_size = int(len(df_gym) * 0.8)
    train_df = df_gym.iloc[:train_size]
    test_df = df_gym.iloc[train_size:]
    
    # 3. Setup Environment
    print("Step 2: Setting up Environment & Agent...")
    env = IndianEquityEnv(train_df, lookback_window=30)
    
    # 4. Initialize PPO Agent with Transformer
    policy_kwargs = dict(
        features_extractor_class=TransformerExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )
    
    model = PPO(
        "MlpPolicy", 
        env, 
        policy_kwargs=policy_kwargs, 
        verbose=1, 
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01 # Encourage exploration
    )
    
    # 5. Train
    print("Step 3: Training Agent (This may take time)...")
    model.learn(total_timesteps=15000) # Increase to 1M+ for final results
    print("Training Complete.")
    
    # 6. Evaluate on Test Set
    print("Step 4: Evaluating on Out-of-Sample Data...")
    env_test = IndianEquityEnv(test_df, lookback_window=30)
    obs, _ = env_test.reset()
    done = False
    
    portfolio_history = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_test.step(action)
        portfolio_history.append(info['portfolio_value'])
        done = terminated or truncated
        
    # 7. Visualization (Dissertation Requirement: Equity Curve)
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_history, label='DRL Agent (Transformer-PPO)')
    plt.title('Out-of-Sample Equity Curve (NSE/BSE)')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value (INR)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 8. Explainability (SHAP) - Optional but requested for dissertation
    # 8. Explainability (SHAP)
    print("Step 5: Generating SHAP Explainability Plot...")
    try:
        # 1. Collect a small background dataset (e.g., 50 samples)
        # We need this to be a numpy array, not a list of arrays
        env_explain = IndianEquityEnv(test_df, lookback_window=30)
        obs, _ = env_explain.reset()
        
        background_samples = []
        for _ in range(50): 
            background_samples.append(obs)
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env_explain.step(action)
            if done: break
            
        # Convert to numpy array: (50, Lookback, Features)
        X_3d = np.array(background_samples)
        
        # 2. FLATTEN the 3D data to 2D for SHAP
        # New Shape: (50, Lookback * Features)
        # This tricks SHAP into treating every time-step/feature pair as a unique variable
        X_2d = X_3d.reshape(X_3d.shape[0], -1)
        
        # 3. Define a Wrapper Function
        # This function takes the 2D data from SHAP, reshapes it back to 3D, and queries the model
        def predict_wrapper(X_flattened):
            # Reshape back to (Batch, Lookback, Features)
            # We derive dimensions from the original 3D array
            batch_size = X_flattened.shape[0]
            lookback = X_3d.shape[1]
            features = X_3d.shape[2]
            
            X_restored = X_flattened.reshape(batch_size, lookback, features)
            
            # Convert to Torch Tensor
            X_torch = torch.as_tensor(X_restored).to(model.device)
            
            with torch.no_grad():
                # Get the mean action (deterministic) from the policy
                # .mode() gives the most likely action
                return model.policy.get_distribution(X_torch).mode().cpu().numpy()

        # 4. Run SHAP KernelExplainer on 2D data
        # We use a small subset (e.g., first 5 samples) as the "background" to speed it up
        explainer = shap.KernelExplainer(predict_wrapper, X_2d[:5])
        
        # Calculate SHAP values for a few instances (e.g., next 10 samples)
        shap_values = explainer.shap_values(X_2d[5:15])
        
        # 5. Generate Feature Names for the Plot
        # Since we flattened the data, we need names like "RSI_t-0", "RSI_t-1", etc.
        base_features = ['log_ret', 'rsi', 'macd', 'bb_width', 'vol_ratio'] * env_explain.n_assets
        feature_names_2d = []
        for t in range(X_3d.shape[1]): # Iterate over lookback steps
            for feat in base_features:
                feature_names_2d.append(f"{feat}_t-{X_3d.shape[1]-t}")

        # 6. Plot
        plt.figure()
        shap.summary_plot(shap_values, X_2d[5:15], feature_names=feature_names_2d)
        plt.show()
        
    except Exception as e:
        print(f"SHAP Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_dissertation_experiment()