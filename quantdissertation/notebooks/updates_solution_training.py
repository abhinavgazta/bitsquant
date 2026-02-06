import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import ta
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import optuna
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. DATA ENGINEERING LAYER
# ==========================================
def fetch_and_process_data(tickers, start_date="2015-01-01", end_date="2025-01-01"):
    print(f"Fetching data for {len(tickers)} assets...")
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    
    processed_frames = []
    
    for ticker in tickers:
        if len(tickers) == 1:
            df = data.copy()
        else:
            try:
                df = data[ticker].copy()
            except KeyError:
                continue
                
        if df.empty: continue
        
        # --- Feature Engineering ---
        df['log_ret'] = np.log(df['Close'] / (df['Close'].shift(1) + 1e-8))
        df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi() / 100.0
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd_diff()
        bb = ta.volatility.BollingerBands(df['Close'])
        df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / (df['Close'] + 1e-8)
        df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        df['adv_20'] = df['Close'] * df['Volume'].rolling(window=20).mean()
        df['vol_ratio'] = (df['Close'] * df['Volume']) / (df['adv_20'] + 1e-8)
        
        df.dropna(inplace=True)
        
        # Normalize
        cols_to_norm = ['log_ret', 'rsi', 'macd', 'bb_width', 'vol_ratio']
        for col in cols_to_norm:
            rolling_mean = df[col].rolling(60).mean()
            rolling_std = df[col].rolling(60).std()
            df[col] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
            
        df['ticker'] = ticker
        processed_frames.append(df.dropna())

    if not processed_frames:
        raise ValueError("No data downloaded.")
        
    combined = pd.concat(processed_frames)
    pivot_df = combined.pivot_table(index='Date', columns='ticker', values=['Close', 'log_ret', 'rsi', 'macd', 'bb_width', 'vol_ratio', 'atr', 'adv_20'])
    pivot_df.dropna(inplace=True)
    
    return pivot_df



# ==========================================
# 2. MARKET SIMULATOR (Dynamic Rewards)
# ==========================================
class IndianEquityEnv(gym.Env):
    def __init__(self, df, reward_params=None, lookback_window=30, initial_balance=1_000_000):
        super(IndianEquityEnv, self).__init__()
        self.df = df
        self.tickers = df['Close'].columns.tolist()
        self.n_assets = len(self.tickers)
        self.lookback = lookback_window
        self.initial_balance = initial_balance
        
        # Default Reward Hyperparameters (can be tuned by Optuna)
        # alpha=turnover, beta=slippage, gamma=risk [cite: 277]
        self.reward_params = reward_params if reward_params else {
            'alpha': 0.001, 
            'beta': 0.1, 
            'gamma': 10.0
        }
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_assets + 1,), dtype=np.float32)
        
        # Features: log_ret, rsi, macd, bb_width, vol_ratio (5 features)
        self.n_features = 5 
        self.obs_shape = (lookback_window, self.n_assets * self.n_features)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32)
        
        self.stt_brokerage = 0.0015
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.lookback
        self.balance = self.initial_balance
        self.weights = np.zeros(self.n_assets + 1)
        self.weights[0] = 1.0 
        self.portfolio_value = self.initial_balance
        return self._get_obs(), {}

    def _get_obs(self):
        idx = self.current_step
        obs_data = []
        for feat in ['log_ret', 'rsi', 'macd', 'bb_width', 'vol_ratio']:
            obs_data.append(self.df[feat].iloc[idx-self.lookback:idx].values)
        return np.concatenate(obs_data, axis=1).astype(np.float32)

    def step(self, action):
        exp_action = np.exp(action)
        weights = exp_action / np.sum(exp_action)
        
        current_prices = self.df['Close'].iloc[self.current_step].values
        next_prices = self.df['Close'].iloc[self.current_step + 1].values
        asset_returns = (next_prices - current_prices) / current_prices
        
        weighted_returns = np.dot(weights[1:], asset_returns)
        
        # --- Cost Calculation ---
        delta_weights = np.abs(weights - self.weights)
        turnover = np.sum(delta_weights)
        
        # Linear Penalty (Alpha)
        cost_linear = turnover * self.stt_brokerage
        
        # Quadratic Slippage (Beta)
        advs = self.df['adv_20'].iloc[self.current_step].values
        trade_vals = delta_weights[1:] * self.portfolio_value
        # Using Optuna-tuned beta parameter
        slippage_penalty = self.reward_params['beta'] * np.sum((trade_vals / (advs + 1e-5))**2)
        
        net_ret = weighted_returns - cost_linear - (slippage_penalty / self.portfolio_value)
        
        # Downside Risk Penalty (Gamma)
        # Using Optuna-tuned gamma parameter
        downside_penalty = min(0, net_ret)**2 * self.reward_params['gamma']
        
        # Alpha is applied as a direct penalty to reward if desired, 
        # or implicitly via the linear cost subtraction above. 
        # Here we add an explicit turnover penalty if alpha is tuned separately from STT.
        explicit_turnover_penalty = turnover * self.reward_params['alpha']
        
        reward = net_ret - downside_penalty - explicit_turnover_penalty
        
        self.portfolio_value *= (1 + net_ret)
        self.weights = weights
        self.current_step += 1
        
        terminated = self.current_step >= len(self.df) - 1
        current_atr = self.df['atr'].iloc[self.current_step].mean() 
        
        info = {
            'portfolio_value': self.portfolio_value,
            'turnover': turnover,
            'atr': current_atr,
            'net_return': net_ret
        }
        
        return self._get_obs(), reward, terminated, False, info



# ==========================================
# 3. EXPLAINABLE TRANSFORMER
# ==========================================
class ExplainableTransformer(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        
        self.seq_len = observation_space.shape[0]
        self.input_dim = observation_space.shape[1]
        self.d_model = 128
        
        self.linear_in = nn.Linear(self.input_dim, self.d_model)
        self.mha = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.linear_out = nn.Linear(self.seq_len * self.d_model, features_dim)
        self.act = nn.Tanh()
        
        self.latest_attn_weights = None

    def forward(self, observations):
        x = self.linear_in(observations)
        attn_output, attn_weights = self.mha(x, x, x, need_weights=True, average_attn_weights=False)
        self.latest_attn_weights = attn_weights.detach().cpu().numpy()
        x = self.norm1(x + attn_output)
        x = x.flatten(start_dim=1)
        return self.act(self.linear_out(x))


# ==========================================
# 4. WARM-UP & OPTIMIZATION FUNCTIONS
# ==========================================

def pretrain_encoder(agent, env, epochs=5):
    """
    Supervised Warm-up: Trains the feature extractor to predict next-day returns.
    This stabilizes the embeddings before RL begins. 
    """
    print(f"   >>> Starting Transformer Warm-up for {epochs} epochs...")
    
    # Extract Feature Extractor from PPO Agent
    feature_extractor = agent.policy.features_extractor
    
    # Create a simple predictor head just for pre-training
    # (Latent -> N_Assets returns)
    latent_dim = feature_extractor.features_dim
    n_targets = env.n_assets
    predictor = nn.Linear(latent_dim, n_targets).to(agent.device)
    
    optimizer = optim.Adam(list(feature_extractor.parameters()) + list(predictor.parameters()), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    # Generate batch of data from Env
    # (Simplification: using random sampling from dataframe for demo speed)
    n_samples = 1000
    obs_batch = []
    target_batch = []
    
    valid_indices = range(env.lookback, len(env.df) - 1)
    sampled_indices = np.random.choice(valid_indices, n_samples)
    
    for idx in sampled_indices:
        # Construct Obs
        obs_data = []
        for feat in ['log_ret', 'rsi', 'macd', 'bb_width', 'vol_ratio']:
            obs_data.append(env.df[feat].iloc[idx-env.lookback:idx].values)
        obs = np.concatenate(obs_data, axis=1).astype(np.float32)
        obs_batch.append(obs)
        
        # Construct Target (Next day log returns)
        curr = env.df['Close'].iloc[idx].values
        next_p = env.df['Close'].iloc[idx+1].values
        target = np.log(next_p / (curr + 1e-8))
        target_batch.append(target)
        
    obs_tensor = torch.tensor(np.array(obs_batch)).to(agent.device)
    target_tensor = torch.tensor(np.array(target_batch), dtype=torch.float32).to(agent.device)
    
    # Training Loop
    feature_extractor.train()
    predictor.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        latent = feature_extractor(obs_tensor)
        preds = predictor(latent)
        loss = loss_fn(preds, target_tensor)
        loss.backward()
        optimizer.step()
        print(f"      [Warmup Epoch {epoch+1}/{epochs}] MSE Loss: {loss.item():.6f}")
        
    print("   >>> Warm-up Complete. Encoder initialized.")

def objective(trial, train_df, test_df):
    """
    Optuna Objective Function.
    Tunes: Learning Rate, Entropy Coef, Reward Penalties (Alpha, Beta, Gamma).
    """
    # 1. Suggest Hyperparameters [cite: 275]
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.1, log=True)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    
    # 2. Suggest Reward Parameters [cite: 277]
    alpha_turnover = trial.suggest_float("alpha_turnover", 0.0, 0.01)
    beta_slippage = trial.suggest_float("beta_slippage", 0.05, 0.5)
    gamma_risk = trial.suggest_float("gamma_risk", 1.0, 20.0)
    
    reward_params = {
        'alpha': alpha_turnover,
        'beta': beta_slippage,
        'gamma': gamma_risk
    }
    
    # 3. Setup Env & Agent
    env = IndianEquityEnv(train_df, reward_params=reward_params)
    
    policy_kwargs = dict(
        features_extractor_class=ExplainableTransformer,
        features_extractor_kwargs=dict(features_dim=128),
    )
    
    model = PPO("MlpPolicy", env, 
                learning_rate=learning_rate,
                ent_coef=ent_coef,
                clip_range=clip_range,
                policy_kwargs=policy_kwargs, 
                verbose=0)
    
    # 4. Supervised Warm-up Phase 
    # We do a short warm-up for every trial to ensure fairness
    pretrain_encoder(model, env, epochs=3)
    
    # 5. RL Training (Short duration for Optimization loop)
    # Reduced steps as requested for demo/debugging
    model.learn(total_timesteps=3000) 
    
    # 6. Evaluation (Return Sharpe Ratio or Mean Reward)
    eval_env = IndianEquityEnv(test_df, reward_params=reward_params)
    obs, _ = eval_env.reset()
    done = False
    total_rewards = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        total_rewards += reward
        done = terminated or truncated
        
    return total_rewards



# ==========================================
# 5. MAIN EXECUTION LOOP
# ==========================================
def run_full_experiment():
    TICKERS = ['RELIANCE.NS', 'INFY.NS', 'HDFCBANK.NS', 'TCS.NS']
    print("--- Step 1: Loading Data ---")
    try:
        df_gym = fetch_and_process_data(TICKERS)
        print(f"Data Loaded: {df_gym.shape}")
    except Exception as e:
        print(f"Data Error: {e}")
        return

    # Split
    train_size = int(len(df_gym) * 0.8)
    train_df = df_gym.iloc[:train_size]
    test_df = df_gym.iloc[train_size:]
    
    # --- Step 2: Bayesian Optimization (Optuna) ---
    print("--- Step 2: Starting Bayesian Optimization (Optuna) ---")
    study = optuna.create_study(direction="maximize")
    # Reduced n_trials for demo speed
    study.optimize(lambda trial: objective(trial, train_df, test_df), n_trials=3) 
    
    print("\nBest Params Found:")
    print(study.best_params)
    
    # --- Step 3: Final Training with Best Params ---
    print("\n--- Step 3: Final Training with Best Parameters ---")
    best_params = study.best_params
    best_reward_params = {
        'alpha': best_params['alpha_turnover'],
        'beta': best_params['beta_slippage'],
        'gamma': best_params['gamma_risk']
    }
    
    final_env = IndianEquityEnv(train_df, reward_params=best_reward_params)
    
    policy_kwargs = dict(
        features_extractor_class=ExplainableTransformer,
        features_extractor_kwargs=dict(features_dim=128),
    )
    
    final_model = PPO("MlpPolicy", final_env,
                      learning_rate=best_params['lr'],
                      ent_coef=best_params['ent_coef'],
                      clip_range=best_params['clip_range'],
                      policy_kwargs=policy_kwargs,
                      verbose=1)
    
    # Final Warmup
    pretrain_encoder(final_model, final_env, epochs=5)
    
    # Longer training for final result
    print("Starting final RL loop...")
    final_model.learn(total_timesteps=5000)
    
    # --- Step 4: Analysis & Visualization ---
    print("--- Step 4: Analysis & Visualizations ---")
    env_test = IndianEquityEnv(test_df, reward_params=best_reward_params)
    obs, _ = env_test.reset()
    done = False
    
    history = {'atr': [], 'turnover': [], 'observations': []}
    
    while not done:
        history['observations'].append(obs)
        action, _ = final_model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env_test.step(action)
        history['turnover'].append(info['turnover'])
        history['atr'].append(info['atr'])
        done = terminated or truncated

    # Analysis A: Turnover vs Volatility
    print("Generating: Cost Awareness Plot...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=history['atr'], y=history['turnover'], alpha=0.6)
    if len(history['atr']) > 1:
        z = np.polyfit(history['atr'], history['turnover'], 1)
        p = np.poly1d(z)
        plt.plot(history['atr'], p(history['atr']), "r--")
    plt.title("Turnover vs Volatility (Optuna Optimized)")
    plt.show()

    # Analysis B: Grouped SHAP (FIXED)
    print("Generating: Grouped SHAP Analysis...")
    X_3d = np.array(history['observations'][:50]) 
    X_2d = X_3d.reshape(X_3d.shape[0], -1)
    
    def predict_wrapper(X_flattened):
        batch_size = X_flattened.shape[0]
        lookback = X_3d.shape[1]
        features = X_3d.shape[2]
        X_restored = X_flattened.reshape(batch_size, lookback, features)
        X_torch = torch.as_tensor(X_restored).to(final_model.device)
        with torch.no_grad():
            return final_model.policy.get_distribution(X_torch).mode().cpu().numpy()

    explainer = shap.KernelExplainer(predict_wrapper, X_2d[:5])
    shap_values = explainer.shap_values(X_2d[5:15])
    
    # --- FIX FOR SHAP DIMENSIONS ---
    # shap_values might be a list (if output is vector) or array
    # We need to collapse the "Output/Action" dimension to get a single scalar importance per feature
    if isinstance(shap_values, list):
        # Sum importance across all output action dimensions
        shap_matrix = np.sum([np.abs(sv) for sv in shap_values], axis=0)
    else:
        # If it's already an array (Samples, Features, Outputs)
        if len(shap_values.shape) == 3:
            shap_matrix = np.sum(np.abs(shap_values), axis=2)
        else:
            shap_matrix = np.abs(shap_values)

    # Now shap_matrix is (Samples, Features). Average over samples.
    mean_shap = np.mean(shap_matrix, axis=0) # Shape: (Total_Features,)
    
    feature_groups = {
        "Momentum": ["rsi", "macd"],
        "Volatility": ["bb_width", "atr"],
        "Liquidity": ["vol_ratio", "adv_20"],
        "Returns": ["log_ret"]
    }
    
    raw_feature_names = ['log_ret', 'rsi', 'macd', 'bb_width', 'vol_ratio']
    group_scores = {k: 0.0 for k in feature_groups} # Ensure floats
    n_features = len(raw_feature_names)
    
    for i, imp in enumerate(mean_shap):
        feat_type_idx = (i // 30) % n_features 
        feat_name = raw_feature_names[feat_type_idx]
        for group, keywords in feature_groups.items():
            if feat_name in keywords:
                # Add scalar value
                group_scores[group] += float(imp)

    plt.figure(figsize=(8, 5))
    plt.bar(list(group_scores.keys()), list(group_scores.values()), color=['#3498db', '#e74c3c', '#f1c40f', '#2ecc71'])
    plt.title("Feature Importance Grouping (Fixed)")
    plt.ylabel("Total |SHAP| Importance")
    plt.show()






if __name__ == "__main__":
    run_full_experiment()