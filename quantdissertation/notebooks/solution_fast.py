import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import ta
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import scipy.stats as stats
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. ADVANCED FINANCIAL METRICS UTILS (ROBUST)
# ==========================================
class FinancialMetrics:
    @staticmethod
    def get_max_drawdown(prices):
        prices = np.array(prices)
        if len(prices) < 2: return 0.0
        peaks = np.maximum.accumulate(prices)
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            drawdowns = (prices - peaks) / (peaks + 1e-9)
        return np.min(np.nan_to_num(drawdowns))

    @staticmethod
    def get_sharpe_ratio(returns, risk_free_rate=0.06):
        returns = np.nan_to_num(returns)
        if len(returns) < 2: return 0.0
        
        # Annualized Excess Return
        excess_ret = returns - (risk_free_rate / 252)
        std_dev = np.std(excess_ret)
        
        if std_dev < 1e-9: return 0.0 # Avoid div by zero
        
        # Annualize Sharpe
        return (np.mean(excess_ret) / std_dev) * np.sqrt(252)

    @staticmethod
    def get_sortino_ratio(returns, risk_free_rate=0.06):
        returns = np.nan_to_num(returns)
        excess_ret = returns - (risk_free_rate / 252)
        downside_returns = excess_ret[excess_ret < 0]
        
        downside_std = np.std(downside_returns)
        if downside_std < 1e-9: return 0.0
        
        return (np.mean(excess_ret) / downside_std) * np.sqrt(252)

    @staticmethod
    def probabilistic_sharpe_ratio(returns, benchmark_sharpe=0):
        returns = np.nan_to_num(returns)
        sr = FinancialMetrics.get_sharpe_ratio(returns)
        skew = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        n = len(returns)
        
        # Prevent math domain errors
        if n < 2: return 0.0
        denominator = n - 1
        
        sr_std = np.sqrt((1 + (0.5 * skew * sr) + ((kurtosis - 3) / 4) * sr**2) / denominator)
        
        if sr_std < 1e-9: return 0.0
        
        z_stat = (sr - benchmark_sharpe) / sr_std
        return stats.norm.cdf(z_stat)

# ==========================================
# 2. DATA ENGINEERING (UNCHANGED)
# ==========================================
def fetch_and_process_data(tickers, start_date="2015-01-01", end_date="2025-01-01"):
    print(f"Fetching data for {len(tickers)} assets...")
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    processed_frames = []
    
    for ticker in tickers:
        if len(tickers) == 1: df = data.copy()
        else:
            try: df = data[ticker].copy()
            except KeyError: continue
        if df.empty: continue
        
        # Stationarity & Normalization
        df['log_ret'] = np.log(df['Close'] / (df['Close'].shift(1) + 1e-8))
        df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi() / 100.0
        df['macd'] = ta.trend.MACD(df['Close']).macd_diff()
        bb = ta.volatility.BollingerBands(df['Close'])
        df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / (df['Close'] + 1e-8)
        df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        df['adv_20'] = df['Close'] * df['Volume'].rolling(window=20).mean()
        df['vol_ratio'] = (df['Close'] * df['Volume']) / (df['adv_20'] + 1e-8)
        
        df.dropna(inplace=True)
        cols_to_norm = ['log_ret', 'rsi', 'macd', 'bb_width', 'vol_ratio']
        for col in cols_to_norm:
            df[col] = (df[col] - df[col].rolling(60).mean()) / (df[col].rolling(60).std() + 1e-8)
            
        df['ticker'] = ticker
        processed_frames.append(df.dropna())

    combined = pd.concat(processed_frames)
    # Use dropna to ensure we don't have NaNs passing into the Env
    pivot_df = combined.pivot_table(index='Date', columns='ticker', values=['Close', 'log_ret', 'rsi', 'macd', 'bb_width', 'vol_ratio', 'atr', 'adv_20'])
    pivot_df.dropna(inplace=True)
    
    # Final check for infinities
    pivot_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    pivot_df.dropna(inplace=True)
    
    return pivot_df

# ==========================================
# 3. ENVIRONMENT (UNCHANGED)
# ==========================================
class IndianEquityEnv(gym.Env):
    def __init__(self, df, lookback_window=30, initial_balance=1_000_000):
        super(IndianEquityEnv, self).__init__()
        self.df = df
        self.tickers = df['Close'].columns.tolist()
        self.n_assets = len(self.tickers)
        self.lookback = lookback_window
        self.initial_balance = initial_balance
        self.reward_params = {'alpha': 0.001, 'beta': 0.1, 'gamma': 10.0}
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_assets + 1,), dtype=np.float32)
        self.obs_shape = (lookback_window, self.n_assets * 5)
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
        
        delta_weights = np.abs(weights - self.weights)
        turnover = np.sum(delta_weights)
        cost_linear = turnover * self.stt_brokerage
        
        advs = self.df['adv_20'].iloc[self.current_step].values
        trade_vals = delta_weights[1:] * self.portfolio_value
        slippage_penalty = self.reward_params['beta'] * np.sum((trade_vals / (advs + 1e-5))**2)
        
        net_ret = weighted_returns - cost_linear - (slippage_penalty / self.portfolio_value)
        downside_penalty = min(0, net_ret)**2 * self.reward_params['gamma']
        
        reward = net_ret - downside_penalty
        
        self.portfolio_value *= (1 + net_ret)
        self.weights = weights
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        
        return self._get_obs(), reward, terminated, False, {
            'portfolio_value': self.portfolio_value, 
            'net_return': net_ret
        }

# ==========================================
# 4. EXPLAINABLE TRANSFORMER (UNCHANGED)
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
# 5. WALK-FORWARD VALIDATION (UNCHANGED)
# ==========================================
class WalkForwardEvaluator:
    def __init__(self, df, train_window_days=750, test_window_days=250):
        self.df = df
        self.train_window = train_window_days
        self.test_window = test_window_days
        self.policy_kwargs = dict(
            features_extractor_class=ExplainableTransformer,
            features_extractor_kwargs=dict(features_dim=128),
        )

    def run(self):
        total_steps = len(self.df)
        current_start = 0
        oos_returns = []
        oos_portfolio_values = [1_000_000] 
        fold = 1
        while current_start + self.train_window + self.test_window <= total_steps:
            train_end = current_start + self.train_window
            test_end = train_end + self.test_window
            print(f"\n>>> FOLD {fold}: Train [{current_start}:{train_end}] | Test [{train_end}:{test_end}]")
            
            train_data = self.df.iloc[current_start:train_end]
            test_data = self.df.iloc[train_end:test_end]
            
            env_train = IndianEquityEnv(train_data)
            model = PPO("MlpPolicy", env_train, policy_kwargs=self.policy_kwargs, 
                        verbose=0, learning_rate=3e-4, ent_coef=0.01)
            model.learn(total_timesteps=8000) 
            
            current_balance = oos_portfolio_values[-1]
            env_test = IndianEquityEnv(test_data, initial_balance=current_balance)
            obs, _ = env_test.reset()
            done = False
            
            fold_returns = []
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env_test.step(action)
                done = terminated or truncated
                fold_returns.append(info['net_return'])
                oos_portfolio_values.append(info['portfolio_value'])
            
            oos_returns.extend(fold_returns)
            current_start += self.test_window
            fold += 1
            
        return np.array(oos_returns), np.array(oos_portfolio_values)

# ==========================================
# 6. MONTE CARLO BOOTSTRAP (FIXED)
# ==========================================
def bootstrap_validation(returns, n_simulations=1000):
    """
    Robust Permutation Test with Safe Handling for NaNs and Identical Values.
    """
    print(f"\nRunning {n_simulations} Monte Carlo Simulations (Permutation Test)...")
    
    # 1. Sanitize Data
    returns = np.nan_to_num(returns)
    if len(returns) == 0:
        print("Error: No returns data available.")
        return 0, np.zeros(n_simulations), 1.0

    real_sharpe = FinancialMetrics.get_sharpe_ratio(returns)
    random_sharpes = []
    
    # 2. Simulation Loop
    for _ in range(n_simulations):
        shuffled = np.random.permutation(returns)
        random_sharpes.append(FinancialMetrics.get_sharpe_ratio(shuffled))
        
    random_sharpes = np.array(random_sharpes)
    random_sharpes = np.nan_to_num(random_sharpes) # Clean results
    
    # 3. Calculate P-Value
    if np.std(random_sharpes) == 0:
        # If all random sharpes are identical (degenerate distribution),
        # p-value is 1.0 (fail to reject) or 0.0 (if we somehow beat it, unlikely)
        p_value = 1.0 if real_sharpe <= np.mean(random_sharpes) else 0.0
    else:
        p_value = np.sum(random_sharpes >= real_sharpe) / n_simulations
    
    return real_sharpe, random_sharpes, p_value

# ==========================================
# 7. MAIN EXECUTION
# ==========================================
def run_dissertation_experiment():
    # 1. Load Data
    TICKERS = ['RELIANCE.NS', 'INFY.NS', 'HDFCBANK.NS', 'TCS.NS']
    try:
        df_gym = fetch_and_process_data(TICKERS)
    except Exception as e:
        print(e); return

    # 2. Run Walk-Forward Validation
    print("--- Starting Walk-Forward Validation (Rigorous Backtesting) ---")
    wf_evaluator = WalkForwardEvaluator(df_gym)
    returns, equity_curve = wf_evaluator.run()
    
    # Check if we have enough data
    if len(returns) < 10:
        print("Not enough Out-of-Sample data generated. Check train/test window sizes.")
        return

    # 3. Compute Metrics
    print("\n--- Final Performance Metrics (Out-of-Sample) ---")
    sharpe = FinancialMetrics.get_sharpe_ratio(returns)
    sortino = FinancialMetrics.get_sortino_ratio(returns)
    max_dd = FinancialMetrics.get_max_drawdown(equity_curve)
    psr = FinancialMetrics.probabilistic_sharpe_ratio(returns)
    
    if len(equity_curve) > 0:
        total_ret = ((equity_curve[-1]/equity_curve[0])-1)*100
    else:
        total_ret = 0.0
        
    print(f"Total Return:       {total_ret:.2f}%")
    print(f"Sharpe Ratio:       {sharpe:.3f}")
    print(f"Sortino Ratio:      {sortino:.3f}")
    print(f"Max Drawdown:       {max_dd*100:.2f}%")
    print(f"Prob. Sharpe (PSR): {psr:.3f}")
    
    # 4. Monte Carlo Validation
    real_sr, random_srs, p_value = bootstrap_validation(returns)
    print(f"Monte Carlo P-Value: {p_value:.4f}")
    
    if p_value < 0.05:
        print(">> RESULT: Statistically Significant Alpha (Reject Null)")
    else:
        print(">> RESULT: Performance may be due to luck (Fail to Reject Null)")

    # 5. Visualizations (ROBUST)
    plt.figure(figsize=(12, 10))
    
    # Plot A: Equity Curve
    plt.subplot(2, 1, 1)
    plt.plot(equity_curve, label='Walk-Forward Equity (OOS)')
    plt.title("Out-of-Sample Walk-Forward Performance")
    plt.ylabel("Portfolio Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot B: Bootstrap Distribution (Fixed)
    plt.subplot(2, 1, 2)
    
    # Check variance to decide on KDE
    if np.std(random_srs) < 1e-6:
        print("Warning: Random Sharpe distribution has near-zero variance. Plotting standard histogram.")
        sns.histplot(random_srs, kde=False, color='gray', label='Random Skill Distribution', bins=30)
    else:
        # Standard KDE plot
        sns.histplot(random_srs, kde=True, color='gray', label='Random Skill Distribution')
        
    plt.axvline(real_sr, color='red', linestyle='--', linewidth=2, label=f'Agent SR ({real_sr:.2f})')
    plt.title(f"Survivorship Bias Check: Monte Carlo Permutation (p={p_value:.3f})")
    plt.xlabel("Sharpe Ratio")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_dissertation_experiment()