import gymnasium as gym
from gymnasium import spaces
import numpy as np

class IndianEquityEnv(gym.Env):
    def __init__(self, df, reward_params=None, lookback_window=30, initial_balance=1_000_000):
        super(IndianEquityEnv, self).__init__()
        self.df = df
        self.tickers = df['Close'].columns.tolist()
        self.n_assets = len(self.tickers)
        self.lookback = lookback_window
        self.initial_balance = initial_balance
        
        # Tunable Reward Penalties
        self.reward_params = reward_params if reward_params else {
            'alpha': 0.001,  # Turnover penalty
            'beta': 0.1,     # Slippage penalty
            'gamma': 10.0    # Downside risk penalty
        }
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_assets + 1,), dtype=np.float32)
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
        # Softmax normalization to get weights
        exp_action = np.exp(action)
        weights = exp_action / np.sum(exp_action)
        
        current_prices = self.df['Close'].iloc[self.current_step].values
        next_prices = self.df['Close'].iloc[self.current_step + 1].values
        asset_returns = (next_prices - current_prices) / current_prices
        
        weighted_returns = np.dot(weights[1:], asset_returns)
        
        # Cost Calculation
        delta_weights = np.abs(weights - self.weights)
        turnover = np.sum(delta_weights)
        cost_linear = turnover * self.stt_brokerage
        
        advs = self.df['adv_20'].iloc[self.current_step].values
        trade_vals = delta_weights[1:] * self.portfolio_value
        slippage_penalty = self.reward_params['beta'] * np.sum((trade_vals / (advs + 1e-5))**2)
        
        net_ret = weighted_returns - cost_linear - (slippage_penalty / self.portfolio_value)
        
        # Downside Risk Penalty
        downside_penalty = min(0, net_ret)**2 * self.reward_params['gamma']
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
