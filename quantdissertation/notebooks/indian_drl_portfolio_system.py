# ============================================================
# Indian Equity DRL Portfolio Management System
# Step 1 & 2 Implementation
# ============================================================

import os
import glob
import numpy as np
import pandas as pd
from typing import List, Dict

import gym
from gym import spaces

import torch
import torch.nn as nn
import torch.optim as optim

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

START_DATE = "2013-01-01"
END_DATE = "2025-12-31"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Transaction cost constants (India specific)
STT_RATE = 0.001        # 0.1%
EXCHANGE_FEES = 0.0001
SLIPPAGE_MULTIPLIER = 1.0

# ------------------------------------------------------------
# STEP 2: DATA ENGINEERING PIPELINE
# ------------------------------------------------------------

class IndianMarketDataLoader:
    """
    Loads and aligns Indian equity, macro, and sentiment data
    into a single research-grade dataframe.
    """

    def __init__(self, base_path: str):
        self.base_path = base_path

    def load_equities(self) -> Dict[str, pd.DataFrame]:
        equity_files = glob.glob(f"{self.base_path}/equities/**/*.csv", recursive=True)
        equity_data = {}

        for file in equity_files:
            symbol = os.path.basename(file).replace(".csv", "")
            df = pd.read_csv(file, parse_dates=["Date"])
            df = df.set_index("Date").sort_index()
            equity_data[symbol] = df

        return equity_data

    def load_liquidity(self) -> pd.DataFrame:
        df = pd.read_csv(
            f"{self.base_path}/liquidity/impact_cost.csv",
            parse_dates=["Date"]
        )
        return df.set_index("Date").sort_index()

    def load_macro(self) -> pd.DataFrame:
        macro_files = glob.glob(f"{self.base_path}/macro/*.csv")
        macro_dfs = []

        for file in macro_files:
            df = pd.read_csv(file, parse_dates=["Date"])
            df = df.set_index("Date").sort_index()
            macro_dfs.append(df)

        macro = pd.concat(macro_dfs, axis=1)
        return macro.ffill()

    def load_sentiment(self) -> pd.DataFrame:
        df = pd.read_csv(
            f"{self.base_path}/sentiment/google_trends_nifty.csv",
            parse_dates=["Date"]
        )
        return df.set_index("Date").sort_index().ffill()

    def build_feature_tensor(self):
        equities = self.load_equities()
        macro = self.load_macro()
        sentiment = self.load_sentiment()

        common_dates = macro.index.intersection(sentiment.index)

        for sym, df in equities.items():
            common_dates = common_dates.intersection(df.index)

        X_prices = []
        for sym, df in equities.items():
            features = df.loc[common_dates][
                ["Open", "High", "Low", "Close", "Volume"]
            ]
            X_prices.append(features.values)

        X_prices = np.stack(X_prices, axis=1)
        X_macro = macro.loc[common_dates].values
        X_sent = sentiment.loc[common_dates].values

        return {
            "dates": common_dates,
            "prices": X_prices,
            "macro": X_macro,
            "sentiment": X_sent,
            "symbols": list(equities.keys())
        }

# ------------------------------------------------------------
# STEP 1: DRL FRAMEWORK DESIGN
# ------------------------------------------------------------

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, model_dim=128, num_heads=4, num_layers=2):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        x = self.input_proj(x)
        return self.encoder(x)[:, -1, :]


class PPOPolicy(nn.Module):
    def __init__(self, state_dim, n_assets):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_assets),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)


class PortfolioEnv(gym.Env):
    """
    Cost-aware Indian equity portfolio environment.
    """

    def __init__(self, data, window=30):
        super().__init__()

        self.prices = data["prices"]
        self.macro = data["macro"]
        self.sent = data["sentiment"]

        self.window = window
        self.t = window
        self.n_assets = self.prices.shape[1]

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window, self.n_assets * 5 + self.macro.shape[1] + self.sent.shape[1])
        )

        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_assets,)
        )

    def reset(self):
        self.t = self.window
        self.weights = np.ones(self.n_assets) / self.n_assets
        return self._get_obs()

    def _get_obs(self):
        price_window = self.prices[self.t - self.window:self.t].reshape(self.window, -1)
        macro = np.repeat(self.macro[self.t][None, :], self.window, axis=0)
        sent = np.repeat(self.sent[self.t][None, :], self.window, axis=0)
        return np.concatenate([price_window, macro, sent], axis=1)

    def step(self, action):
        action = action / action.sum()

        prev_prices = self.prices[self.t - 1][:, 3]
        curr_prices = self.prices[self.t][:, 3]

        returns = (curr_prices - prev_prices) / prev_prices
        portfolio_return = np.dot(self.weights, returns)

        turnover = np.sum(np.abs(action - self.weights))
        cost = turnover * (STT_RATE + EXCHANGE_FEES)

        reward = portfolio_return - cost

        self.weights = action
        self.t += 1

        done = self.t >= len(self.prices) - 1
        return self._get_obs(), reward, done, {}

# ------------------------------------------------------------
# MAIN (Research Entry Point)
# ------------------------------------------------------------

if __name__ == "__main__":
    loader = IndianMarketDataLoader(base_path="data")
    data = loader.build_feature_tensor()

    env = PortfolioEnv(data)

    obs = env.reset()
    print("State shape:", obs.shape)
    print("Assets:", len(data["symbols"]))

