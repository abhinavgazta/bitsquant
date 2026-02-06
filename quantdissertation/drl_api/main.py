from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np

from stable_baselines3 import PPO

from environment import IndianEquityEnv
from model import ExplainableTransformer
from utils import fetch_and_process_data

app = FastAPI()

class Tickers(BaseModel):
    tickers: List[str]

@app.get("/")
def read_root():
    return {"Hello": "DRL API"}

@app.post("/predict")
def predict(tickers: Tickers):
    """
    Takes a list of tickers, fetches data, trains a model (for demo), 
    and returns the predicted portfolio allocation.
    """
    try:
        # 1. Fetch and process data
        df_gym = fetch_and_process_data(tickers.tickers)
        if len(df_gym) < 100:
            return {"error": "Not enough data to process. Try a longer date range or different tickers."}

        # 2. Set up environment
        env = IndianEquityEnv(df_gym)

        # 3. Define and train model (short training for demonstration)
        # In a real scenario, you would load a pre-trained model here.
        policy_kwargs = dict(
            features_extractor_class=ExplainableTransformer,
            features_extractor_kwargs=dict(features_dim=128),
        )
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=0)
        model.learn(total_timesteps=100) # Very short training

        # 4. Get the latest observation and predict
        obs, _ = env.reset()
        # The environment resets to the start, so we need to step to the end 
        # to get the *latest* observation for a *current* prediction.
        for i in range(env.lookback, len(env.df) - 2):
            # We don't need to do anything with the output here, just advance the environment
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _, _ = env.step(action)
        
        # Now predict the action for the current state
        action, _ = model.predict(obs, deterministic=True)

        # 5. Get weights from the action
        exp_action = np.exp(action)
        weights = exp_action / np.sum(exp_action)

        # 6. Format the output
        allocation = {"cash": weights[0]}
        for i, ticker in enumerate(tickers.tickers):
            allocation[ticker] = weights[i+1]

        return {"allocation": allocation}

    except Exception as e:
        return {"error": str(e)}