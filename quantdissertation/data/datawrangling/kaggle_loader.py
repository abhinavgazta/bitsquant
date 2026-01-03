# data_ingestion/kaggle_loader.py
import pandas as pd

def load_kaggle_nifty(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df.columns = df.columns.str.lower()
    df = df.rename(columns={'symbol': 'ticker'})
    return df
