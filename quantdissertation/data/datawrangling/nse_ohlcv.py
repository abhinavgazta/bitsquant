# data_ingestion/nse_ohlcv.py
import pandas as pd

def load_nse_impact_cost(file_path):
    """
    NSE impact cost CSV (manually downloaded)
    """
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower()
    df['date'] = pd.to_datetime(df['date'])
    return df[['date', 'symbol', 'impact_cost']]
