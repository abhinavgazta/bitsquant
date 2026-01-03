# data_ingestion/yfinance_fetch.py
import yfinance as yf
import pandas as pd

def fetch_yfinance_data(tickers, start, end):
    """
    tickers: ['RELIANCE.NS', 'TCS.NS']
    """
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        group_by='ticker',
        threads=True
    )

    if isinstance(data.columns, pd.MultiIndex):
        out = []
        for ticker in tickers:
            df = data[ticker].copy()
            df['ticker'] = ticker
            out.append(df.reset_index())
        return pd.concat(out, ignore_index=True)

    data = data.reset_index()
    data['ticker'] = tickers[0]
    return data


fetch_yfinance_data('RELIANCE.NS', '01012026', '02012026')