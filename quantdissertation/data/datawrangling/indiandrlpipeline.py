"""
Indian Equity Market Data Pipeline (2013–2025)

Purpose:
---------
Unified data ingestion pipeline for Deep Reinforcement Learning based
portfolio management research on Indian equity markets (NSE/BSE).

Author: Abhinav Gazta
Context: M.Tech (AI & ML) Dissertation – PPO + Transformer Agent

Data Sources:
--------------
- Yahoo Finance (OHLCV)
- Kaggle (NIFTY datasets – optional local load)
- RBI DBIE (macro indicators – CSV)
- FRED (FX, CPI)
- Google Trends (retail sentiment proxy)
- NSE Impact Cost (manual CSV)
- StockEdge (manual CSV)

NOTE:
-----
NSE India & StockEdge do NOT provide stable public APIs.
For academic compliance:
- Download CSVs manually
- Place them in ./data/manual/
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from fredapi import Fred
from pytrends.request import TrendReq
import pyarrow as pa
import pyarrow.parquet as pq

# =========================
# CONFIGURATION
# =========================

START_DATE = "2013-01-01"
END_DATE = "2025-12-31"

TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS"
]

DATA_DIR = "data"
RAW_DIR = f"{DATA_DIR}/raw"
PROCESSED_DIR = f"{DATA_DIR}/processed"
MANUAL_DIR = f"{DATA_DIR}/manual"

FRED_API_KEY = "PUT_YOUR_FRED_API_KEY_HERE"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MANUAL_DIR, exist_ok=True)


# =========================
# UTILS
# =========================

def save_parquet(df, path):
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)


def normalize_dates(df, col="date"):
    df[col] = pd.to_datetime(df[col])
    return df.sort_values(col)


# =========================
# 1. YAHOO FINANCE (OHLCV)
# =========================

def fetch_yahoo_ohlcv(tickers):
    print("Fetching Yahoo Finance OHLCV...")
    data = yf.download(
        tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        group_by="ticker",
        threads=True
    )

    frames = []
    for t in tickers:
        df = data[t].copy()
        df.reset_index(inplace=True)
        df["ticker"] = t.replace(".NS", "")
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out.columns = [c.lower() for c in out.columns]
    return out


# =========================
# 2. KAGGLE (OPTIONAL)
# =========================

def load_kaggle_csv(path):
    print("Loading Kaggle dataset...")
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df.rename(columns={"symbol": "ticker"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


# =========================
# 3. RBI DBIE (MACRO)
# =========================

def load_rbi_dbie(csv_path):
    print("Loading RBI DBIE macro data...")
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["Date"])
    df = df.rename(columns={
        "10-Year G-Sec Yield": "gsec_10y",
        "CPI Inflation": "cpi"
    })
    return df[["date", "gsec_10y", "cpi"]].sort_values("date")


# =========================
# 4. FRED (FX, CPI)
# =========================

def fetch_fred_data():
    print("Fetching FRED macro data...")
    fred = Fred(api_key=FRED_API_KEY)

    usd_inr = fred.get_series("DEXINUS")
    cpi = fred.get_series("INDCPIALLMINMEI")

    df = pd.concat([
        usd_inr.rename("usd_inr"),
        cpi.rename("cpi_fred")
    ], axis=1)

    df.index = pd.to_datetime(df.index)
    df = df[df.index >= START_DATE]
    df.reset_index(inplace=True)
    df.rename(columns={"index": "date"}, inplace=True)

    return df


# =========================
# 5. GOOGLE TRENDS
# =========================

def fetch_google_trends(keyword="Nifty 50"):
    print(f"Fetching Google Trends for '{keyword}'...")
    pytrends = TrendReq(hl="en-IN", tz=330)
    pytrends.build_payload(
        [keyword],
        timeframe=f"{START_DATE} {END_DATE}",
        geo="IN"
    )
    df = pytrends.interest_over_time()
    df.drop(columns=["isPartial"], inplace=True)
    df.columns = ["trend_score"]
    df.reset_index(inplace=True)
    df.rename(columns={"date": "date"}, inplace=True)
    return df


# =========================
# 6. NSE IMPACT COST (MANUAL)
# =========================

def load_nse_impact_cost(path):
    print("Loading NSE Impact Cost (manual)...")
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "symbol", "impact_cost"]]


# =========================
# 7. STOCKEDGE (MANUAL)
# =========================

def load_stockedge(path):
    print("Loading StockEdge data (manual)...")
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "fii_flow", "dii_flow", "delivery_pct"]]


# =========================
# 8. FEATURE ALIGNMENT
# =========================

def align_features(dfs):
    print("Aligning features to daily calendar...")
    base = dfs[0].set_index("date")
    for df in dfs[1:]:
        base = base.join(df.set_index("date"), how="left")
    return base.ffill().reset_index()


# =========================
# MAIN PIPELINE
# =========================

def run_pipeline():
    prices = fetch_yahoo_ohlcv(TICKERS)
    save_parquet(prices, f"{PROCESSED_DIR}/ohlcv.parquet")

    trends = fetch_google_trends()
    save_parquet(trends, f"{PROCESSED_DIR}/google_trends.parquet")

    fred_macro = fetch_fred_data()
    save_parquet(fred_macro, f"{PROCESSED_DIR}/fred_macro.parquet")

    print("\nPipeline completed successfully.")
    print("Manual datasets expected in ./data/manual/")
    print("- NSE Impact Cost CSV")
    print("- StockEdge FII/DII CSV")


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    run_pipeline()
