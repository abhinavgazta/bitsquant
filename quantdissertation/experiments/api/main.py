import uvicorn
from fastapi import FastAPI, HTTPException, Query
import yfinance as yf
import pandas as pd
from datetime import datetime

# --- THE ZSCALER BYPASS ---
from curl_cffi import requests as cf_requests


def get_zscaler_bypass_session():
    """
    Creates a curl_cffi session that mimics a real Chrome browser
    and explicitly disables SSL verification to bypass Zscaler/Proxy errors.
    """
    session = cf_requests.Session(impersonate="chrome")
    session.verify = False  # <--- This is the key line
    return session


# ---------------------------

app = FastAPI(
    title="Indian Stock OHLCV API",
    description="Fetch adjusted daily OHLCV data for Indian stocks using Yahoo Finance.",
    version="1.0.0"
)


def format_stock_data(df: pd.DataFrame):
    if df.empty:
        return []
    df.reset_index(inplace=True)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df = df.round(2)
    return df.to_dict(orient="records")


@app.get("/latest/{symbol}")
def get_latest_stock_data(symbol: str):
    try:
        symbol = symbol.upper()

        # Pass the Zscaler-proof session to the Ticker
        session = get_zscaler_bypass_session()
        ticker = yf.Ticker(symbol, session=session)

        df = ticker.history(period="1d", auto_adjust=True)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}. Check .NS/.BO suffix.")

        data = format_stock_data(df)
        return {"symbol": symbol, "data": data}

    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        # Return the actual error to the API response for easier debugging
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{symbol}")
def get_stock_history(
        symbol: str,
        start_date: str = Query(..., description="Start date YYYY-MM-DD"),
        end_date: str = Query(..., description="End date YYYY-MM-DD")
):
    try:
        symbol = symbol.upper()

        # Reuse the session generator
        session = get_zscaler_bypass_session()
        ticker = yf.Ticker(symbol, session=session)

        df = ticker.history(start=start_date, end=end_date, auto_adjust=True)

        if df.empty:
            raise HTTPException(status_code=404, detail="No data found for the given range.")

        data = format_stock_data(df)
        return {"symbol": symbol, "count": len(data), "data": data}

    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Remove the SSL context override from previous attempts (it won't help here)
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)