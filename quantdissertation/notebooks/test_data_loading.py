#!/usr/bin/env python3
"""
Diagnostic script to test data loading from CSV files.
Run this to verify the data loading pipeline works correctly.
"""

import os
import glob
import pandas as pd
import numpy as np
import ta
from pathlib import Path

# Configuration
DATA_DIR = '/Users/abhinavgazta/Downloads/bits/bitsquant/quantdissertation/data/raw/ohlc-data-10yrs'

print("=" * 70)
print("DATA LOADING DIAGNOSTIC SCRIPT")
print("=" * 70)

# Step 1: Check directory exists
print(f"\n1. Checking data directory: {DATA_DIR}")
if not os.path.exists(DATA_DIR):
    print(f"   ✗ Directory NOT found")
    exit(1)
print(f"   ✓ Directory found")

# Step 2: List CSV files
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
print(f"\n2. Found {len(csv_files)} CSV files")
if len(csv_files) == 0:
    print("   ✗ No CSV files found!")
    exit(1)
print(f"   ✓ Sample files: {[os.path.basename(f) for f in csv_files[:3]]}")

# Step 3: Test loading a single CSV file
print(f"\n3. Testing CSV file loading...")
test_file = csv_files[0]
ticker_name = os.path.basename(test_file).replace('.csv', '')
print(f"   Testing file: {ticker_name}")

try:
    df_raw = pd.read_csv(test_file)
    print(f"   ✓ File loaded successfully")
    print(f"   Shape: {df_raw.shape}")
    print(f"   Columns: {df_raw.columns.tolist()}")
    print(f"   First row:\n{df_raw.iloc[0]}")
except Exception as e:
    print(f"   ✗ Failed to load file: {e}")
    exit(1)

# Step 4: Test data processing
print(f"\n4. Testing data processing for {ticker_name}...")
try:
    # Handle column names
    df = pd.read_csv(test_file)
    df.columns = df.columns.str.strip().str.lower()
    
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        print(f"   ✗ Missing required columns")
        print(f"   Available: {df.columns.tolist()}")
        exit(1)
    
    # Select columns
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    # Parse dates
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.set_index('Date').sort_index()
    
    # Convert to numeric
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN prices
    df = df.dropna(subset=['Close', 'High', 'Low', 'Open', 'Volume'])
    
    print(f"   ✓ Data cleaned: {len(df)} rows")
    print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
    
except Exception as e:
    print(f"   ✗ Data processing failed: {e}")
    exit(1)

# Step 5: Test feature engineering
print(f"\n5. Testing feature engineering...")
try:
    # Log returns
    df['log_ret'] = np.log(df['Close'] / (df['Close'].shift(1) + 1e-8))
    print(f"   ✓ Log returns calculated")
    
    # RSI & MACD
    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi() / 100.0
    macd = ta.trend.MACD(df['Close'])
    df['macd'] = macd.macd_diff()
    print(f"   ✓ RSI & MACD calculated")
    
    # Volatility
    bb = ta.volatility.BollingerBands(df['Close'])
    df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / (df['Close'] + 1e-8)
    df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    print(f"   ✓ Bollinger Bands & ATR calculated")
    
    # Liquidity
    df['adv_20'] = df['Close'] * df['Volume'].rolling(window=20).mean()
    df['vol_ratio'] = (df['Close'] * df['Volume']) / (df['adv_20'] + 1e-8)
    print(f"   ✓ Liquidity indicators calculated")
    
    # Drop NaN from initial calculations
    df = df.dropna(subset=['log_ret', 'rsi', 'macd', 'bb_width', 'atr', 'adv_20', 'vol_ratio'])
    print(f"   ✓ After dropna: {len(df)} rows")
    
    if len(df) < 60:
        print(f"   ⚠ Warning: Less than 60 rows remaining")
    
except Exception as e:
    print(f"   ✗ Feature engineering failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 6: Test normalization
print(f"\n6. Testing normalization...")
try:
    cols_to_norm = ['log_ret', 'rsi', 'macd', 'bb_width', 'vol_ratio']
    for col in cols_to_norm:
        rolling_mean = df[col].rolling(60).mean()
        rolling_std = df[col].rolling(60).std()
        rolling_std = rolling_std.replace(0, 1e-8)
        df[col] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
        df[col] = df[col].bfill().ffill()
    
    # Replace infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.bfill().ffill()
    
    print(f"   ✓ Normalization complete: {len(df)} rows remaining")
    
except Exception as e:
    print(f"   ✗ Normalization failed: {e}")
    exit(1)

# Step 7: Test pivoting with multiple stocks
print(f"\n7. Testing full pipeline with all stocks...")
try:
    processed_frames = []
    successful = 0
    failed = 0
    
    for csv_file in csv_files[:5]:  # Test first 5 files
        ticker_name = os.path.basename(csv_file).replace('.csv', '')
        
        try:
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip().str.lower()
            
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                failed += 1
                continue
            
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
            df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            df = df.set_index('Date').sort_index()
            
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=['Close', 'High', 'Low', 'Open', 'Volume'])
            
            if len(df) < 60:
                failed += 1
                continue
            
            # Features
            df['log_ret'] = np.log(df['Close'] / (df['Close'].shift(1) + 1e-8))
            df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi() / 100.0
            macd = ta.trend.MACD(df['Close'])
            df['macd'] = macd.macd_diff()
            bb = ta.volatility.BollingerBands(df['Close'])
            df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / (df['Close'] + 1e-8)
            df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
            df['adv_20'] = df['Close'] * df['Volume'].rolling(window=20).mean()
            df['vol_ratio'] = (df['Close'] * df['Volume']) / (df['adv_20'] + 1e-8)
            
            df = df.dropna(subset=['log_ret', 'rsi', 'macd', 'bb_width', 'atr', 'adv_20', 'vol_ratio'])
            
            # Normalize
            cols_to_norm = ['log_ret', 'rsi', 'macd', 'bb_width', 'vol_ratio']
            for col in cols_to_norm:
                rolling_mean = df[col].rolling(60).mean()
                rolling_std = df[col].rolling(60).std()
                rolling_std = rolling_std.replace(0, 1e-8)
                df[col] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
                df[col] = df[col].bfill().ffill()
            
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.bfill().ffill()
            
            df['ticker'] = ticker_name
            processed_frames.append(df)
            successful += 1
            print(f"   ✓ {ticker_name}: {len(df)} rows")
            
        except Exception as e:
            failed += 1
            print(f"   ✗ {ticker_name}: {str(e)[:50]}")
    
    print(f"\n   Summary: {successful} successful, {failed} failed")
    
    if not processed_frames:
        print(f"   ✗ No successfully processed stocks!")
        exit(1)
    
    combined = pd.concat(processed_frames)
    print(f"   ✓ Combined {len(processed_frames)} stocks, {len(combined)} total rows")
    
    # Pivot
    combined_reset = combined.reset_index()
    pivot_df = combined_reset.pivot_table(
        index='Date',
        columns='ticker',
        values=['Close', 'log_ret', 'rsi', 'macd', 'bb_width', 'vol_ratio', 'atr', 'adv_20'],
        aggfunc='first'
    )
    
    print(f"   ✓ Pivoted shape: {pivot_df.shape}")
    print(f"   ✓ Date range: {pivot_df.index[0].date()} to {pivot_df.index[-1].date()}")
    
    if pivot_df.empty:
        print(f"   ✗ Pivoted DataFrame is empty!")
        exit(1)
    
except Exception as e:
    print(f"   ✗ Full pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED - Data loading should work correctly!")
print("=" * 70)
