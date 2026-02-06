import pandas as pd
import numpy as np
import yfinance as yf
import ta
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import shap

def fetch_and_process_data(tickers, start_date="2015-01-01", end_date="2025-01-01"):
    print(f"Fetching data for {len(tickers)} assets...")
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    processed_frames = []
    
    for ticker in tickers:
        if len(tickers) == 1:
            df = data.copy()
        else:
            try:
                df = data[ticker].copy()
            except KeyError:
                continue
                
        if df.empty: continue
        
        # --- Feature Engineering ---
        # 1. Log Returns: Stationarity
        df['log_ret'] = np.log(df['Close'] / (df['Close'].shift(1) + 1e-8))
        
        # 2. Momentum & Trend
        df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi() / 100.0
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd_diff()
        
        # 3. Volatility
        bb = ta.volatility.BollingerBands(df['Close'])
        df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / (df['Close'] + 1e-8)
        df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        
        # 4. Liquidity / Cost Proxies
        df['adv_20'] = df['Close'] * df['Volume'].rolling(window=20).mean()
        df['vol_ratio'] = (df['Close'] * df['Volume']) / (df['adv_20'] + 1e-8)
        
        df.dropna(inplace=True)
        
        # 5. Normalization (Rolling Z-Score)
        cols_to_norm = ['log_ret', 'rsi', 'macd', 'bb_width', 'vol_ratio']
        for col in cols_to_norm:
            rolling_mean = df[col].rolling(60).mean()
            rolling_std = df[col].rolling(60).std()
            df[col] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
            
        df['ticker'] = ticker
        processed_frames.append(df.dropna())

    if not processed_frames:
        raise ValueError("No data downloaded.")
        
    combined = pd.concat(processed_frames)
    pivot_df = combined.pivot_table(index='Date', columns='ticker', values=['Close', 'log_ret', 'rsi', 'macd', 'bb_width', 'vol_ratio', 'atr', 'adv_20'])
    
    # Clean Infinite/NaN values
    pivot_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    pivot_df.dropna(inplace=True)
    
    return pivot_df

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
        
        if std_dev < 1e-9: return 0.0
        
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
        
        if n < 2: return 0.0
        
        sr_std = np.sqrt((1 + (0.5 * skew * sr) + ((kurtosis - 3) / 4) * sr**2) / (n - 1))
        if sr_std < 1e-9: return 0.0
        
        z_stat = (sr - benchmark_sharpe) / sr_std
        return stats.norm.cdf(z_stat)

def analyze_agent(model, env, obs_history, turnover_history, atr_history):
    # --- Analysis 1: Cost Awareness ---
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=atr_history, y=turnover_history, alpha=0.6)
    if len(atr_history) > 1:
        z = np.polyfit(atr_history, turnover_history, 1)
        p = np.poly1d(z)
        plt.plot(atr_history, p(atr_history), "r--", label='Trend')
    plt.title("Dynamic Execution: Turnover vs Volatility (ATR)")
    plt.xlabel("Market Volatility")
    plt.ylabel("Portfolio Turnover")
    plt.legend()
    plt.show()
    
    # --- Analysis 2: Grouped SHAP ---
    ticker = getattr(env, 'tickers', None)
    ticker_label = None
    if isinstance(ticker, (list, tuple)) and len(ticker) > 0:
        ticker_label = ticker[0]

    print("Calculating SHAP Values (this may take time)...")
    X_3d = np.array(obs_history[:50]) # Use subset for speed
    X_2d = X_3d.reshape(X_3d.shape[0], -1)

    def predict_wrapper(X_flattened):
        batch_size = X_flattened.shape[0]
        lookback = X_3d.shape[1]
        features = X_3d.shape[2]
        X_restored = X_flattened.reshape(batch_size, lookback, features)
        X_torch = torch.as_tensor(X_restored).to(model.device)
        with torch.no_grad():
            return model.policy.get_distribution(X_torch).mode().cpu().numpy()

    explainer = shap.KernelExplainer(predict_wrapper, X_2d[:5])
    shap_values = explainer.shap_values(X_2d[5:15])

    if isinstance(shap_values, list):
        shap_matrix = np.sum([np.abs(sv) for sv in shap_values], axis=0)
    elif len(shap_values.shape) == 3:
        shap_matrix = np.sum(np.abs(shap_values), axis=2)
    else:
        shap_matrix = np.abs(shap_values)

    mean_shap = np.mean(shap_matrix, axis=0)

    feature_groups = {
        "Momentum": ["rsi", "macd"],
        "Volatility": ["bb_width", "atr"],
        "Liquidity": ["vol_ratio", "adv_20"],
        "Returns": ["log_ret"]
    }

    raw_feature_names = ['log_ret', 'rsi', 'macd', 'bb_width', 'vol_ratio']
    group_scores = {k: 0.0 for k in feature_groups}
    n_features = len(raw_feature_names)

    for i, imp in enumerate(mean_shap):
        feat_type_idx = (i // 30) % n_features 
        feat_name = raw_feature_names[feat_type_idx]
        for group, keywords in feature_groups.items():
            if feat_name in keywords:
                group_scores[group] += float(imp)

    positive_score = group_scores.get('Returns', 0.0) + group_scores.get('Momentum', 0.0)
    negative_score = group_scores.get('Volatility', 0.0) + group_scores.get('Liquidity', 0.0)
    rec = 'HOLD'
    if positive_score > negative_score * 1.05:
        rec = 'BUY'
    elif negative_score > positive_score * 1.05:
        rec = 'SELL'

    plt.figure(figsize=(8, 5))
    bars = plt.bar(list(group_scores.keys()), list(group_scores.values()), color=['#3498db', '#e74c3c', '#f1c40f', '#2ecc71'])
    plt.title("Policy Drivers: Feature Importance Grouping")
    plt.ylabel("Mean |SHAP|")

    overlay_text = ''
    if ticker_label:
        overlay_text += f"Ticker: {ticker_label}\n"
    overlay_text += f"Recommendation: {rec}"

    plt.gca().text(0.98, 0.95, overlay_text, horizontalalignment='right', verticalalignment='top',
                   transform=plt.gca().transAxes, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round'))

    if rec == 'BUY':
        color = 'green'
    elif rec == 'SELL':
        color = 'red'
    else:
        color = 'black'
    plt.gca().text(0.98, 0.83, rec, horizontalalignment='right', verticalalignment='top',
                   transform=plt.gca().transAxes, fontsize=12, fontweight='bold', color=color)

    plt.show()