// Mock data generators for the Transformer-PPO Trading System

export interface TrainingMetrics {
  episode: number;
  avgReward: number;
  policyLoss: number;
  valueLoss: number;
  entropy: number;
  sharpeRatio: number;
  maxDrawdown: number;
}

export interface PortfolioWeight {
  symbol: string;
  weight: number;
  previousWeight: number;
  expectedReturn: number;
  confidence: number;
}

export interface PerformanceData {
  date: string;
  portfolioValue: number;
  nifty50: number;
  drawdown: number;
  sharpe: number;
}

// Generate training metrics showing convergence
export function generateTrainingData(episodes: number): TrainingMetrics[] {
  const data: TrainingMetrics[] = [];
  
  for (let i = 0; i < episodes; i++) {
    // Simulate learning curve with convergence
    const progress = i / episodes;
    const noise = (Math.random() - 0.5) * 0.1;
    
    data.push({
      episode: i + 1,
      avgReward: 0.0001 + progress * 0.0015 + noise * 0.0003,
      policyLoss: 0.5 * Math.exp(-progress * 2) + Math.random() * 0.05,
      valueLoss: 0.3 * Math.exp(-progress * 1.5) + Math.random() * 0.03,
      entropy: 0.8 - progress * 0.4 + Math.random() * 0.1,
      sharpeRatio: 0.5 + progress * 1.5 + Math.random() * 0.2,
      maxDrawdown: -0.05 - progress * 0.1 + Math.random() * 0.02
    });
  }
  
  return data;
}

// Generate optimal portfolio weights
export function generatePortfolioWeights(stocks: string[]): PortfolioWeight[] {
  const weights: PortfolioWeight[] = [];
  const numStocks = stocks.length;
  
  // Generate random weights that sum to 1 (softmax-like distribution)
  const rawWeights = stocks.map(() => Math.random() * Math.random()); // Square for concentration
  const sum = rawWeights.reduce((a, b) => a + b, 0);
  
  stocks.forEach((symbol, i) => {
    const currentWeight = rawWeights[i] / sum;
    const previousWeight = currentWeight * (0.8 + Math.random() * 0.4); // Slight variation
    
    weights.push({
      symbol,
      weight: currentWeight,
      previousWeight: previousWeight,
      expectedReturn: (Math.random() - 0.4) * 0.15, // -6% to +9% expected
      confidence: 0.5 + Math.random() * 0.4 // 50-90% confidence
    });
  });
  
  return weights;
}

// Generate backtest performance data
export function generatePerformanceData(days: number = 365): PerformanceData[] {
  const data: PerformanceData[] = [];
  const startDate = new Date('2020-01-01');
  
  let portfolioValue = 100000;
  let niftyValue = 100000;
  let peak = portfolioValue;
  
  for (let i = 0; i < days; i++) {
    const date = new Date(startDate);
    date.setDate(date.getDate() + i);
    
    // Portfolio: better performance with higher volatility
    const portfolioReturn = (Math.random() - 0.45) * 0.02; // Slight positive bias
    portfolioValue *= (1 + portfolioReturn);
    
    // Nifty: steady growth
    const niftyReturn = (Math.random() - 0.48) * 0.015; // More stable
    niftyValue *= (1 + niftyReturn);
    
    // Track drawdown
    if (portfolioValue > peak) peak = portfolioValue;
    const drawdown = ((portfolioValue - peak) / peak) * 100;
    
    // Calculate rolling Sharpe (simplified)
    const sharpe = 1.2 + Math.random() * 0.8 + (i / days) * 0.5;
    
    data.push({
      date: date.toISOString().split('T')[0],
      portfolioValue,
      nifty50: niftyValue,
      drawdown,
      sharpe
    });
  }
  
  return data;
}

// Generate time series for a single episode showing state transitions
export function generateEpisodeStateData(steps: number = 252) {
  const data = [];
  
  for (let i = 0; i < steps; i++) {
    data.push({
      step: i,
      reward: (Math.random() - 0.5) * 0.003,
      portfolioReturn: (Math.random() - 0.48) * 0.02,
      turnoverCost: Math.random() * 0.0005,
      slippageCost: Math.random() * 0.0003,
      netReward: (Math.random() - 0.5) * 0.002
    });
  }
  
  return data;
}

// Explanation text for DRL concepts
export const DRL_EXPLANATIONS = {
  transformer: `The Transformer Encoder processes the 60-day rolling window of market data using multi-head self-attention. 
    This allows the model to capture long-range temporal dependencies and identify market regime changes that traditional 
    LSTMs struggle with. The attention mechanism learns which historical time steps are most relevant for predicting 
    optimal portfolio weights.`,
  
  ppo: `Proximal Policy Optimization (PPO) is the reinforcement learning algorithm that trains the agent. The Actor network 
    outputs portfolio weights, while the Critic estimates the expected future returns. PPO uses a clipped objective function 
    to ensure stable learning by preventing drastic policy changes between updates.`,
  
  reward: `The reward function is critical: r_t = Portfolio_Return - Turnover_Cost - Slippage_Penalty. 
    Turnover cost models STT and brokerage as a linear function of portfolio rebalancing. 
    Slippage is modeled quadratically to heavily penalize large trades in illiquid stocks. 
    This incentivizes the agent to find a balance between returns and transaction costs.`,
  
  state: `The state space combines multiple data sources: 
    - Price/Volume data (OHLCV) from NSE/BSE via Yahoo Finance
    - Technical indicators (RSI, MACD, Bollinger Bands) 
    - Macro features (G-Sec yields, USD/INR, CPI) forward-filled from monthly to daily
    - Sentiment proxies (Google Trends for retail interest)
    - Impact cost estimates (trade size relative to daily volume)`,
  
  inference: `Real-time inference follows this pipeline:
    1. Fetch live market data at market open (9:15 AM IST)
    2. Apply same preprocessing (StandardScaler with saved parameters from training)
    3. Create 60-day rolling window
    4. Feed through Transformer encoder
    5. Actor network outputs softmax weights
    6. Weights are used for intraday rebalancing
    Latency is ~200ms on CPU, <50ms on GPU.`
};
