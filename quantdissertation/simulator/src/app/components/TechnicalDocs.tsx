import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Badge } from './ui/badge';

export function TechnicalDocs() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Technical Implementation Guide</CardTitle>
        <CardDescription>
          Python code structure and production deployment guidelines
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="structure">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="structure">Structure</TabsTrigger>
            <TabsTrigger value="transformer">Transformer</TabsTrigger>
            <TabsTrigger value="ppo">PPO Agent</TabsTrigger>
            <TabsTrigger value="deployment">Deployment</TabsTrigger>
          </TabsList>

          <TabsContent value="structure" className="space-y-4">
            <div className="space-y-3">
              <h4 className="font-semibold">Project Structure</h4>
              <pre className="bg-muted p-4 rounded-lg text-xs overflow-x-auto">
{`trading_system/
├── data/
│   ├── data_engine.py          # DataEngine class
│   ├── yfinance_fetcher.py     # Yahoo Finance API wrapper
│   ├── macro_fetcher.py        # RBI DBIE, FRED integration
│   └── sentiment_fetcher.py    # Google Trends, StockEdge
├── models/
│   ├── transformer_encoder.py  # Transformer with self-attention
│   ├── actor_critic.py         # PPO Actor and Critic networks
│   └── hybrid_model.py         # Combined Transformer-PPO
├── environment/
│   ├── trading_env.py          # Custom Gym environment
│   └── reward_calculator.py    # Cost-aware reward function
├── training/
│   ├── ppo_trainer.py          # PPO training loop
│   ├── buffer.py               # Experience replay buffer
│   └── train.py                # Main training script
├── inference/
│   ├── real_time_trader.py     # RealTimeTrader class
│   └── portfolio_manager.py    # Weight management
├── utils/
│   ├── preprocessing.py        # Scalers, feature engineering
│   ├── metrics.py              # Sharpe, drawdown calculations
│   └── visualization.py        # Plotting utilities
├── config/
│   └── hyperparameters.yaml    # All hyperparameters
├── checkpoints/                # Saved model weights
└── logs/                       # TensorBoard logs`}
              </pre>
            </div>
          </TabsContent>

          <TabsContent value="transformer" className="space-y-4">
            <div className="space-y-3">
              <h4 className="font-semibold">Transformer Encoder Implementation</h4>
              <pre className="bg-muted p-4 rounded-lg text-xs overflow-x-auto">
{`import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    """
    Temporal encoder using Multi-Head Self-Attention.
    Replaces LSTM to capture long-range dependencies in market data.
    """
    def __init__(self, input_dim=50, d_model=128, nhead=4, 
                 num_layers=2, dropout=0.1):
        super().__init__()
        
        # Linear projection to d_model dimension
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding for temporal information
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Multi-head self-attention encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        self.d_model = d_model
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) - Rolling window of features
        Returns:
            out: (batch, d_model) - Encoded representation
        """
        # Project input to d_model
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer
        encoded = self.transformer(x)  # (batch, seq_len, d_model)
        
        # Take the last time step (most recent market state)
        out = encoded[:, -1, :]  # (batch, d_model)
        
        return out

class PositionalEncoding(nn.Module):
    """Injects temporal position information."""
    def __init__(self, d_model, dropout=0.1, max_len=252):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create position encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)`}
              </pre>
            </div>
          </TabsContent>

          <TabsContent value="ppo" className="space-y-4">
            <div className="space-y-3">
              <h4 className="font-semibold">PPO Actor-Critic Networks</h4>
              <pre className="bg-muted p-4 rounded-lg text-xs overflow-x-auto">
{`class ActorCritic(nn.Module):
    """
    Hybrid Transformer-PPO model.
    Combines Transformer encoder with Actor-Critic heads.
    """
    def __init__(self, num_stocks, input_dim=50, d_model=128):
        super().__init__()
        
        # Shared Transformer encoder
        self.encoder = TransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=4,
            num_layers=2
        )
        
        # Actor network: Outputs portfolio weights
        self.actor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_stocks),
            nn.Softmax(dim=-1)  # Ensures weights sum to 1
        )
        
        # Critic network: Estimates state value V(s)
        self.critic = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state):
        """
        Args:
            state: (batch, seq_len, features) - Market state
        Returns:
            weights: (batch, num_stocks) - Portfolio weights
            value: (batch, 1) - State value estimate
        """
        # Encode temporal features
        encoded = self.encoder(state)  # (batch, d_model)
        
        # Get portfolio weights from actor
        weights = self.actor(encoded)  # (batch, num_stocks)
        
        # Get value estimate from critic
        value = self.critic(encoded)   # (batch, 1)
        
        return weights, value
    
    def get_action(self, state, deterministic=False):
        """Sample action from policy."""
        weights, value = self.forward(state)
        
        if deterministic:
            return weights, value
        else:
            # Add noise for exploration
            dist = torch.distributions.Normal(weights, 0.1)
            noisy_weights = dist.sample()
            # Re-normalize to sum to 1
            noisy_weights = torch.softmax(noisy_weights, dim=-1)
            return noisy_weights, value`}
              </pre>
              
              <h4 className="font-semibold mt-4">PPO Training Loop</h4>
              <pre className="bg-muted p-4 rounded-lg text-xs overflow-x-auto">
{`def ppo_update(model, optimizer, states, actions, old_log_probs, 
               returns, advantages, clip_epsilon=0.2):
    """
    PPO policy update with clipped objective.
    """
    # Forward pass
    weights, values = model(states)
    
    # Calculate log probabilities
    dist = torch.distributions.Categorical(weights)
    log_probs = dist.log_prob(actions)
    
    # Policy loss with clipping
    ratio = torch.exp(log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon)
    policy_loss = -torch.min(
        ratio * advantages,
        clipped_ratio * advantages
    ).mean()
    
    # Value loss (MSE)
    value_loss = nn.MSELoss()(values.squeeze(), returns)
    
    # Entropy bonus for exploration
    entropy = dist.entropy().mean()
    
    # Total loss
    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
    
    # Backprop
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    
    return policy_loss.item(), value_loss.item(), entropy.item()`}
              </pre>
            </div>
          </TabsContent>

          <TabsContent value="deployment" className="space-y-4">
            <div className="space-y-3">
              <h4 className="font-semibold">Real-Time Deployment</h4>
              <pre className="bg-muted p-4 rounded-lg text-xs overflow-x-auto">
{`class RealTimeTrader:
    """
    Production inference class for live trading.
    Fetches data, preprocesses, and generates portfolio weights.
    """
    def __init__(self, model_path, scaler_path, stocks):
        # Load trained model
        self.model = torch.load(model_path)
        self.model.eval()
        
        # Load preprocessing scalers (saved from training)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.stocks = stocks
        self.window_size = 60
    
    def fetch_live_data(self):
        """Fetch real-time data from Yahoo Finance."""
        data = {}
        for symbol in self.stocks:
            ticker = yf.Ticker(f"{symbol}.NS")  # NSE
            hist = ticker.history(period="90d")  # Get 90 days
            data[symbol] = hist
        return data
    
    def preprocess(self, raw_data):
        """
        Apply same preprocessing as training.
        CRITICAL: Use saved scaler parameters to prevent look-ahead bias.
        """
        features = self.extract_features(raw_data)
        
        # Scale using training statistics
        scaled = self.scaler.transform(features)
        
        # Create rolling window
        window = scaled[-self.window_size:]
        
        return torch.FloatTensor(window).unsqueeze(0)
    
    def get_portfolio_weights(self):
        """
        Main inference method called at market open.
        """
        # 1. Fetch live data
        raw_data = self.fetch_live_data()
        
        # 2. Preprocess
        state = self.preprocess(raw_data)
        
        # 3. Get weights from model
        with torch.no_grad():
            weights, _ = self.model(state)
        
        # 4. Return as dictionary
        portfolio = {
            stock: weight.item() 
            for stock, weight in zip(self.stocks, weights[0])
        }
        
        return portfolio
    
    def execute_trades(self, target_weights, current_portfolio):
        """
        Calculate rebalancing trades.
        In production, this would integrate with broker API.
        """
        trades = {}
        for stock in self.stocks:
            current = current_portfolio.get(stock, 0)
            target = target_weights.get(stock, 0)
            diff = target - current
            
            if abs(diff) > 0.01:  # 1% threshold
                trades[stock] = diff
        
        return trades`}
              </pre>

              <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
                <h4 className="font-semibold text-sm mb-2">Production Checklist</h4>
                <ul className="text-xs space-y-1">
                  <li>✓ GPU deployment for &lt;50ms latency</li>
                  <li>✓ Redis cache for market data</li>
                  <li>✓ Model versioning (MLflow)</li>
                  <li>✓ TensorBoard monitoring</li>
                  <li>✓ Broker API integration (Zerodha Kite, etc.)</li>
                  <li>✓ Risk limits & position sizing</li>
                  <li>✓ Logging & alerting (Prometheus)</li>
                </ul>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
