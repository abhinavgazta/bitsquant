import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { ArrowRight, Brain, Layers, Target } from 'lucide-react';

export function ArchitectureViewer() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Hybrid Transformer-PPO Architecture</CardTitle>
        <CardDescription>
          Deep Reinforcement Learning system for autonomous portfolio optimization
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {/* Architecture Flow */}
          <div className="flex flex-col md:flex-row items-center justify-between gap-4 p-4 bg-muted/50 rounded-lg">
            {/* Input Layer */}
            <div className="flex-1 space-y-2 text-center">
              <div className="flex justify-center">
                <div className="p-3 bg-blue-500/10 rounded-lg">
                  <Layers className="size-8 text-blue-500" />
                </div>
              </div>
              <h4 className="font-semibold">Input Layer</h4>
              <p className="text-xs text-muted-foreground">
                (Batch, T=60, Features)
              </p>
              <div className="flex flex-wrap gap-1 justify-center">
                <Badge variant="outline" className="text-xs">OHLCV</Badge>
                <Badge variant="outline" className="text-xs">Technical</Badge>
                <Badge variant="outline" className="text-xs">Macro</Badge>
                <Badge variant="outline" className="text-xs">Sentiment</Badge>
              </div>
            </div>

            <ArrowRight className="size-6 text-muted-foreground hidden md:block" />

            {/* Transformer Encoder */}
            <div className="flex-1 space-y-2 text-center">
              <div className="flex justify-center">
                <div className="p-3 bg-purple-500/10 rounded-lg">
                  <Brain className="size-8 text-purple-500" />
                </div>
              </div>
              <h4 className="font-semibold">Transformer Encoder</h4>
              <p className="text-xs text-muted-foreground">
                Multi-Head Self-Attention
              </p>
              <div className="space-y-1 text-xs">
                <p>• Captures long-range dependencies</p>
                <p>• Market regime detection</p>
                <p>• 4 attention heads, 128d hidden</p>
              </div>
            </div>

            <ArrowRight className="size-6 text-muted-foreground hidden md:block" />

            {/* Actor-Critic */}
            <div className="flex-1 space-y-2 text-center">
              <div className="flex justify-center">
                <div className="p-3 bg-green-500/10 rounded-lg">
                  <Target className="size-8 text-green-500" />
                </div>
              </div>
              <h4 className="font-semibold">PPO Actor-Critic</h4>
              <p className="text-xs text-muted-foreground">
                Portfolio Optimizer
              </p>
              <div className="space-y-1">
                <Badge className="bg-green-500">Actor: w_t (Softmax)</Badge>
                <Badge className="bg-amber-500">Critic: V(s)</Badge>
              </div>
            </div>
          </div>

          {/* Key Features */}
          <div className="grid md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <h4 className="font-semibold text-sm">State Space (s_t)</h4>
              <ul className="text-xs space-y-1 text-muted-foreground">
                <li>• Rolling 60-day OHLCV windows</li>
                <li>• Technical indicators (RSI, MACD, Bollinger)</li>
                <li>• Macro features (10Y G-Sec, USD/INR, CPI)</li>
                <li>• Sentiment proxies (Google Trends, Delivery %)</li>
                <li>• Impact cost estimates (Volume/Turnover)</li>
              </ul>
            </div>

            <div className="space-y-2">
              <h4 className="font-semibold text-sm">Action Space (a_t)</h4>
              <ul className="text-xs space-y-1 text-muted-foreground">
                <li>• Continuous: Portfolio weights w_t ∈ [0,1]^N</li>
                <li>• Constraint: Σw_i = 1 (fully invested)</li>
                <li>• Output: Softmax over N stocks</li>
              </ul>
            </div>

            <div className="space-y-2">
              <h4 className="font-semibold text-sm">Reward Function</h4>
              <ul className="text-xs space-y-1 text-muted-foreground">
                <li>• r_t = Portfolio Return - Turnover Cost - Slippage</li>
                <li>• Turnover: α × |w_t - w_(t-1)| (STT + Brokerage)</li>
                <li>• Slippage: β × (TradeValue/DailyVolume)²</li>
                <li>• Net reward optimizes risk-adjusted returns</li>
              </ul>
            </div>

            <div className="space-y-2">
              <h4 className="font-semibold text-sm">PPO Training</h4>
              <ul className="text-xs space-y-1 text-muted-foreground">
                <li>• Clip ratio ε = 0.2 (policy constraint)</li>
                <li>• GAE (λ = 0.95) for advantage estimation</li>
                <li>• Entropy bonus for exploration</li>
                <li>• Mini-batch updates (32 samples)</li>
              </ul>
            </div>
          </div>

          {/* Data Sources */}
          <div className="space-y-2 p-4 bg-muted/30 rounded-lg">
            <h4 className="font-semibold text-sm">Data Pipeline (2013-2025)</h4>
            <div className="grid md:grid-cols-3 gap-4 text-xs">
              <div>
                <p className="font-medium mb-1">Price Data</p>
                <p className="text-muted-foreground">Yahoo Finance (yfinance) - Adjusted OHLCV, real-time streaming</p>
              </div>
              <div>
                <p className="font-medium mb-1">Macro Data</p>
                <p className="text-muted-foreground">RBI DBIE, FRED - G-Sec yields, FX rates, forward-filled to daily</p>
              </div>
              <div>
                <p className="font-medium mb-1">Sentiment Data</p>
                <p className="text-muted-foreground">Google Trends (pytrends) - Retail crowding proxies</p>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
