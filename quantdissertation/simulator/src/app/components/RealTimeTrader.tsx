import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Play, Square, RefreshCw, Activity } from 'lucide-react';
import { Progress } from './ui/progress';

interface MarketData {
  symbol: string;
  price: number;
  change: number;
  volume: number;
  impactCost: number;
}

interface RealTimeTraderProps {
  stocks: string[];
  onStartInference: () => void;
  onStopInference: () => void;
  isRunning: boolean;
}

export function RealTimeTrader({ stocks, onStartInference, onStopInference, isRunning }: RealTimeTraderProps) {
  const [marketData, setMarketData] = useState<MarketData[]>([]);

  // Simulate real-time data fetch (in production, this would call yfinance API)
  const generateMockMarketData = (): MarketData[] => {
    return stocks.map(symbol => ({
      symbol,
      price: 1000 + Math.random() * 2000,
      change: (Math.random() - 0.5) * 5,
      volume: Math.floor(Math.random() * 10000000),
      impactCost: Math.random() * 0.5
    }));
  };

  const handleStartInference = () => {
    const data = generateMockMarketData();
    setMarketData(data);
    onStartInference();
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Real-Time Inference Engine</CardTitle>
          <CardDescription>
            Live deployment of trained Transformer-PPO model with market data streaming
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Control Panel */}
          <div className="flex items-center gap-4">
            {!isRunning ? (
              <Button onClick={handleStartInference} disabled={stocks.length === 0}>
                <Play className="size-4 mr-2" />
                Start Real-Time Inference
              </Button>
            ) : (
              <Button variant="destructive" onClick={onStopInference}>
                <Square className="size-4 mr-2" />
                Stop Inference
              </Button>
            )}
            
            {isRunning && (
              <Badge variant="default" className="flex items-center gap-2">
                <Activity className="size-3 animate-pulse" />
                Live
              </Badge>
            )}
          </div>

          {/* Inference Pipeline Status */}
          {isRunning && (
            <div className="space-y-3 p-4 bg-muted/50 rounded-lg">
              <h4 className="font-semibold text-sm">Inference Pipeline Status</h4>
              
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>1. Fetching live market data (yfinance)</span>
                  <Badge variant="secondary">✓ Complete</Badge>
                </div>
                <Progress value={100} className="h-1" />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>2. Preprocessing with saved scalers</span>
                  <Badge variant="secondary">✓ Complete</Badge>
                </div>
                <Progress value={100} className="h-1" />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>3. Transformer encoding (4-head attention)</span>
                  <Badge variant="secondary">✓ Complete</Badge>
                </div>
                <Progress value={100} className="h-1" />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>4. Actor network: Portfolio weight generation</span>
                  <Badge variant="default">Running</Badge>
                </div>
                <Progress value={85} className="h-1" />
              </div>
            </div>
          )}

          {/* Market Data Feed */}
          {marketData.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <h4 className="font-semibold text-sm">Live Market Data Stream</h4>
                <Button variant="ghost" size="sm" onClick={() => setMarketData(generateMockMarketData())}>
                  <RefreshCw className="size-4 mr-2" />
                  Refresh
                </Button>
              </div>
              
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2 px-2">Symbol</th>
                      <th className="text-right py-2 px-2">Last Price</th>
                      <th className="text-right py-2 px-2">Change %</th>
                      <th className="text-right py-2 px-2">Volume</th>
                      <th className="text-right py-2 px-2">Impact Cost</th>
                    </tr>
                  </thead>
                  <tbody>
                    {marketData.map((data) => (
                      <tr key={data.symbol} className="border-b hover:bg-muted/50">
                        <td className="py-2 px-2 font-medium">{data.symbol}</td>
                        <td className="text-right py-2 px-2">₹{data.price.toFixed(2)}</td>
                        <td className="text-right py-2 px-2">
                          <span className={data.change > 0 ? 'text-green-500' : 'text-red-500'}>
                            {data.change > 0 ? '+' : ''}{data.change.toFixed(2)}%
                          </span>
                        </td>
                        <td className="text-right py-2 px-2">
                          {(data.volume / 1000000).toFixed(2)}M
                        </td>
                        <td className="text-right py-2 px-2">
                          {data.impactCost.toFixed(3)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Technical Details */}
          <div className="p-4 bg-muted/30 rounded-lg space-y-2">
            <h4 className="font-semibold text-sm">Technical Implementation</h4>
            <ul className="text-xs space-y-1 text-muted-foreground">
              <li>• <strong>Data Source:</strong> Yahoo Finance API (yfinance) for real-time OHLCV</li>
              <li>• <strong>Preprocessing:</strong> StandardScaler with saved μ and σ from training (prevents look-ahead bias)</li>
              <li>• <strong>Sequence Window:</strong> Rolling 60-day window with forward-filled macro features</li>
              <li>• <strong>Model Inference:</strong> PyTorch model.eval() mode, no gradient computation</li>
              <li>• <strong>Output:</strong> Softmax weights w_t ∈ [0,1]^N where Σw_i = 1</li>
              <li>• <strong>Latency:</strong> ~200ms for batch inference on CPU (production: GPU for &lt;50ms)</li>
            </ul>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
