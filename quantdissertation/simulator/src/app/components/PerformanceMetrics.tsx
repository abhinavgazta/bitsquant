import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TrendingUp, TrendingDown, Activity, DollarSign } from 'lucide-react';

interface PerformanceData {
  date: string;
  portfolioValue: number;
  nifty50: number;
  drawdown: number;
  sharpe: number;
}

interface PerformanceMetricsProps {
  data: PerformanceData[];
}

export function PerformanceMetrics({ data }: PerformanceMetricsProps) {
  const latestData = data[data.length - 1];
  const firstData = data[0];
  
  const totalReturn = ((latestData.portfolioValue - firstData.portfolioValue) / firstData.portfolioValue) * 100;
  const niftyReturn = ((latestData.nifty50 - firstData.nifty50) / firstData.nifty50) * 100;
  const alpha = totalReturn - niftyReturn;
  const maxDrawdown = Math.min(...data.map(d => d.drawdown));

  return (
    <div className="space-y-4">
      {/* Key Metrics Cards */}
      <div className="grid md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Total Return</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-2xl font-bold">{totalReturn.toFixed(2)}%</p>
                <p className="text-xs text-muted-foreground mt-1">vs Nifty: {niftyReturn.toFixed(2)}%</p>
              </div>
              {totalReturn > niftyReturn ? (
                <TrendingUp className="size-8 text-green-500" />
              ) : (
                <TrendingDown className="size-8 text-red-500" />
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Alpha (vs Nifty 50)</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-2xl font-bold">{alpha.toFixed(2)}%</p>
                <p className="text-xs text-muted-foreground mt-1">Excess return</p>
              </div>
              <Activity className="size-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Sharpe Ratio</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-2xl font-bold">{latestData.sharpe.toFixed(2)}</p>
                <p className="text-xs text-muted-foreground mt-1">Risk-adjusted</p>
              </div>
              <DollarSign className="size-8 text-purple-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Max Drawdown</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-2xl font-bold text-red-500">{maxDrawdown.toFixed(2)}%</p>
                <p className="text-xs text-muted-foreground mt-1">Peak to trough</p>
              </div>
              <TrendingDown className="size-8 text-red-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Portfolio Value Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Cumulative Returns (Backtest)</CardTitle>
          <CardDescription>
            Transformer-PPO Agent vs Nifty 50 Benchmark (2013-2025)
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={350}>
            <AreaChart data={data}>
              <defs>
                <linearGradient id="portfolioGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                </linearGradient>
                <linearGradient id="niftyGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
              <XAxis dataKey="date" />
              <YAxis tickFormatter={(value) => `₹${(value / 1000).toFixed(0)}k`} />
              <Tooltip 
                formatter={(value: number) => `₹${value.toLocaleString('en-IN')}`}
                labelStyle={{ color: '#000' }}
              />
              <Legend />
              <Area 
                type="monotone" 
                dataKey="portfolioValue" 
                stroke="#10b981" 
                fill="url(#portfolioGradient)" 
                name="DRL Agent Portfolio"
                strokeWidth={2}
              />
              <Area 
                type="monotone" 
                dataKey="nifty50" 
                stroke="#3b82f6" 
                fill="url(#niftyGradient)" 
                name="Nifty 50 Index"
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Drawdown Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Drawdown Analysis</CardTitle>
          <CardDescription>
            Percentage decline from peak portfolio value
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={data}>
              <defs>
                <linearGradient id="drawdownGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
              <XAxis dataKey="date" />
              <YAxis tickFormatter={(value) => `${value.toFixed(0)}%`} />
              <Tooltip 
                formatter={(value: number) => `${value.toFixed(2)}%`}
                labelStyle={{ color: '#000' }}
              />
              <Area 
                type="monotone" 
                dataKey="drawdown" 
                stroke="#ef4444" 
                fill="url(#drawdownGradient)" 
                name="Drawdown"
              />
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Rolling Sharpe */}
      <Card>
        <CardHeader>
          <CardTitle>Rolling Sharpe Ratio (90-day)</CardTitle>
          <CardDescription>
            Time-varying risk-adjusted performance
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip labelStyle={{ color: '#000' }} />
              <Line type="monotone" dataKey="sharpe" stroke="#8b5cf6" strokeWidth={2} dot={false} name="Sharpe Ratio" />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
}
