import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Badge } from './ui/badge';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface PortfolioWeight {
  symbol: string;
  weight: number;
  previousWeight: number;
  expectedReturn: number;
  confidence: number;
}

interface PortfolioWeightsProps {
  weights: PortfolioWeight[];
  timestamp: string;
}

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444', '#06b6d4', '#ec4899', '#84cc16'];

export function PortfolioWeights({ weights, timestamp }: PortfolioWeightsProps) {
  // Sort by weight descending
  const sortedWeights = [...weights].sort((a, b) => b.weight - a.weight);

  // Calculate total allocation
  const totalAllocation = weights.reduce((sum, w) => sum + w.weight, 0);

  // Prepare data for pie chart
  const pieData = sortedWeights.map(w => ({
    name: w.symbol,
    value: w.weight * 100
  }));

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Optimal Portfolio Weights (w_t)</CardTitle>
          <CardDescription>
            Transformer-PPO output: Softmax probability distribution over {weights.length} stocks
          </CardDescription>
          <div className="flex items-center gap-2 mt-2">
            <Badge variant="outline">Updated: {timestamp}</Badge>
            <Badge variant="secondary">Total: {(totalAllocation * 100).toFixed(1)}%</Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-6">
            {/* Bar Chart */}
            <div>
              <h4 className="text-sm font-semibold mb-4">Weight Distribution</h4>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={sortedWeights}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                  <XAxis dataKey="symbol" angle={-45} textAnchor="end" height={80} />
                  <YAxis tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                  <Tooltip 
                    formatter={(value: number) => `${(value * 100).toFixed(2)}%`}
                    labelStyle={{ color: '#000' }}
                  />
                  <Bar dataKey="weight" fill="#3b82f6" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Pie Chart */}
            <div>
              <h4 className="text-sm font-semibold mb-4">Portfolio Composition</h4>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, value }) => `${name}: ${value.toFixed(1)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value: number) => `${value.toFixed(2)}%`} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Detailed Table */}
      <Card>
        <CardHeader>
          <CardTitle>Position Details & Agent Confidence</CardTitle>
          <CardDescription>
            Actor network output with rebalancing signals and expected returns
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-2 px-2">Symbol</th>
                  <th className="text-right py-2 px-2">Current Weight</th>
                  <th className="text-right py-2 px-2">Previous Weight</th>
                  <th className="text-right py-2 px-2">Change</th>
                  <th className="text-right py-2 px-2">Expected Return</th>
                  <th className="text-right py-2 px-2">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {sortedWeights.map((w) => {
                  const weightChange = w.weight - w.previousWeight;
                  const isIncrease = weightChange > 0;
                  
                  return (
                    <tr key={w.symbol} className="border-b hover:bg-muted/50">
                      <td className="py-3 px-2 font-medium">{w.symbol}</td>
                      <td className="text-right py-3 px-2">
                        <span className="font-semibold">{(w.weight * 100).toFixed(2)}%</span>
                      </td>
                      <td className="text-right py-3 px-2 text-muted-foreground">
                        {(w.previousWeight * 100).toFixed(2)}%
                      </td>
                      <td className="text-right py-3 px-2">
                        <div className="flex items-center justify-end gap-1">
                          {isIncrease ? (
                            <>
                              <TrendingUp className="size-4 text-green-500" />
                              <span className="text-green-500">+{(weightChange * 100).toFixed(2)}%</span>
                            </>
                          ) : weightChange < 0 ? (
                            <>
                              <TrendingDown className="size-4 text-red-500" />
                              <span className="text-red-500">{(weightChange * 100).toFixed(2)}%</span>
                            </>
                          ) : (
                            <span className="text-muted-foreground">0.00%</span>
                          )}
                        </div>
                      </td>
                      <td className="text-right py-3 px-2">
                        <span className={w.expectedReturn > 0 ? 'text-green-500' : 'text-red-500'}>
                          {(w.expectedReturn * 100).toFixed(2)}%
                        </span>
                      </td>
                      <td className="text-right py-3 px-2">
                        <Badge variant={w.confidence > 0.7 ? 'default' : 'secondary'}>
                          {(w.confidence * 100).toFixed(0)}%
                        </Badge>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
