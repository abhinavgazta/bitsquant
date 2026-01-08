import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Progress } from './ui/progress';
import { Badge } from './ui/badge';

interface TrainingMetrics {
  episode: number;
  avgReward: number;
  policyLoss: number;
  valueLoss: number;
  entropy: number;
  sharpeRatio: number;
  maxDrawdown: number;
}

interface TrainingDashboardProps {
  trainingData: TrainingMetrics[];
  isTraining: boolean;
  currentEpisode: number;
  totalEpisodes: number;
}

export function TrainingDashboard({ trainingData, isTraining, currentEpisode, totalEpisodes }: TrainingDashboardProps) {
  const progress = (currentEpisode / totalEpisodes) * 100;
  const latestMetrics = trainingData[trainingData.length - 1];

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Training Progress - Transformer-PPO Agent</CardTitle>
          <CardDescription>
            Hybrid architecture learning optimal portfolio weights with Indian market friction costs
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Episode {currentEpisode} / {totalEpisodes}</span>
              <span>{progress.toFixed(1)}%</span>
            </div>
            <Progress value={progress} />
            {isTraining && (
              <Badge variant="default" className="mt-2">Training in Progress...</Badge>
            )}
          </div>

          {/* Key Metrics Grid */}
          {latestMetrics && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Avg Reward</p>
                <p className="text-2xl">{latestMetrics.avgReward.toFixed(4)}</p>
              </div>
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Sharpe Ratio</p>
                <p className="text-2xl">{latestMetrics.sharpeRatio.toFixed(2)}</p>
              </div>
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Max Drawdown</p>
                <p className="text-2xl text-destructive">{(latestMetrics.maxDrawdown * 100).toFixed(1)}%</p>
              </div>
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Entropy</p>
                <p className="text-2xl">{latestMetrics.entropy.toFixed(3)}</p>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Reward Curve */}
      <Card>
        <CardHeader>
          <CardTitle>Episode Reward (Net of STT & Slippage)</CardTitle>
          <CardDescription>
            Cumulative reward after deducting linear turnover costs and quadratic slippage penalties
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={trainingData}>
              <defs>
                <linearGradient id="rewardGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
              <XAxis dataKey="episode" />
              <YAxis />
              <Tooltip />
              <Area type="monotone" dataKey="avgReward" stroke="#3b82f6" fill="url(#rewardGradient)" />
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Loss Curves */}
      <Card>
        <CardHeader>
          <CardTitle>PPO Loss Components</CardTitle>
          <CardDescription>
            Actor (Policy) and Critic (Value) network optimization
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={trainingData}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
              <XAxis dataKey="episode" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="policyLoss" stroke="#10b981" strokeWidth={2} name="Policy Loss" dot={false} />
              <Line type="monotone" dataKey="valueLoss" stroke="#f59e0b" strokeWidth={2} name="Value Loss" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Performance Metrics */}
      <Card>
        <CardHeader>
          <CardTitle>Risk-Adjusted Performance</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={trainingData}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
              <XAxis dataKey="episode" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="sharpeRatio" stroke="#8b5cf6" strokeWidth={2} name="Sharpe Ratio" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
}
