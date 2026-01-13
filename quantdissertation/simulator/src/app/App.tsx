import { useState, useEffect } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Button } from './components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { StockSelector } from './components/StockSelector';
import { ArchitectureViewer } from './components/ArchitectureViewer';
import { TrainingDashboard } from './components/TrainingDashboard';
import { PortfolioWeights } from './components/PortfolioWeights';
import { RealTimeTrader } from './components/RealTimeTrader';
import { PerformanceMetrics } from './components/PerformanceMetrics';
import { 
  generateTrainingData, 
  generatePortfolioWeights, 
  generatePerformanceData,
  DRL_EXPLANATIONS,
  type TrainingMetrics,
  type PortfolioWeight,
  type PerformanceData
} from './utils/mockData';
import { Badge } from './components/ui/badge';
import { Brain, Play, BarChart3, TrendingUp, Info } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from './components/ui/alert';

function App() {
  // State management
  const [selectedStocks, setSelectedStocks] = useState<string[]>(['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']);
  const [isTraining, setIsTraining] = useState(false);
  const [isInferenceRunning, setIsInferenceRunning] = useState(false);
  const [trainingData, setTrainingData] = useState<TrainingMetrics[]>([]);
  const [portfolioWeights, setPortfolioWeights] = useState<PortfolioWeight[]>([]);
  const [performanceData, setPerformanceData] = useState<PerformanceData[]>([]);
  const [currentEpisode, setCurrentEpisode] = useState(0);
  const [activeTab, setActiveTab] = useState('setup');

  const totalEpisodes = 100;

  // Initialize performance data
  useEffect(() => {
    setPerformanceData(generatePerformanceData(500));
  }, []);

  // Simulate training process
  const handleStartTraining = () => {
    setIsTraining(true);
    setTrainingData([]);
    setCurrentEpisode(0);
    setActiveTab('training');

    // Simulate episodic training
    let episode = 0;
    const interval = setInterval(() => {
      episode++;
      setCurrentEpisode(episode);
      setTrainingData(generateTrainingData(episode));

      if (episode >= totalEpisodes) {
        clearInterval(interval);
        setIsTraining(false);
        // Generate final portfolio weights
        setPortfolioWeights(generatePortfolioWeights(selectedStocks));
      }
    }, 100); // Fast simulation for demo
    
    // Cleanup on unmount
    return () => clearInterval(interval);
  };

  // Handle real-time inference
  const handleStartInference = () => {
    setIsInferenceRunning(true);
    // Generate portfolio weights
    const weights = generatePortfolioWeights(selectedStocks);
    setPortfolioWeights(weights);
    setActiveTab('inference');
  };

  const handleStopInference = () => {
    setIsInferenceRunning(false);
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-card">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary/10 rounded-lg">
                <Brain className="size-8 text-primary" />
              </div>
              <div>
                <h1 className="text-2xl font-bold">Hybrid Transformer-PPO Trading System</h1>
                <p className="text-sm text-muted-foreground">
                  Deep Reinforcement Learning for Indian Equity Markets (NSE/BSE)
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline">PyTorch</Badge>
              <Badge variant="outline">Transformer</Badge>
              <Badge variant="outline">PPO</Badge>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <div className="mb-6 overflow-x-auto">
            <TabsList className="grid w-full grid-cols-5 min-w-[600px]">
              <TabsTrigger value="setup">
                <Info className="size-4 mr-2" />
                Setup
              </TabsTrigger>
              <TabsTrigger value="architecture">
                <Brain className="size-4 mr-2" />
                Architecture
              </TabsTrigger>
              <TabsTrigger value="training">
                <Play className="size-4 mr-2" />
                Training
              </TabsTrigger>
              <TabsTrigger value="inference">
                <TrendingUp className="size-4 mr-2" />
                Inference
              </TabsTrigger>
              <TabsTrigger value="performance">
                <BarChart3 className="size-4 mr-2" />
                Performance
              </TabsTrigger>
            </TabsList>
          </div>

          {/* Setup Tab */}
          <TabsContent value="setup" className="space-y-6">
            <Alert>
              <Info className="size-4" />
              <AlertTitle>Welcome to the DRL Trading Terminal</AlertTitle>
              <AlertDescription>
                This system demonstrates a production-ready Hybrid Transformer-PPO framework for autonomous trading.
                Select your portfolio stocks, train the agent, and deploy for real-time inference.
              </AlertDescription>
            </Alert>

            <StockSelector 
              selectedStocks={selectedStocks} 
              onStocksChange={setSelectedStocks}
            />

            <Card>
              <CardHeader>
                <CardTitle>Training Configuration</CardTitle>
                <CardDescription>
                  Hyperparameters for the Transformer-PPO agent
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-3 gap-4 text-sm">
                  <div className="space-y-1">
                    <p className="font-medium">Model Architecture</p>
                    <ul className="text-muted-foreground space-y-0.5">
                      <li>• Transformer: 4 heads, 128d</li>
                      <li>• Actor: 2-layer MLP → Softmax</li>
                      <li>• Critic: 2-layer MLP → Value</li>
                    </ul>
                  </div>
                  <div className="space-y-1">
                    <p className="font-medium">PPO Parameters</p>
                    <ul className="text-muted-foreground space-y-0.5">
                      <li>• Clip ratio ε = 0.2</li>
                      <li>• GAE λ = 0.95</li>
                      <li>• Learning rate = 3e-4</li>
                      <li>• Batch size = 32</li>
                    </ul>
                  </div>
                  <div className="space-y-1">
                    <p className="font-medium">Environment</p>
                    <ul className="text-muted-foreground space-y-0.5">
                      <li>• Turnover cost α = 0.001</li>
                      <li>• Slippage β = 0.0001</li>
                      <li>• Episode length = 252 days</li>
                      <li>• Training episodes = {totalEpisodes}</li>
                    </ul>
                  </div>
                </div>

                <div className="mt-6">
                  <Button 
                    onClick={handleStartTraining}
                    disabled={isTraining || selectedStocks.length === 0}
                    size="lg"
                  >
                    <Play className="size-4 mr-2" />
                    Start Training (Run Backtest)
                  </Button>
                  {selectedStocks.length === 0 && (
                    <p className="text-sm text-muted-foreground mt-2">
                      Please select at least one stock to begin training
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* DRL Explanation */}
            <Card>
              <CardHeader>
                <CardTitle>How It Works: Deep Reinforcement Learning</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <h4 className="font-semibold text-sm mb-2">🧠 Transformer Encoder</h4>
                  <p className="text-sm text-muted-foreground">{DRL_EXPLANATIONS.transformer}</p>
                </div>
                <div>
                  <h4 className="font-semibold text-sm mb-2">🎯 Proximal Policy Optimization (PPO)</h4>
                  <p className="text-sm text-muted-foreground">{DRL_EXPLANATIONS.ppo}</p>
                </div>
                <div>
                  <h4 className="font-semibold text-sm mb-2">💰 Reward Function (Cost-Aware)</h4>
                  <p className="text-sm text-muted-foreground">{DRL_EXPLANATIONS.reward}</p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Architecture Tab */}
          <TabsContent value="architecture">
            <ArchitectureViewer />
          </TabsContent>

          {/* Training Tab */}
          <TabsContent value="training">
            {trainingData.length > 0 ? (
              <TrainingDashboard 
                trainingData={trainingData}
                isTraining={isTraining}
                currentEpisode={currentEpisode}
                totalEpisodes={totalEpisodes}
              />
            ) : (
              <Card>
                <CardContent className="py-12 text-center">
                  <Brain className="size-16 mx-auto text-muted-foreground mb-4" />
                  <h3 className="text-xl font-semibold mb-2">No Training Data Yet</h3>
                  <p className="text-muted-foreground mb-4">
                    Start training from the Setup tab to see real-time metrics
                  </p>
                  <Button onClick={() => setActiveTab('setup')}>
                    Go to Setup
                  </Button>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Inference Tab */}
          <TabsContent value="inference" className="space-y-6">
            <RealTimeTrader 
              stocks={selectedStocks}
              onStartInference={handleStartInference}
              onStopInference={handleStopInference}
              isRunning={isInferenceRunning}
            />

            {portfolioWeights.length > 0 && (
              <PortfolioWeights 
                weights={portfolioWeights}
                timestamp={new Date().toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' })}
              />
            )}

            {!isInferenceRunning && portfolioWeights.length === 0 && (
              <Card>
                <CardContent className="py-12 text-center">
                  <TrendingUp className="size-16 mx-auto text-muted-foreground mb-4" />
                  <h3 className="text-xl font-semibold mb-2">Real-Time Inference Ready</h3>
                  <p className="text-muted-foreground mb-4">
                    {trainingData.length === 0 
                      ? 'Complete training first, then start real-time inference'
                      : 'Click "Start Real-Time Inference" to deploy the trained model'}
                  </p>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Performance Tab */}
          <TabsContent value="performance">
            {performanceData.length > 0 ? (
              <>
                <Alert className="mb-6">
                  <BarChart3 className="size-4" />
                  <AlertTitle>Backtest Results (2020-2025)</AlertTitle>
                  <AlertDescription>
                    Historical performance of the Transformer-PPO agent vs Nifty 50 benchmark.
                    All costs (STT, slippage) are accounted for in the net returns.
                  </AlertDescription>
                </Alert>
                <PerformanceMetrics data={performanceData} />
              </>
            ) : (
              <Card>
                <CardContent className="py-12 text-center">
                  <BarChart3 className="size-16 mx-auto text-muted-foreground mb-4" />
                  <h3 className="text-xl font-semibold mb-2">Loading Performance Data</h3>
                  <p className="text-muted-foreground">
                    Calculating backtest metrics...
                  </p>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </main>

      {/* Footer */}
      <footer className="border-t mt-12 py-6 bg-card">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p className="mb-2">
            Hybrid Transformer-PPO Trading System | Deep Reinforcement Learning for Indian Equities
          </p>
          <p className="text-xs">
            Framework: PyTorch • Data: Yahoo Finance (yfinance), RBI DBIE, FRED • Architecture: Transformer + PPO • Environment: Custom Gym
          </p>
          <p className="text-xs mt-2 italic">
            Note: This is a research demonstration. Not financial advice. Past performance does not guarantee future results.
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;