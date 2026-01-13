import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Input } from './ui/input';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { X, Plus, Download } from 'lucide-react';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';

interface OHLCVData {
  symbol: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  timestamp: string;
}

interface StockSelectorProps {
  selectedStocks: string[];
  onStocksChange: (stocks: string[]) => void;
}

export function StockSelector({ selectedStocks, onStocksChange }: StockSelectorProps) {
  const [inputValue, setInputValue] = useState('');
  const [popularStocks, setPopularStocks] = useState<string[]>([]);
  const [ohlcvData, setOhlcvData] = useState<OHLCVData[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch('/src/app/config/popular-stocks.json')
      .then(response => response.json())
      .then(data => setPopularStocks(data.popular_stocks))
      .catch(error => console.error('Error fetching popular stocks:', error));
  }, []);

  const handleAddStock = () => {
    const stock = inputValue.toUpperCase().trim();
    if (stock && !selectedStocks.includes(stock)) {
      onStocksChange([...selectedStocks, stock]);
      setInputValue('');
    }
  };

  const handleRemoveStock = (stock: string) => {
    onStocksChange(selectedStocks.filter(s => s !== stock));
  };

  const handleQuickAdd = (stock: string) => {
    if (!selectedStocks.includes(stock)) {
      onStocksChange([...selectedStocks, stock]);
    }
  };

  const handleFetchOHLCV = async () => {
    setIsLoading(true);
    setError(null);
    setOhlcvData([]);

    const promises = selectedStocks.map(async (symbol) => {
      try {
        let response = await fetch(`http://127.0.0.1:8000/latest/${symbol}.NS`);
        if (!response.ok) {
          response = await fetch(`http://127.0.0.1:8000/latest/${symbol}.BO`);
        }
        if (!response.ok) {
          throw new Error(`Failed to fetch data for ${symbol}`);
        }
        const responseData = await response.json();
        
        if (!responseData.data || responseData.data.length === 0) {
          throw new Error(`No data returned for ${symbol}`);
        }

        const ohlcv = responseData.data[0];

        return {
          symbol,
          open: ohlcv.Open,
          high: ohlcv.High,
          low: ohlcv.Low,
          close: ohlcv.Close,
          volume: ohlcv.Volume,
          timestamp: new Date(ohlcv.Date).toLocaleDateString('en-IN', {
            day: '2-digit',
            month: '2-digit',
            year: 'numeric',
          }),
        };
      } catch (err) {
        console.error(`Error fetching ${symbol}:`, err);
        throw err; // Re-throw to be caught by Promise.allSettled
      }
    });

    const results = await Promise.allSettled(promises);
    
    const successfulData: OHLCVData[] = [];
    const failedStocks: string[] = [];

    results.forEach((result, index) => {
      if (result.status === 'fulfilled') {
        successfulData.push(result.value as OHLCVData);
      } else {
        failedStocks.push(selectedStocks[index]);
      }
    });

    setOhlcvData(successfulData);

    if (failedStocks.length > 0) {
      setError(`Failed to fetch data for: ${failedStocks.join(', ')}. Please ensure the API is running and the stock symbols are correct.`);
    }

    setIsLoading(false);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Portfolio Stock Selection</CardTitle>
        <CardDescription>
          Select NSE/BSE stocks for the Transformer-PPO agent to optimize
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Input for custom stock */}
        <div className="flex gap-2">
          <Input
            placeholder="Enter stock symbol (e.g., RELIANCE)"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleAddStock()}
          />
          <Button onClick={handleAddStock}>
            <Plus className="size-4 mr-2" />
            Add
          </Button>
        </div>

        {/* Selected stocks */}
        {selectedStocks.length > 0 && (
          <div>
            <p className="text-sm mb-2">Selected Stocks ({selectedStocks.length}):</p>
            <div className="flex flex-wrap gap-2">
              {selectedStocks.map(stock => (
                <Badge key={stock} variant="secondary" className="px-3 py-1">
                  {stock}
                  <button
                    onClick={() => handleRemoveStock(stock)}
                    className="ml-2 hover:text-destructive"
                  >
                    <X className="size-3" />
                  </button>
                </Badge>
              ))}
            </div>
          </div>
        )}

        {/* Quick add popular stocks */}
        <div>
          <p className="text-sm mb-2">Quick Add (Popular NSE Stocks):</p>
          <div className="flex flex-wrap gap-2">
            {popularStocks.map(stock => (
              <Button
                key={stock}
                variant="outline"
                size="sm"
                onClick={() => handleQuickAdd(stock)}
                disabled={selectedStocks.includes(stock)}
              >
                {stock}
              </Button>
            ))}
          </div>
        </div>

        {/* OHLCV Fetching */}
        <div className="space-y-4 pt-4 border-t">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold">Real-Time OHLCV Data</h3>
              <p className="text-sm text-muted-foreground">
                Fetch OHLCV data for the selected stocks from the API.
              </p>
            </div>
            <Button onClick={handleFetchOHLCV} disabled={selectedStocks.length === 0 || isLoading}>
              <Download className="size-4 mr-2" />
              {isLoading ? 'Fetching...' : 'Fetch Prices'}
            </Button>
          </div>

          {isLoading && <p>Loading data...</p>}
          
          {error && (
            <Alert variant="destructive">
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {ohlcvData.length > 0 && (
            <div>
              <p className="text-sm text-muted-foreground mb-2">
                Data fetched on: {new Date().toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' })}
              </p>
              <div className="rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Symbol</TableHead>
                      <TableHead>Date</TableHead>
                      <TableHead className="text-right">Open</TableHead>
                      <TableHead className="text-right">High</TableHead>
                      <TableHead className="text-right">Low</TableHead>
                      <TableHead className="text-right">Close</TableHead>
                      <TableHead className="text-right">Volume</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {ohlcvData.map((data) => (
                      <TableRow key={data.symbol}>
                        <TableCell className="font-medium">{data.symbol}</TableCell>
                        <TableCell>{data.timestamp}</TableCell>
                        <TableCell className="text-right">{data.open?.toFixed(2) ?? 'N/A'}</TableCell>
                        <TableCell className="text-right">{data.high?.toFixed(2) ?? 'N/A'}</TableCell>
                        <TableCell className="text-right">{data.low?.toFixed(2) ?? 'N/A'}</TableCell>
                        <TableCell className="text-right">{data.close?.toFixed(2) ?? 'N/A'}</TableCell>
                        <TableCell className="text-right">
                          {data.volume ? `${(data.volume / 1000000).toFixed(2)}M` : 'N/A'}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
