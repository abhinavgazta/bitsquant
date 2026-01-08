import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Input } from './ui/input';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { X, Plus } from 'lucide-react';

// Popular NSE stocks for quick selection
const POPULAR_NSE_STOCKS = [
  'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
  'HINDUNILVR', 'KOTAKBANK', 'SBIN', 'BHARTIARTL', 'ITC',
  'LT', 'AXISBANK', 'BAJFINANCE', 'ASIANPAINT', 'MARUTI'
];

interface StockSelectorProps {
  selectedStocks: string[];
  onStocksChange: (stocks: string[]) => void;
}

export function StockSelector({ selectedStocks, onStocksChange }: StockSelectorProps) {
  const [inputValue, setInputValue] = useState('');

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
            {POPULAR_NSE_STOCKS.map(stock => (
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
      </CardContent>
    </Card>
  );
}
