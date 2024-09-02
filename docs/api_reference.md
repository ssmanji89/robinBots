# API Reference

## Main Functions

### `get_stock_recommendation(rhSymbol, interval, span, logon, quantity, average_buy_price, previous_close, risk_tolerance=0.05)`

Get a stock recommendation based on trend following and risk management.

Parameters:
- `rhSymbol`: str, Robinhood symbol for the stock
- `interval`: str, time interval for historical data
- `span`: str, time span for historical data
- `logon`: object, Robinhood logon instance
- `quantity`: float, current quantity of stock held
- `average_buy_price`: float, average purchase price of the stock
- `previous_close`: float, previous closing price
- `risk_tolerance`: float, maximum allowed loss as a fraction of position value

Returns: dict with 'action', 'stop_loss', and 'message'

### `analyze_stock(rhSymbol, df, quantity, average_buy_price, previous_close, risk_tolerance=0.02)`

Analyze a stock and provide trading recommendations.

Parameters:
- `rhSymbol`: str, Robinhood symbol for the stock
- `df`: DataFrame, historical stock data and technical indicators
- `quantity`: float, current quantity of stock held
- `average_buy_price`: float, average purchase price of the stock
- `previous_close`: float, previous closing price
- `risk_tolerance`: float, maximum allowed loss as a fraction of position value

Returns: dict with 'action' and 'message'

### `calculate_trade_size(price, sentiment_score, volatility_forecast, market_regime, risk_tolerance=0.02, min_trade_value=1.11, max_position_pct=0.1)`

Calculate the optimal trade size based on various factors.

Parameters:
- `price`: float, current price of the stock
- `sentiment_score`: float, sentiment score from analysis
- `volatility_forecast`: float, expected volatility of the stock
- `market_regime`: int, market regime detection result
- `risk_tolerance`: float, base risk tolerance level
- `min_trade_value`: float, minimum trade value
- `max_position_pct`: float, maximum percentage of portfolio for a single position

Returns: float, number of shares to buy (may be fractional)

## Utility Functions

### `get_stock_historicals(rhSymbol, interval, span, logon)`

Fetch historical stock data and calculate technical indicators.

### `detect_market_regime(df)`

Detect the current market regime (Bullish, Bearish, Sideways).

### `advanced_exit_strategy(df, quantity, entry_price, risk_tolerance=0.02)`

Determine if an exit signal is triggered based on various factors.

### `machine_learning_volatility(df)`

Use machine learning to predict stock volatility.

### `calculate_volatility_clustering(df, window=30)`

Calculate volatility clustering for risk assessment.

For more detailed information on each function, refer to the inline documentation in the source code.
