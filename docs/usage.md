# Usage Guide

## Starting the System

1. Activate the virtual environment:
   ```
   source venv/bin/activate
   ```

2. Run the main script:
   ```
   python main.py
   ```

The system will start and run according to the configured schedule.

## Monitoring

- Check the console output for real-time logs
- Review log files in the `logs/` directory for detailed information

## Manual Operations

### Cancelling All Stock Orders

```python
from main import cancel_all_stockOrders
cancel_all_stockOrders()
```

### Analyzing a Specific Stock

```python
from main import get_stock_recommendation
recommendation = get_stock_recommendation('AAPL', 'hour', '3month', logon, quantity, average_buy_price, previous_close, 0.05)
print(recommendation)
```

### Updating Watchlists

Edit the `WATCHLIST_NAMES` list in `config.py` and restart the system.

## Stopping the System

Press Ctrl+C in the terminal running the main script to stop the system gracefully.

Remember to always monitor the system's performance and adjust your risk management settings as needed.
