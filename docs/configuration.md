# Configuration Guide

## Environment Variables

Edit the `.env` file with your specific settings:

- `ROBINHOOD_USERNAME`: Your Robinhood account username
- `ROBINHOOD_PASSWORD`: Your Robinhood account password
- `ROBINHOOD_TOTP_SECRET`: Your Robinhood Two-Factor Authentication secret
- `RISK_TOLERANCE`: Default risk tolerance (e.g., 0.02 for 2%)
- `MAX_POSITION_PCT`: Maximum percentage of portfolio for a single position (e.g., 0.1 for 10%)
- `MIN_TRADE_VALUE`: Minimum trade value in dollars (e.g., 1.11)

## Watchlist Configuration

Edit `config.py` to set up your watchlists:

```python
WATCHLIST_NAMES = ["2024GPTd", "AWP", "100 Most Popular", "Daily Movers", "Upcoming Earnings", "Energy & Water"]
```

## Scheduling Configuration

Modify the `main()` function in `main.py` to adjust the scheduling of tasks:

```python
scheduler.add_job(cancel_all_stockOrders, trigger='cron', day_of_week='*', hour='22', minute='11')
scheduler.add_job(_main_open_positions, trigger='cron', day_of_week='mon-fri', hour='6,14', minute='11')
```

## Trading Strategy Configuration

Adjust parameters in `analyze_stock()` function to fine-tune your trading strategy:

- Modify technical indicator calculations
- Adjust buy/sell signal thresholds
- Fine-tune machine learning model parameters

Remember to test your configuration thoroughly before running with real money!
