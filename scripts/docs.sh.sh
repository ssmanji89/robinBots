#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    printf "${1}${2}${NC}\n"
}

# Create docs directory if it doesn't exist
mkdir -p docs

# Function to create a markdown file with content
create_doc() {
    local file="$1"
    local content="$2"
    echo "$content" > "docs/$file"
    print_color $BLUE "Created $file"
}

# Create README.md
print_color $GREEN "Creating README.md..."
create_doc "README.md" "# robinBots Documentation

Welcome to the documentation for robinBots, an advanced programmatic trading system.

## Table of Contents

1. [Introduction](introduction.md)
2. [Installation](installation.md)
3. [Configuration](configuration.md)
4. [Usage](usage.md)
5. [API Reference](api_reference.md)
6. [Architecture](architecture.md)
7. [Development](development.md)
8. [Testing](testing.md)
9. [Deployment](deployment.md)
10. [Troubleshooting](troubleshooting.md)
11. [FAQ](faq.md)
12. [Change Log](changelog.md)
"

# Create Introduction
print_color $GREEN "Creating introduction.md..."
create_doc "introduction.md" "# Introduction to robinBots

robinBots is a sophisticated programmatic trading system designed to automate stock trading strategies. It leverages various technical indicators, machine learning algorithms, and risk management techniques to make informed trading decisions.

## Key Features

- Automated trading based on technical analysis
- Integration with Robinhood API
- Machine learning-based price prediction
- Advanced risk management
- Real-time market data analysis
- Customizable trading strategies
- Sentiment analysis of market news

## System Overview

The system consists of several key components:

1. Data Collection: Fetches historical and real-time stock data from Robinhood API.
2. Technical Analysis: Calculates various technical indicators like EMA, RSI, MACD, etc.
3. Machine Learning: Uses Random Forest algorithm for price prediction.
4. Decision Making: Combines technical indicators, ML predictions, and market sentiment for trade decisions.
5. Risk Management: Implements stop-loss, trailing stop, and volatility-based exit strategies.
6. Order Execution: Interfaces with Robinhood API to execute buy/sell orders.
7. Scheduling: Uses BackgroundScheduler for periodic tasks like data updates and position checks.

This documentation will guide you through the setup, configuration, and usage of robinBots."

# Create Installation guide
print_color $GREEN "Creating installation.md..."
create_doc "installation.md" "# Installation Guide

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git

## Steps

1. Clone the repository:
   \`\`\`
   git clone https://github.com/yourusername/robinBots.git
   cd robinBots
   \`\`\`

2. Create a virtual environment:
   \`\`\`
   python3 -m venv venv
   source venv/bin/activate
   \`\`\`

3. Install required packages:
   \`\`\`
   pip install -r requirements.txt
   \`\`\`

4. Set up environment variables:
   - Copy the \`.env.example\` file to \`.env\`
   - Fill in your Robinhood credentials and API keys

5. Initialize the database:
   \`\`\`
   python init_db.py
   \`\`\`

6. Run the initial setup script:
   \`\`\`
   ./setup.sh
   \`\`\`

Now you're ready to configure and run robinBots!"

# Create Configuration guide
print_color $GREEN "Creating configuration.md..."
create_doc "configuration.md" "# Configuration Guide

## Environment Variables

Edit the \`.env\` file with your specific settings:

- \`ROBINHOOD_USERNAME\`: Your Robinhood account username
- \`ROBINHOOD_PASSWORD\`: Your Robinhood account password
- \`ROBINHOOD_TOTP_SECRET\`: Your Robinhood Two-Factor Authentication secret
- \`RISK_TOLERANCE\`: Default risk tolerance (e.g., 0.02 for 2%)
- \`MAX_POSITION_PCT\`: Maximum percentage of portfolio for a single position (e.g., 0.1 for 10%)
- \`MIN_TRADE_VALUE\`: Minimum trade value in dollars (e.g., 1.11)

## Watchlist Configuration

Edit \`config.py\` to set up your watchlists:

\`\`\`python
WATCHLIST_NAMES = [\"2024GPTd\", \"AWP\", \"100 Most Popular\", \"Daily Movers\", \"Upcoming Earnings\", \"Energy & Water\"]
\`\`\`

## Scheduling Configuration

Modify the \`main()\` function in \`main.py\` to adjust the scheduling of tasks:

\`\`\`python
scheduler.add_job(cancel_all_stockOrders, trigger='cron', day_of_week='*', hour='22', minute='11')
scheduler.add_job(_main_open_positions, trigger='cron', day_of_week='mon-fri', hour='6,14', minute='11')
\`\`\`

## Trading Strategy Configuration

Adjust parameters in \`analyze_stock()\` function to fine-tune your trading strategy:

- Modify technical indicator calculations
- Adjust buy/sell signal thresholds
- Fine-tune machine learning model parameters

Remember to test your configuration thoroughly before running with real money!"

# Create Usage guide
print_color $GREEN "Creating usage.md..."
create_doc "usage.md" "# Usage Guide

## Starting the System

1. Activate the virtual environment:
   \`\`\`
   source venv/bin/activate
   \`\`\`

2. Run the main script:
   \`\`\`
   python main.py
   \`\`\`

The system will start and run according to the configured schedule.

## Monitoring

- Check the console output for real-time logs
- Review log files in the \`logs/\` directory for detailed information

## Manual Operations

### Cancelling All Stock Orders

\`\`\`python
from main import cancel_all_stockOrders
cancel_all_stockOrders()
\`\`\`

### Analyzing a Specific Stock

\`\`\`python
from main import get_stock_recommendation
recommendation = get_stock_recommendation('AAPL', 'hour', '3month', logon, quantity, average_buy_price, previous_close, 0.05)
print(recommendation)
\`\`\`

### Updating Watchlists

Edit the \`WATCHLIST_NAMES\` list in \`config.py\` and restart the system.

## Stopping the System

Press Ctrl+C in the terminal running the main script to stop the system gracefully.

Remember to always monitor the system's performance and adjust your risk management settings as needed."

# Create API Reference
print_color $GREEN "Creating api_reference.md..."
create_doc "api_reference.md" "# API Reference

## Main Functions

### \`get_stock_recommendation(rhSymbol, interval, span, logon, quantity, average_buy_price, previous_close, risk_tolerance=0.05)\`

Get a stock recommendation based on trend following and risk management.

Parameters:
- \`rhSymbol\`: str, Robinhood symbol for the stock
- \`interval\`: str, time interval for historical data
- \`span\`: str, time span for historical data
- \`logon\`: object, Robinhood logon instance
- \`quantity\`: float, current quantity of stock held
- \`average_buy_price\`: float, average purchase price of the stock
- \`previous_close\`: float, previous closing price
- \`risk_tolerance\`: float, maximum allowed loss as a fraction of position value

Returns: dict with 'action', 'stop_loss', and 'message'

### \`analyze_stock(rhSymbol, df, quantity, average_buy_price, previous_close, risk_tolerance=0.02)\`

Analyze a stock and provide trading recommendations.

Parameters:
- \`rhSymbol\`: str, Robinhood symbol for the stock
- \`df\`: DataFrame, historical stock data and technical indicators
- \`quantity\`: float, current quantity of stock held
- \`average_buy_price\`: float, average purchase price of the stock
- \`previous_close\`: float, previous closing price
- \`risk_tolerance\`: float, maximum allowed loss as a fraction of position value

Returns: dict with 'action' and 'message'

### \`calculate_trade_size(price, sentiment_score, volatility_forecast, market_regime, risk_tolerance=0.02, min_trade_value=1.11, max_position_pct=0.1)\`

Calculate the optimal trade size based on various factors.

Parameters:
- \`price\`: float, current price of the stock
- \`sentiment_score\`: float, sentiment score from analysis
- \`volatility_forecast\`: float, expected volatility of the stock
- \`market_regime\`: int, market regime detection result
- \`risk_tolerance\`: float, base risk tolerance level
- \`min_trade_value\`: float, minimum trade value
- \`max_position_pct\`: float, maximum percentage of portfolio for a single position

Returns: float, number of shares to buy (may be fractional)

## Utility Functions

### \`get_stock_historicals(rhSymbol, interval, span, logon)\`

Fetch historical stock data and calculate technical indicators.

### \`detect_market_regime(df)\`

Detect the current market regime (Bullish, Bearish, Sideways).

### \`advanced_exit_strategy(df, quantity, entry_price, risk_tolerance=0.02)\`

Determine if an exit signal is triggered based on various factors.

### \`machine_learning_volatility(df)\`

Use machine learning to predict stock volatility.

### \`calculate_volatility_clustering(df, window=30)\`

Calculate volatility clustering for risk assessment.

For more detailed information on each function, refer to the inline documentation in the source code."

# Create Architecture overview
print_color $GREEN "Creating architecture.md..."
create_doc "architecture.md" "# Architecture Overview

robinBots is designed with a modular architecture to ensure flexibility, scalability, and maintainability.

## High-Level Components

1. **Data Collection Module**
   - Interfaces with Robinhood API to fetch real-time and historical stock data
   - Implements rate limiting and error handling for API requests

2. **Technical Analysis Module**
   - Calculates various technical indicators (EMA, RSI, MACD, Bollinger Bands, etc.)
   - Provides functions for trend analysis and pattern recognition

3. **Machine Learning Module**
   - Implements Random Forest algorithm for price prediction
   - Handles feature engineering and model training

4. **Decision Making Engine**
   - Combines inputs from technical analysis, ML predictions, and market sentiment
   - Applies trading rules and generates buy/sell signals

5. **Risk Management Module**
   - Implements various exit strategies (stop-loss, trailing stop, volatility-based)
   - Calculates optimal position sizes based on risk tolerance

6. **Order Execution Module**
   - Interfaces with Robinhood API to place and manage orders
   - Handles order types (market, limit) and implements smart order routing

7. **Scheduling and Task Management**
   - Uses BackgroundScheduler for periodic tasks and data updates
   - Manages concurrent operations and ensures thread safety

8. **Logging and Monitoring**
   - Implements comprehensive logging for all system activities
   - Provides real-time monitoring and alerting capabilities

9. **Configuration Management**
   - Handles environment variables and configuration files
   - Allows for easy customization of trading parameters

## Data Flow

1. Scheduler triggers data collection for watchlist stocks
2. Technical indicators are calculated on the fetched data
3. Machine learning model makes price predictions
4. Decision making engine analyzes all inputs and generates trading signals
5. Risk management module calculates appropriate position sizes
6. Order execution module places trades based on the generated signals
7. System logs all activities and updates relevant databases

## Key Design Principles

- **Modularity**: Each component is designed to be independent and easily replaceable
- **Scalability**: The system can handle multiple stocks and strategies concurrently
- **Robustness**: Comprehensive error handling and failsafe mechanisms are implemented
- **Flexibility**: Trading strategies and parameters are easily configurable
- **Performance**: Efficient algorithms and data structures are used for real-time processing

## Future Enhancements

- Implement a microservices architecture for better scalability
- Add support for additional data sources and APIs
- Integrate advanced ML techniques like deep learning for improved predictions
- Develop a web-based dashboard for real-time monitoring and control

This architecture provides a solid foundation for the robinBots system, allowing for easy expansion and improvement as trading strategies evolve."

# Create Development guide
print_color $GREEN "Creating development.md..."
create_doc "development.md" "# Development Guide

## Setting Up the Development Environment

1. Clone the repository:
   \`\`\`
   git clone https://github.com/yourusername/robinBots.git
   cd robinBots
   \`\`\`

2. Create and activate a virtual environment:
   \`\`\`
   python3 -m venv venv
   source venv/bin/activate
   \`\`\`

3. Install development dependencies:
   \`\`\`
   pip install -r requirements-dev.txt
   \`\`\`

4. Set up pre-commit hooks:
   \`\`\`
   pre-commit install
   \`\`\`

## Code Style and Linting

We use Black for code formatting and Flake8 for linting. Run these before committing:

\`\`\`
black .
flake8
\`\`\`

## Running Tests

Use pytest to run the test suite:

\`\`\`
pytest
\`\`\`

For test coverage report:

\`\`\`
pytest --cov=src tests/
\`\`\`

## Adding New Features

1. Create a new branch for your feature:
   \`\`\`
   git checkout -b feature/your-feature-name
   \`\`\`

2. Implement your feature, adding tests as necessary.

3. Update documentation in the \`docs/\` directory.

4. Commit your changes:
   \`\`\`
   git add .
   git commit -m \"Add your feature description\"
   \`\`\`

5. Push your branch and create a pull request on GitHub.

## Modifying Trading Strategies

When modifying trading strategies:

1. Update the \`analyze_stock()\` function in \`src/analysis.py\`.
2. Adjust risk management parameters in \`src/risk_management.py\`.
3. If adding new technical indicators, implement them in \`src/indicators.py\`.
4. Update the decision making logic in \`src/decision_engine.py\`.

## Working with the Robinhood API

- All Robinhood API interactions should go through the \`RobinhoodClient\` class in \`src/robinhood_client.py\`.
- Implement proper error handling and rate limiting.
- Use environment variables for sensitive information like API keys.

## Debugging Tips

- Use logging extensively. Import the logger from \`src/logger.py\`.
- For complex issues, use the Python debugger (pdb) or an IDE debugger.
- Monitor system performance using the built-in profiling tools.

## Contribution Guidelines

- Follow the PEP 8 style guide.
- Write clear, self-documenting code with appropriate comments.
- Include unit tests for all new features.
- Update the documentation for any user-facing changes.
- Be responsive to code review comments.

Remember, the goal is to maintain a robust and efficient trading system. Always consider the implications of your changes on the overall system performance and reliability."

# Create Testing guide
print_color $GREEN "Creating testing.md..."
create_doc "testing.md" "# Testing Guide

## Test Structure

Our test suite is organized as follows:

- \`tests/unit/\`: Unit tests for individual functions and classes
- \`tests/integration/\`: Integration tests for module interactions
- \`tests/e2e/\`: End-to-end tests simulating real trading scenarios

## Running Tests

To run the entire test suite:

\`\`\`
pytest
\`\`\`

To run a specific test file:

\`\`\`
pytest tests/unit/test_analysis.py
\`\`\`

For verbose output:

\`\`\`
pytest -v
\`\`\`

## Writing Tests

1. Create a new test file in the appropriate directory (unit, integration, or e2e).
2. Import the necessary modules and the function/class you're testing.
3. Write test functions prefixed with \`test_\`.
4. Use assertions to check expected outcomes.

Example:

\`\`\`python
# tests/unit/test_analysis.py
from src.analysis import calculate_rsi

def test_calculate_rsi():
    data = [10, 12, 15, 14, 13, 16, 17, 15]
    rsi = calculate_rsi(data, period=14)
    assert 0 <= rsi <= 100
    assert round(rsi, 2) == 57.14  # Expected RSI value
\`\`\`

## Mocking

Use the \`unittest.mock\` module to mock external dependencies, especially API calls:

\`\`\`python
from unittest.mock import patch
from src.robinhood_client import RobinhoodClient

@patch('src.robinhood_client.rs.robinhood.get_stock_quote_by_symbol')
def test_get_stock_price(mock_get_quote):
    mock_get_quote.return_value = {'last_trade_price': '150.00'}
    client = RobinhoodClient()
    price = client.get_stock_price('AAPL')
    assert price == 150.00
\`\`\`

## Test Coverage

To check test coverage:

\`\`\`
pytest --cov=src tests/
\`\`\`

Aim for at least 80% coverage for critical components.

## Continuous Integration

We use GitHub Actions for CI. The workflow is defined in \`.github/workflows/ci.yml\`.

It runs on every push and pull request, executing:
1. Linting with Flake8
2. Unit tests with pytest
3. Integration tests
4. Coverage report

## Performance Testing

For performance-critical functions, use the \`timeit\` module:

\`\`\`python
import timeit

def test_analysis_performance():
    setup = \"from src.analysis import analyze_stock; import pandas as pd\"
    stmt = \"analyze_stock('AAPL', pd.DataFrame({'close': range(1000)})\"
    result = timeit.timeit(stmt, setup, number=100)
    assert result < 1.0  # Should complete in less than 1 second
\`\`\`

## Security Testing

Regularly scan dependencies for vulnerabilities:

\`\`\`
safety check
\`\`\`

## Troubleshooting Tests

- Use \`pytest -vv\` for more detailed output.
- Use \`pytest.set_trace()\` to debug within a test.
- Check log files for additional information during test runs.

Remember, thorough testing is crucial for maintaining the reliability and performance of our trading system. Always write tests for new features and bug fixes."

# Create Deployment guide
print_color $GREEN "Creating deployment.md..."
create_doc "deployment.md" "# Deployment Guide

## Prerequisites

- Docker installed on the deployment machine
- Access to a Docker registry (e.g., Docker Hub)
- SSH access to the deployment server
- Necessary environment variables and configuration files

## Building the Docker Image

1. Navigate to the project root directory.

2. Build the Docker image:
   \`\`\`
   docker build -t robinbots:latest .
   \`\`\`

3. Tag the image for your registry:
   \`\`\`
   docker tag robinbots:latest your-registry/robinbots:latest
   \`\`\`

4. Push the image to your registry:
   \`\`\`
   docker push your-registry/robinbots:latest
   \`\`\`

## Preparing the Deployment Server

1. SSH into your server:
   \`\`\`
   ssh user@your-server-ip
   \`\`\`

2. Install Docker if not already installed:
   \`\`\`
   sudo apt-get update
   sudo apt-get install docker.io
   \`\`\`

3. Create a directory for the application:
   \`\`\`
   mkdir -p /opt/robinbots
   cd /opt/robinbots
   \`\`\`

4. Create a \`.env\` file with necessary environment variables:
   \`\`\`
   touch .env
   nano .env
   # Add your environment variables here
   \`\`\`

## Deploying the Application

1. Pull the latest image:
   \`\`\`
   docker pull your-registry/robinbots:latest
   \`\`\`

2. Stop and remove the existing container (if any):
   \`\`\`
   docker stop robinbots || true
   docker rm robinbots || true
   \`\`\`

3. Run the new container:
   \`\`\`
   docker run -d --name robinbots \
     --env-file /opt/robinbots/.env \
     -v /opt/robinbots/data:/app/data \
     -v /opt/robinbots/logs:/app/logs \
     --restart unless-stopped \
     your-registry/robinbots:latest
   \`\`\`

## Monitoring the Deployment

1. Check if the container is running:
   \`\`\`
   docker ps
   \`\`\`

2. View the logs:
   \`\`\`
   docker logs -f robinbots
   \`\`\`

3. Monitor system resources:
   \`\`\`
   docker stats robinbots
   \`\`\`

## Updating the Application

1. Pull the latest image:
   \`\`\`
   docker pull your-registry/robinbots:latest
   \`\`\`

2. Stop and remove the existing container:
   \`\`\`
   docker stop robinbots
   docker rm robinbots
   \`\`\`

3. Run the new container (same command as in the Deploying section).

## Backup and Restore

1. Backup the data directory:
   \`\`\`
   tar -czvf robinbots_data_backup.tar.gz /opt/robinbots/data
   \`\`\`

2. To restore:
   \`\`\`
   tar -xzvf robinbots_data_backup.tar.gz -C /
   \`\`\`

## Troubleshooting

- If the container fails to start, check the logs:
  \`\`\`
  docker logs robinbots
  \`\`\`

- Ensure all required environment variables are set in the \`.env\` file.

- Check system resources (CPU, memory, disk space) to ensure they're not exhausted.

Remember to always test the deployment process in a staging environment before applying it to production. Regularly review and update your deployment process to incorporate best practices and new features."

# Create Troubleshooting guide
print_color $GREEN "Creating troubleshooting.md..."
create_doc "troubleshooting.md" "# Troubleshooting Guide

## Common Issues and Solutions

### 1. Application Fails to Start

**Symptoms:**
- Docker container exits immediately after starting
- Error messages in logs about missing environment variables

**Possible Solutions:**
- Check if all required environment variables are set in the \`.env\` file
- Verify that the \`.env\` file is in the correct location and is being read by the container
- Check Docker logs for specific error messages:
  \`\`\`
  docker logs robinbots
  \`\`\`

### 2. API Connection Issues

**Symptoms:**
- Error messages about failed API requests
- Unexpected \`None\` values in stock data

**Possible Solutions:**
- Verify Robinhood API credentials in the \`.env\` file
- Check internet connectivity on the server
- Ensure you're not exceeding API rate limits
- Verify that the Robinhood API is operational

### 3. Unexpected Trading Behavior

**Symptoms:**
- Trades are not executed as expected
- Unusual buy/sell signals

**Possible Solutions:**
- Review the logs for any warnings or errors
- Check if the analysis parameters in \`config.py\` are set correctly
- Verify that the historical data being used is accurate and up-to-date
- Ensure that the risk management settings are appropriate

### 4. Performance Issues

**Symptoms:**
- Slow response times
- High CPU or memory usage

**Possible Solutions:**
- Check system resources using \`docker stats robinbots\`
- Review and optimize database queries if applicable
- Consider scaling up the server resources
- Analyze logs for any operations taking unusually long time

### 5. Data Inconsistencies

**Symptoms:**
- Mismatched data between different parts of the application
- Unexpected \`NaN\` values in calculations

**Possible Solutions:**
- Verify data sources and ensure they're reliable
- Check for any data transformation errors in the code
- Ensure that timezone handling is consistent throughout the application
- Review data cleaning and preprocessing steps

### 6. Scheduling Issues

**Symptoms:**
- Tasks not running at expected times
- Missed trading opportunities

**Possible Solutions:**
- Check if the server time is set correctly
- Review the cron job configurations
- Ensure that the BackgroundScheduler is running properly
- Check logs for any errors related to scheduled tasks

### 7. Docker-related Issues

**Symptoms:**
- Container stops unexpectedly
- Unable to access files or directories

**Possible Solutions:**
- Check Docker logs: \`docker logs robinbots\`
- Verify file permissions for mounted volumes
- Ensure there's enough disk space on the host machine
- Check if the Docker daemon is running properly

## Debugging Steps

1. **Check Logs:**
   - Application logs: \`docker exec robinbots cat /app/logs/app.log\`
   - Docker logs: \`docker logs robinbots\`

2. **Verify Configurations:**
   - Review \`.env\` file for correct settings
   - Check \`config.py\` for proper parameters

3. **Test API Connectivity:**
   - Use a tool like Postman to test API endpoints
   - Verify API credentials and permissions

4. **Analyze Data:**
   - Use Jupyter Notebook to analyze historical data
   - Check for data integrity and consistency

5. **Monitor System Resources:**
   - Use \`docker stats robinbots\` to monitor container resource usage
   - Check host machine resources with \`top\` or \`htop\`

6. **Review Code:**
   - Use debugging tools in your IDE
   - Add additional logging for problematic areas

7. **Test in Isolation:**
   - Run specific components separately to isolate issues
   - Use mock data to test analysis and decision-making logic

If problems persist after trying these solutions, consider reaching out to the development team or consulting the project's issue tracker on GitHub."

# Create FAQ
print_color $GREEN "Creating faq.md..."
create_doc "faq.md" "# Frequently Asked Questions (FAQ)

## General Questions

### Q: What is robinBots?
A: robinBots is an automated trading system designed to execute trades based on technical analysis, machine learning predictions, and risk management strategies. It integrates with the Robinhood API to perform trades.

### Q: Is robinBots suitable for beginners?
A: While robinBots is designed to be user-friendly, it's recommended that users have a good understanding of trading concepts, technical analysis, and risk management before using the system with real money.

### Q: Can I use robinBots with other brokers besides Robinhood?
A: Currently, robinBots is designed specifically for use with Robinhood. Integration with other brokers would require significant modifications to the codebase.

## Setup and Configuration

### Q: How do I set up robinBots?
A: Refer to the [Installation Guide](installation.md) for detailed setup instructions. You'll need to clone the repository, set up a Python environment, install dependencies, and configure your Robinhood API credentials.

### Q: Where do I input my Robinhood credentials?
A: Your Robinhood credentials should be stored in the \`.env\` file. Never commit this file to version control. See the [Configuration Guide](configuration.md) for more details.

### Q: How can I customize the trading strategy?
A: You can modify the trading strategy by adjusting parameters in the \`config.py\` file and by modifying the analysis logic in the \`analyze_stock()\` function. Refer to the [Development Guide](development.md) for more information.

## Usage and Operation

### Q: How often does robinBots make trades?
A: The frequency of trades depends on your configuration and market conditions. By default, the system analyzes positions twice daily on weekdays, but actual trades are made based on the analysis results and configured thresholds.

### Q: Can I set a maximum amount of money to trade?
A: Yes, you can set a maximum position size as a percentage of your portfolio in the configuration. This helps manage risk and prevent overexposure to any single stock.

### Q: How does robinBots handle risk management?
A: robinBots implements several risk management strategies, including stop-loss orders, position sizing based on volatility, and diversification across multiple stocks. You can adjust risk parameters in the configuration.

## Troubleshooting and Maintenance

### Q: What should I do if robinBots isn't making any trades?
A: Check the logs for any error messages. Ensure that your API credentials are correct and that you have sufficient funds in your account. Also, verify that your trading thresholds aren't set too conservatively.

### Q: How can I update robinBots to the latest version?
A: Pull the latest changes from the GitHub repository, update dependencies, and follow any migration instructions provided in the changelog.

### Q: Is it safe to run robinBots continuously?
A: While robinBots is designed for continuous operation, it's important to regularly monitor its performance and check for any unusual behavior or errors in the logs.

## Performance and Results

### Q: What kind of returns can I expect from robinBots?
A: Returns can vary widely based on market conditions, your configuration, and the stocks you're trading. Always backtest your strategies and start with small amounts until you're comfortable with the system's performance.

### Q: How does robinBots perform in different market conditions?
A: robinBots uses adaptive strategies that aim to perform in various market conditions. However, like any trading system, it may perform differently in bull markets, bear markets, or highly volatile conditions.

### Q: Can robinBots guarantee profits?
A: No trading system can guarantee profits. robinBots is a tool to assist in trading decisions, but it comes with risks. Always understand the risks involved in algorithmic trading and never invest more than you can afford to lose.

## Legal and Compliance

### Q: Is it legal to use robinBots?
A: Automated trading is generally legal, but you should ensure you comply with all relevant financial regulations in your jurisdiction. Consult with a legal professional if you have concerns.

### Q: Do I need to report trades made by robinBots for tax purposes?
A: Yes, trades made by robinBots are subject to the same tax reporting requirements as manually executed trades. Consult with a tax professional for specific advice on your situation.

If you have a question that's not answered here, please check the documentation or reach out to the development team for assistance."

# Create Change Log
print_color $GREEN "Creating changelog.md..."
create_doc "changelog.md" "# Change Log

All notable changes to the robinBots project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation in the \`docs/\` directory

## [1.0.0] 

### Added
- Initial release of robinBots
- Integration with Robinhood API
- Technical analysis module with various indicators (EMA, RSI, MACD, etc.)
- Machine learning-based price prediction using Random Forest
- Risk management module with stop-loss and position sizing
- Automated scheduling of trading tasks
- Basic logging and monitoring capabilities

### Changed
- N/A (initial release)

### Deprecated
- N/A (initial release)

### Removed
- N/A (initial release)

### Fixed
- N/A (initial release)

### Security
- Implemented secure handling of API credentials

## [0.9.0] 

### Added
- Beta version for internal testing
- Core functionality for stock data retrieval and analysis
- Basic trading logic implementation

### Changed
- Refactored code structure for better modularity

### Fixed
- Various bugs related to data processing and API interactions

## [0.8.0] 

### Added
- Alpha version with preliminary features
- Basic integration with Robinhood API
- Simple moving average crossover strategy

### Changed
- Updated project structure and dependencies

### Security
- Initial implementation of secure credential management"

print_color $GREEN "Documentation creation complete!"
print_color $BLUE "The following files have been created in the docs/ directory:"
ls -1 docs/

print_color $GREEN "Next steps:"
echo "1. Review and edit the generated documentation files as needed."
echo "2. Consider adding more specific details to each document based on your project's current state and future plans."
echo "3. Keep the documentation up-to-date as you develop new features or make changes to the project."
echo "4. Consider setting up a documentation hosting solution (e.g., Read the Docs, GitHub Pages) for easy access."

print_color $BLUE "Remember to commit these changes to your repository:"
echo "git add docs/"
echo "git commit -m \"Add comprehensive documentation\""
echo "git push origin main"

print_color $GREEN "Documentation process completed successfully!"