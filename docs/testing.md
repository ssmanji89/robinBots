# Testing Guide

## Test Structure

Our test suite is organized as follows:

- `tests/unit/`: Unit tests for individual functions and classes
- `tests/integration/`: Integration tests for module interactions
- `tests/e2e/`: End-to-end tests simulating real trading scenarios

## Running Tests

To run the entire test suite:

```
pytest
```

To run a specific test file:

```
pytest tests/unit/test_analysis.py
```

For verbose output:

```
pytest -v
```

## Writing Tests

1. Create a new test file in the appropriate directory (unit, integration, or e2e).
2. Import the necessary modules and the function/class you're testing.
3. Write test functions prefixed with `test_`.
4. Use assertions to check expected outcomes.

Example:

```python
# tests/unit/test_analysis.py
from src.analysis import calculate_rsi

def test_calculate_rsi():
    data = [10, 12, 15, 14, 13, 16, 17, 15]
    rsi = calculate_rsi(data, period=14)
    assert 0 <= rsi <= 100
    assert round(rsi, 2) == 57.14  # Expected RSI value
```

## Mocking

Use the `unittest.mock` module to mock external dependencies, especially API calls:

```python
from unittest.mock import patch
from src.robinhood_client import RobinhoodClient

@patch('src.robinhood_client.rs.robinhood.get_stock_quote_by_symbol')
def test_get_stock_price(mock_get_quote):
    mock_get_quote.return_value = {'last_trade_price': '150.00'}
    client = RobinhoodClient()
    price = client.get_stock_price('AAPL')
    assert price == 150.00
```

## Test Coverage

To check test coverage:

```
pytest --cov=src tests/
```

Aim for at least 80% coverage for critical components.

## Continuous Integration

We use GitHub Actions for CI. The workflow is defined in `.github/workflows/ci.yml`.

It runs on every push and pull request, executing:
1. Linting with Flake8
2. Unit tests with pytest
3. Integration tests
4. Coverage report

## Performance Testing

For performance-critical functions, use the `timeit` module:

```python
import timeit

def test_analysis_performance():
    setup = "from src.analysis import analyze_stock; import pandas as pd"
    stmt = "analyze_stock('AAPL', pd.DataFrame({'close': range(1000)})"
    result = timeit.timeit(stmt, setup, number=100)
    assert result < 1.0  # Should complete in less than 1 second
```

## Security Testing

Regularly scan dependencies for vulnerabilities:

```
safety check
```

## Troubleshooting Tests

- Use `pytest -vv` for more detailed output.
- Use `pytest.set_trace()` to debug within a test.
- Check log files for additional information during test runs.

Remember, thorough testing is crucial for maintaining the reliability and performance of our trading system. Always write tests for new features and bug fixes.
