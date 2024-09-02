#!/bin/bash
# Set the default encoding to UTF-8

# Standard library imports
import os
import json
import logging
import random
import subprocess
import threading
from collections import deque
from datetime import datetime
from time import sleep, time
from dotenv import load_dotenv
import os

load_dotenv()  # This loads environment variables from a .env file
# Third-party imports
import ccxt
import numpy as np
import openai
import pandas as pd
import pandas_ta as ta
import pendulum
import pyotp
import pytz
import requests
import robin_stocks as rs
import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, request, jsonify
from scipy.stats import norm, zscore
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import logging
from config import Config
from utils import setup_logging


# Project-specific imports
import config

# Configure pandas settings
pd.set_option('future.no_silent_downcasting', True)

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import os
import base64
import binascii


WATCHLIST_NAMES = ["2024GPTd","AWP","100 Most Popular","Daily Movers","Upcoming Earnings","Energy & Water"]
import base64
import os
import base64
import binascii
import pyotp
import rs.robinhood.authentication

def _1_init():
    try:
        # Decode CURUSER with padding if necessary
        cur_user_encoded = os.environ['CURUSER']
        cur_user_encoded += '=' * (-len(cur_user_encoded) % 8)
        cur_user = base64.b32decode(cur_user_encoded, casefold=True).decode('utf-8')

        # Decode CURPASS
        cur_pass_encoded = os.environ['CURPASS']
        cur_pass_encoded += '=' * (-len(cur_pass_encoded) % 8)
        cur_pass = base64.b32decode(cur_pass_encoded, casefold=True).decode('utf-8')

        # Decode CURTOTP
        cur_totp_secret_encoded = os.environ['CURTOTP']
        cur_totp_secret_encoded += '=' * (-len(cur_totp_secret_encoded) % 8)
        cur_totp_secret = base64.b32decode(cur_totp_secret_encoded, casefold=True)

        # Generate TOTP
        totp = pyotp.TOTP(cur_totp_secret).now()

        # Print and attempt login
        print(f"{cur_user} / {totp}")
        login_response = rs.robinhood.authentication.login(cur_user, cur_pass, mfa_code=totp)
        print(login_response)

    except binascii.Error as e:
        print(f"Decoding failed: {e}")
    except KeyError as e:
        print(f"Environment variable {e} is missing.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return


def cancel_all_stockOrders(): logon=_1_init(); return print(rs.robinhood.cancel_all_stock_orders())

def cancel_all_cryptoOrders(): logon=_1_init(); return print(rs.robinhood.cancel_all_crypto_orders())

def log_info(message):
    # Log an informational message
    return print(f"INFO: {message}")

def log_error(message):
    # Log an error message
    return print(f"ERROR: {message}")

import pandas as pd
import os
from threading import Lock

def post_message_to_discord(message):
    """
    Post a message to a Discord channel using a webhook
    """
    data = {"content": message}
    stockls = "https://discord.com/api/webhooks/1231301813463023616/d6QSrL-ZCvUTDMzr9gdb4D4DjILBKJLAYMU_YgRDV9ZIpNP-E89q0tkzZwH1XAsPqZCY"
    response = requests.post(stockls, json=data)
    if response.status_code == 204:
        logging.info(f"Message posted to Discord successfully. {message}")
    else:
        logging.info(f"Failed to post message to Discord.{message}")
    return

# Function to calculate volatility-based lookback period
def adaptive_lookback(df):
    # Using ATR (Average True Range) as a measure of volatility
    df['tr'] = np.maximum(df['high_price'] - df['low_price'],
                          np.maximum(abs(df['high_price'] - df['close_price'].shift(1)),
                                     abs(df['low_price'] - df['close_price'].shift(1))))
    df['atr'] = df['tr'].rolling(window=14).mean()
    return int(max(20, min(100, 14 * (1 + df['atr'].mean() / df['atr'].std()))))


def write_dataframe_to_csv(df, filename):
    df.to_csv(filename, index=False)
    print(f"DataFrame written to {filename}")

def read_dataframe_from_csv(filename):
    return pd.read_csv(filename)

class ThreadSafeLookupTable:
    def __init__(self, filename):
        self.filename = filename
        self.lock = Lock()
        
    def update_table(self, new_df):
        with self.lock:
            write_dataframe_to_csv(new_df, self.filename)
    
    def lookup(self, key, column):
        with self.lock:
            df = read_dataframe_from_csv(self.filename)
            return df.loc[df[key].isin(column)].to_dict('records')


def fetch_open_positions():
    return pd.DataFrame(rs.robinhood.get_open_stock_positions())


def execute_stock_sell_order(ticker, quantity, last_trade_price, average_buy_price):
    quantity = float(quantity)
    last_trade_price = float(last_trade_price)
    average_buy_price = float(average_buy_price)
    # Initialize the lookup table
    lookup_table = ThreadSafeLookupTable('/etc/allPositions.csv')
    # Check if ticker exists in the lookup table
    ticks = []; ticks.append(ticker)
    position_info = lookup_table.lookup('symbol', ticks)
    if position_info:
        # If ticker exists, use the quantity from the lookup table
        actual_quantity = float(position_info[0]['quantity'])
        quantity = min(quantity, actual_quantity)  # Ensure we don't sell more than we have
    else:
        message = (f"Ticker {ticker} not found in open positions.")
        post_message_to_discord(message)
        logging.info(message)
        return
    try:
        sleep(random.randint(3, 5))
        order_result = rs.robinhood.orders.order(
            symbol=ticker, 
            quantity=round(quantity, 6), 
            side="sell", 
            timeInForce='gfd'
        )
        logging.warning(f"Order placed for {ticker}: {order_result}")
    except Exception as e:
        logging.error(f"{ticker}: Error placing sell order - {e}")
        
def execute_stock_buy_order(ticker, tradeSize):
    # Initialize the lookup table
    lookup_table = ThreadSafeLookupTable('/etc/allPositions.csv')
    ticks = []; ticks.append(ticker)
    position_info = lookup_table.lookup('symbol', ticks)
    tradeSize = trade_size = 1.11
    # Get stock quote information
    quote = rs.robinhood.get_stock_quote_by_symbol(ticker)
    last_trade_price = float(quote['last_trade_price'])
    bid_price = float(quote['bid_price'])
    ask_price = float(quote['ask_price'])
    previous_close = float(quote['previous_close'])
    trading_halted = quote['trading_halted']
    bid_size = float(quote['bid_size'])
    ask_size = float(quote['ask_size'])
    # Check if trading is halted
    if trading_halted:
        message = (f"Trading is halted for {ticker}. Skipping buy.")
        post_message_to_discord(message)
        logging.info(message)
        return
    # Use position info to adjust buy strategy
    if position_info:
        actual_quantity = float(position_info[0]['quantity'])
        average_buy_price = float(position_info[0]['average_buy_price'])
        logging.info(f"Existing position for {ticker}: Quantity = {actual_quantity}, Avg Buy Price = {average_buy_price}")
        # Avoid buying if already holding a significant quantity
        if actual_quantity > 0.5:  # Example threshold
            message = (f"Skipping buy for {ticker} due to large existing position.")
            post_message_to_discord(message)
            logging.info(message)
            return
        # Avoid buying if current price is much higher than average buy price
        if last_trade_price > average_buy_price * 1.1:
            message = (f"Skipping buy for {ticker} due to unfavorable price.")
            post_message_to_discord(message)
            logging.info(message)
            return
    try:
        sleep(random.randint(3, 5))
        try:
            order_result = rs.robinhood.orders.order_buy_fractional_by_price(
                symbol=ticker, amountInDollars=tradeSize
            )
            logging.error(order_result)
        except:
            order_result = rs.robinhood.orders.order(
                symbol=ticker, 
                quantity=round(float(last_trade_price / round(1.11, 4)),4), 
                side="buy", 
                timeInForce='gfd'
            )
            logging.error(order_result)
    except Exception as e:
        logging.error(f"{ticker}: Error placing buy order - {e}")
    return

def get_stock_historicals(rhSymbol, interval, span, logon):
    try:
        # replace any "-USD" suffix with ""
        rhSymbol = rhSymbol.replace("-USD","")
        # fetch historical data
        df = historical_data = pd.DataFrame(rs.robinhood.get_stock_historicals(inputSymbols=rhSymbol,interval=interval,span=span))
        df[['open_price','close_price','high_price','low_price','volume']] = df[['open_price','close_price','high_price','low_price','volume']].astype(float)
    except Exception as exception:
        logging.error(f"> {rhSymbol} {interval}:{span} - Error assembling dataframe; do not continue.")
        return
    # list of lengths for multiple calculations
    lengths = [2, 3, 5, 7, 9, 14]
    try:
        # perform multiple calculations using loop for efficiency
        for length in lengths:
            df = df.join(ta.mom(close=df['close_price'], length=length))
            df = df.join(ta.rsi(close=df['close_price'], length=length))
            df = df.join(ta.ema(close=df['close_price'], length=length))
        # perform other calculations
        try: df = df.join(ta.ema(close=df['close_price'], length=20))
        except: pass
        try: df = df.join(ta.ema(close=df['close_price'], length=50))
        except: pass
        try: df = df.join(ta.ema(close=df['close_price'], length=70))
        except: pass
        try: df = df.join(ta.sma(close=df['close_price'], length=50))
        except: pass
        df = df.join(ta.adx(high=df['high_price'], low=df['low_price'], close=df['close_price'], length=3))
        df = df.join(ta.macd(close=df['close_price'], fast=12, slow=26, signal=9))
        df = df.join(ta.psar(high=df['high_price'], low=df['low_price'], close=df['close_price']))
        df = df.join(ta.bbands(close=df['close_price'], length=5))
        df = df.join(ta.atr(high=df['high_price'], low=df['low_price'], close=df['close_price'], length=14))
        df = df.join(ta.kc(df['high_price'], df['low_price'], df['close_price'], 3))
        # replace any NaN values with 0
        df = df.fillna(value=0,axis=1)
        # cast prices into float type
        # List of all potential columns for conversion
        potential_columns = ['open_price', 'close_price', 'high_price', 'low_price', 'volume', 
                            'MOM_2', 'RSI_2', 'EMA_2', 'MOM_3', 'RSI_3', 'EMA_3', 'MOM_5', 'RSI_5', 'EMA_5', 
                            'MOM_7', 'RSI_7', 'EMA_7', 'MOM_9', 'RSI_9', 'EMA_9', 'MOM_14', 'RSI_14', 'EMA_14', 
                            'EMA_20', 'EMA_50', 'EMA_70', 'SMA_50', 'ADX_3', 'DMP_3', 'DMN_3', 
                            'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'PSARl_0.02_0.2', 'PSARs_0.02_0.2', 
                            'PSARaf_0.02_0.2', 'PSARr_0.02_0.2', 'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0', 
                            'BBB_5_2.0', 'BBP_5_2.0', 'ATRr_14', 'KCLe_3_2', 'KCBe_3_2', 'KCUe_3_2']
        # Filter the list to include only columns that exist in the DataFrame
        columns_to_convert = [col for col in potential_columns if col in df.columns]
        # Perform the type conversion on the filtered list of columns
        df[columns_to_convert] = df[columns_to_convert].astype(float)
    except Exception as e:
        logging.error(f"Error performing calculations: {e}")
        return
    return df

import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
import numpy as np

def machine_learning_volatility(df):
    try:
        # Step 1: Calculate returns
        df['returns'] = df['close_price'].pct_change()
        if df['returns'].isna().all():
            raise ValueError("The 'returns' column is entirely NaN. Check the 'close_price' data.")
        # Step 2: Prepare feature matrix X and target variable y
        X = df[['returns']].shift(1, axis=0).dropna(axis=0)  # Specify axis explicitly
        y = df['returns'].rolling(window=30).std().dropna(axis=0)  # Specify axis explicitly
        # Align X and y indices to ensure they have the same number of samples
        X, y = X.align(y, join='inner', axis=0)
        
        if X.empty or y.empty:
            raise ValueError("The feature matrix X or the target variable y is empty after alignment. Check the data preprocessing steps.")
        
        logging.info(f"Shape of X after alignment: {X.shape}")
        logging.info(f"Shape of y after alignment: {y.shape}")
        # Step 3: Initialize and train the RandomForest model
        model = RandomForestRegressor()
        try:
            model.fit(X, y)
        except ValueError as e:
            raise ValueError(f"Error in model fitting: {str(e)}")
        
        # Step 4: Predict volatility using the trained model
        try:
            prediction = model.predict(X.iloc[[-1]])[0]
        except IndexError as e:
            raise IndexError(f"Prediction failed: {str(e)}")
        except NotFittedError as e:
            raise NotFittedError(f"Model is not fitted properly: {str(e)}")

        return prediction
    except KeyError as e:
        logging.error(f"KeyError - Check if 'close_price' column exists in the DataFrame: {str(e)}")
    except ValueError as e:
        logging.error(f"ValueError - {str(e)}")
    except NotFittedError as e:
        logging.error(f"NotFittedError - {str(e)}")
    except Exception as e:
        logging.error(f"An unexpected error occurred in machine_learning_volatility: {str(e)}")
        return np.nan  # Return NaN if an error occurs

def calculate_volatility_clustering(df, window=30):
    df['volatility'] = df['close_price'].pct_change().rolling(window).std()
    df['volatility_shifted'] = df['volatility'].shift(1)
    clustering_factor = df['volatility'].corr(df['volatility_shifted'])
    return clustering_factor * df['volatility'].iloc[-1]


import logging
from functools import wraps
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Error logging decorator
def log_exceptions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    return wrapper

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import IsolationForest
from scipy import stats
from textblob import TextBlob
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from scipy import stats


def analyze_stock(rhSymbol, df, quantity, average_buy_price, previous_close, risk_tolerance=0.02):
    """
    Modified stock analysis with entry and exit strategies, incorporating machine learning and risk management.
    
    :param rhSymbol: str, Robinhood symbol for the stock
    :param df: DataFrame containing historical stock data and technical indicators
    :param quantity: float, current quantity of stock held
    :param average_buy_price: float, average purchase price of the stock
    :param risk_tolerance: float, maximum allowed loss as a fraction of position value
    :return: dict, containing 'action' (Buy, Sell, or Hold), 'stop_loss' price, and explanatory message
    """
    try:
        # Define possible column names for each indicator
        column_mappings = {
            'EMA_9': ['EMA_9', 'EMA_10'], 'EMA_20': ['EMA_20', 'EMA_21'],
            'EMA_50': ['EMA_50', 'EMA_55'], 'EMA_200': ['EMA_200', 'SMA_200'],
            'RSI_14': ['RSI_14', 'RSI_13', 'RSI_15'], 'ADX_14': ['ADX_14', 'ADX_13', 'ADX_15', 'ADX_3'],
            'MACD_12_26_9': ['MACD_12_26_9', 'MACD'], 'MACDs_12_26_9': ['MACDs_12_26_9', 'MACDs'],
            'BBL_20_2.0': ['BBL_20_2.0', 'BBL_5_2.0', 'BBL'],
            'BBM_20_2.0': ['BBM_20_2.0', 'BBM_5_2.0', 'BBM'],
            'BBU_20_2.0': ['BBU_20_2.0', 'BBU_5_2.0', 'BBU']
        }
        def safe_format(value, format_spec='.2f'):
            """Safely format a value, returning 'N/A' for None values."""
            if value is None:
                return 'N/A'
            try:
                return f"{value:{format_spec}}"
            except (ValueError, TypeError):
                return str(value)
            
        def adaptive_lookback(df):
            volatility = df['close_price'].pct_change().rolling(window=30).std().iloc[-1]
            return int(max(min(200, len(df) // 2), 50 * volatility))
        
        lookback = adaptive_lookback(df)

        def get_column(mapping):
            return next((col for col in mapping if col in df.columns), None)

        # Ensure all relevant columns are float type
        float_columns = ['open_price', 'close_price', 'high_price', 'low_price', 'volume']
        float_columns.extend([get_column(mapping) for mapping in column_mappings.values() if get_column(mapping)])
        for col in float_columns:
            if col in df.columns:
                df[col] = df[col].astype(float)

        if len(df) < 200:
            return {"action": "Hold", "stop_loss": None, "message": "Insufficient data for analysis"}

        latest = df.iloc[-1]
        current_time = pd.Timestamp.now().strftime('%B %d, %Y')
        
        short_term_trend, medium_term_trend, long_term_trend, rsi_trend, adx, macd_crossover, lower_bb, upper_bb, bb_width, price_prediction, sma_forecast, volume_trend, regime, volatility_forecast, rsi, macd_histogram, adx, excess_return, signal_col = init_decision_making(df,latest,column_mappings,get_column,lookback)
        buy_signals, sell_signals, sentiment_score = decision_making(rhSymbol, latest, short_term_trend, medium_term_trend, long_term_trend, rsi_trend, adx, macd_crossover, lower_bb, upper_bb, bb_width, price_prediction, sma_forecast, volume_trend, regime)
        # Exit Strategy
        exit_signal, exit_reason = advanced_exit_strategy(df, quantity, average_buy_price, risk_tolerance)
        # Calculate the trade size using the updated function
        trade_size = calculate_trade_size(
            price=latest['close_price'], 
            sentiment_score=sentiment_score,
            volatility_forecast=volatility_forecast,
            market_regime=regime, 
            risk_tolerance=0.02
        )
        final_decision_making(rhSymbol, safe_format, quantity, rsi, macd_histogram, adx, volatility_forecast, df, sentiment_score, bb_width, exit_signal, latest, average_buy_price, price_prediction, excess_return, sell_signals, buy_signals, signal_col, trade_size)
    except Exception as e:
        error_message = f"Error in analyze_stock for {rhSymbol}: {str(e)}"
        logging.error(f"ANALYSIS ERROR: {rhSymbol} - {error_message}")
        return {"action": "Hold", "stop_loss": None, "message": error_message}

def final_decision_making(rhSymbol, safe_format, quantity, rsi, macd_histogram, adx, volatility_forecast, df, sentiment_score, bb_width, exit_signal, latest, average_buy_price, price_prediction, excess_return, sell_signals, buy_signals, signal_col, trade_size):
    try: 
        # Decision Making
        if exit_signal and (price_prediction < 0 and excess_return < 0) and sell_signals > buy_signals and sell_signals > 2:
            action = "Sell"
            message = (f"{action} SIGNAL: {rhSymbol} - Current: ${safe_format(latest['close_price'])}, Avg Buy: ${safe_format(average_buy_price)}, Quantity: {quantity}, Size: ${safe_format(trade_size)}, {buy_signals:.4f}, {sell_signals:.4f}, {price_prediction:.4f}, {excess_return:.4f}")
            post_message_to_discord(f":red_circle: {message}")
            logging.error(message)
            # threading.Thread(target=execute_stock_sell_order, args=(rhSymbol, quantity, latest['close_price'], average_buy_price)).start()
            return {"action": action, "message": message}
        elif (price_prediction > 0 and excess_return > -0.04) and buy_signals > sell_signals and buy_signals > 0:
            action = "Buy"  
            message = (f"{action} SIGNAL: {rhSymbol} - Current: ${safe_format(latest['close_price'])}, Avg Buy: ${safe_format(average_buy_price)}, Quantity: {quantity}, Size: ${safe_format(trade_size)}, {buy_signals:.4f}, {sell_signals:.4f}, {price_prediction:.4f}, {excess_return:.4f}")
            post_message_to_discord(f":green_circle: {message}")
            logging.error(message)
            # threading.Thread(target=execute_stock_buy_order, args=(rhSymbol, 1.11)).start()
            return {"action": action,"message": message}
        else:
            if quantity > 1 and latest > average_buy_price and price_prediction < 0 and excess_return < 0:
                threading.Thread(target=execute_stock_sell_order, args=(rhSymbol, quantity, latest['close_price'], average_buy_price)).start()
                action = "Sell; quantity and avgBuyPrice"
                message = (f"{action} SIGNAL: {rhSymbol} - Current: ${safe_format(latest['close_price'])}, Avg Buy: ${safe_format(average_buy_price)}, Quantity: {quantity}, Size: ${safe_format(trade_size)}, {buy_signals:.4f}, {sell_signals:.4f}, {price_prediction:.4f}, {excess_return:.4f}")
                post_message_to_discord(f":red_circle: {message}")
                logging.error(message)
                # threading.Thread(target=execute_stock_sell_order, args=(rhSymbol, quantity, latest['close_price'], average_buy_price)).start()
                return {"action": action, "message": message}
            action = "Hold"
            message = (f"{action} SIGNAL: {rhSymbol} - Current: ${safe_format(latest['close_price'])}, Avg Buy: ${safe_format(average_buy_price)}, Quantity: {quantity}, Size: ${safe_format(trade_size)}, {buy_signals:.4f}, {sell_signals:.4f}, {price_prediction:.4f}, {excess_return:.4f}")
            logging.error(message)
            return {"action": action, "message": message}
    except Exception as e: 
        import sys
        exc_type, exc_obj, exc_tb = sys.exc_info()
        return logging.error(f"Error {exc_tb.tb_lineno}")

def init_decision_making(df,latest,column_mappings,get_column,lookback):
    try: 
        # 1. Enhanced Trend Analysis
        ema_cols = [get_column(column_mappings[f'EMA_{period}']) for period in [9, 20, 50, 200]]
        ema_values = [latest[col] for col in ema_cols if col]
        short_term_trend = ema_values[0] > ema_values[1] if len(ema_values) > 1 else None
        medium_term_trend = ema_values[1] > ema_values[2] if len(ema_values) > 2 else None
        long_term_trend = ema_values[2] > ema_values[3] if len(ema_values) > 3 else None

        # 2. Advanced Momentum
        rsi_col = get_column(column_mappings['RSI_14'])
        rsi = latest[rsi_col] if rsi_col else None
        rsi_trend = rsi < df[rsi_col].rolling(lookback).mean().iloc[-1] if rsi is not None else None

        # 3. Trend Strength
        adx_col = get_column(column_mappings['ADX_14'])
        adx = latest[adx_col] if adx_col else None

        # 4. MACD with Signal Line Crossover
        macd_col, signal_col = get_column(column_mappings['MACD_12_26_9']), get_column(column_mappings['MACDs_12_26_9'])
        if macd_col and signal_col:
            macd_line, signal_line = latest[macd_col], latest[signal_col]
            macd_histogram = macd_line - signal_line
            macd_crossover = np.sign(df[macd_col] - df[signal_col]).diff().iloc[-1]
        else:
            macd_line = signal_line = macd_histogram = macd_crossover = None

        # 5. Dynamic Bollinger Bands
        bb_cols = [get_column(column_mappings[f'BB{band}_20_2.0']) for band in ['L', 'M', 'U']]
        if all(bb_cols):
            lower_bb, middle_bb, upper_bb = [latest[col] for col in bb_cols]
            bb_width = (upper_bb - lower_bb) / middle_bb
        else:
            lower_bb = middle_bb = upper_bb = bb_width = None

        # 6. Volume Analysis
        volume_sma = df['volume'].rolling(window=20).mean().iloc[-1]
        volume_trend = latest['volume'] > volume_sma * 1.5

        # 7. Machine Learning Prediction
        RISK_FREE_RATE = .05 # 5%
        features = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
        features.extend([col for col in df.columns if col.startswith(('EMA', 'RSI', 'ADX', 'MACD', 'BB'))])
        X, y = df[features].iloc[-lookback:], df['close_price'].pct_change().shift(-1).iloc[-lookback:]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X[:-1], y[:-1])
        price_prediction = model.predict(X.iloc[[-1]])[0]

        # Adjust price prediction for RFR
        excess_return = price_prediction - RISK_FREE_RATE

        # 8. Simple Moving Average Forecast (replacing ARIMA)
        sma_forecast = df['close_price'].rolling(window=30).mean().iloc[-1]

        # 9. Simple Volatility Forecast (replacing GARCH)
        volatility_forecast = calculate_volatility_clustering(df)

        # 10. Market Regime Detection
        regime = detect_market_regime(df)
    except Exception as e: 
        import sys
        exc_type, exc_obj, exc_tb = sys.exc_info()
        return logging.error(f"Error {exc_tb.tb_lineno}")
    return short_term_trend, medium_term_trend, long_term_trend, rsi_trend, adx, macd_crossover, lower_bb, upper_bb, bb_width, price_prediction, sma_forecast, volume_trend, regime, volatility_forecast, rsi, macd_histogram, adx, excess_return, signal_col

def decision_making(rhSymbol, latest, short_term_trend, medium_term_trend, long_term_trend, rsi_trend, adx, macd_crossover, lower_bb, upper_bb, bb_width, price_prediction, sma_forecast, volume_trend, regime): 
    try: 
        # Decision Making
        buy_signals = sell_signals = 0
        # Trend signals
        if all([short_term_trend, medium_term_trend, long_term_trend]):
            buy_signals += 1
        elif not any([short_term_trend, medium_term_trend, long_term_trend]):
            sell_signals += 1

        # RSI signals
        if rsi_trend is not None:
            buy_signals += 1 if rsi_trend else 0
            sell_signals += 1 if not rsi_trend else 0

        # ADX signal
        if adx > 25:
            if short_term_trend and medium_term_trend:
                buy_signals += 1
            elif not short_term_trend and not medium_term_trend:
                sell_signals += 1

        # MACD signals
        if macd_crossover is not None:
            buy_signals += 1 if macd_crossover > 0 else 0
            sell_signals += 1 if macd_crossover < 0 else 0

        # Bollinger Bands signals
        if all([lower_bb, upper_bb, bb_width]):
            if latest['close_price'] < lower_bb and bb_width > 0.1:
                buy_signals += 1
            elif latest['close_price'] > upper_bb and bb_width > 0.1:
                sell_signals += 1

        # Volume confirmation
        if volume_trend:
            if buy_signals > sell_signals:
                buy_signals += 1
            elif sell_signals > buy_signals:
                sell_signals += 1

        # Machine Learning and Time Series Forecasts
        if price_prediction > 0 and sma_forecast > latest['close_price']:
            buy_signals += (price_prediction*10)
        elif price_prediction < 0 and sma_forecast < latest['close_price']:
            sell_signals += (price_prediction*10)

        # Market Regime
        if regime == 0:  # Bullish
            buy_signals += regime
        elif regime == 2:  # Bearish
            sell_signals += regime

        # 11. Sentiment Analysis
        news_data = rs.robinhood.stocks.get_news(rhSymbol)
        if news_data:
            summaries = [article['summary'] for article in news_data if article.get('summary')]
            sentiments = [TextBlob(summary).sentiment.polarity for summary in summaries]
            sentiment_score = sum(sentiments) / len(sentiments) if sentiments else 0
        else:
            sentiment_score = 0
        # Sentiment signals
        if sentiment_score > 0.2:
            buy_signals += sentiment_score
        elif sentiment_score < -0.2:
            sell_signals += sentiment_score
    except Exception as e: 
        import sys
        exc_type, exc_obj, exc_tb = sys.exc_info()
        return logging.error(f"Error {exc_tb.tb_lineno}")
    return buy_signals, sell_signals, sentiment_score

def advanced_exit_strategy(df, quantity, entry_price, risk_tolerance=0.02):
    latest = df.iloc[-1]
    close_price = latest['close_price']
    
    # 1. Stop Loss and Trailing Stop
    basic_stop_loss = entry_price * (1 - risk_tolerance)
    highest_price = df['high_price'].max()
    trailing_stop = highest_price * (1 - risk_tolerance)
    stop_loss = max(basic_stop_loss, trailing_stop)
    
    if close_price <= stop_loss:
        return True, "Stop loss triggered"
    
    # 2. Volatility-Based Stop Loss
    volatility = df['close_price'].pct_change().std()
    vol_stop = entry_price * (1 - volatility * 2)
    
    if close_price <= vol_stop:
        return True, "Volatility-based stop loss triggered"
    
    # 3. Trend Reversal Detection
    ema_short = df['close_price'].ewm(span=10, adjust=False).mean()
    ema_long = df['close_price'].ewm(span=30, adjust=False).mean()
    trend_reversal = (ema_short.iloc[-2] > ema_long.iloc[-2] and ema_short.iloc[-1] <= ema_long.iloc[-1])
    
    if trend_reversal:
        return True, "Trend reversal detected"
    
    # 4. Momentum Exhaustion
    if 'RSI_14' in df.columns:
        rsi = latest['RSI_14']
        if rsi > 70:
            return True, "Momentum exhaustion (RSI)"
    
    # 5. Volume Climax
    volume_sma = df['volume'].rolling(window=20).mean()
    if latest['volume'] > volume_sma.iloc[-1] * 3:
        return True, "Volume climax detected"
    
    # 6. Take Profit
    take_profit = entry_price * (1 + risk_tolerance * 2)
    if close_price >= take_profit:
        return True, "Take profit target reached"
    
    # 7. Time-based Exit
    days_held = len(df) - 1
    if days_held > 30:  # Adjust based on your trading timeframe
        return True, "Maximum holding period reached"
    
    return False, "No exit signal detected"

def get_stock_recommendation(rhSymbol, interval, span, logon, quantity, average_buy_price, previous_close, risk_tolerance=0.05):
    """
    Get a stock recommendation based on trend following and risk management.
    
    :param rhSymbol: str, Robinhood symbol for the stock
    :param interval: str, time interval for historical data
    :param span: str, time span for historical data
    :param logon: object, Robinhood logon instance
    :param risk_tolerance: float, maximum allowed loss as a fraction of position value
    :return: dict, containing 'action' (Buy, Sell, or Hold), 'stop_loss' price, and 'message'
    """
    try:
        df = get_stock_historicals(rhSymbol, interval, span, logon)
        
        if df is None or df.empty:
            return {"action": "Hold", "stop_loss": None, "message": "Insufficient data"}
        
        recommendation = analyze_stock(rhSymbol, df, quantity, average_buy_price, previous_close, risk_tolerance)
        return recommendation
    
    except Exception as e:
        error_message = f"Error in get_stock_recommendation for {rhSymbol}: {str(e)}"
        logging.error(error_message)
        return {"action": "Hold", "stop_loss": None, "message": error_message}

import math 
def calculate_trade_size(price, sentiment_score, volatility_forecast, market_regime, risk_tolerance=0.02, min_trade_value=1.11, max_position_pct=0.1):
    """
    Calculate the trade size considering current cash, portfolio value, market sentiment, volatility, and market regime.
    Ensures no trades less than $1.11 are executed.
    
    :param price: float, current price of the stock
    :param sentiment_score: float, sentiment score from analysis (positive or negative)
    :param volatility_forecast: float, expected volatility of the stock
    :param market_regime: int, market regime detection result (0 for bullish, 1 for neutral, 2 for bearish)
    :param risk_tolerance: float, base risk tolerance level (default is 0.02)
    :param min_trade_value: float, minimum trade value (default is $1.11)
    :param max_position_pct: float, maximum percentage of portfolio for a single position (default is 0.1)
    :return: float, the number of shares to buy (may be fractional), or 0 if no trade should be made
    """

    # Get the current cash and portfolio value
    available_cash = get_current_cash()
    portfolio_value = get_portfolio_value()
    
    # Check if we have enough cash for the minimum trade
    if available_cash < min_trade_value:
        return 1.11
    
    # Calculate minimum number of shares to meet min_trade_value
    min_shares = math.ceil((min_trade_value / price) * 10000) / 10000  # Round up to 4 decimal places

    # Adjust risk tolerance based on cash to portfolio ratio
    cash_ratio = available_cash / portfolio_value
    adjusted_risk_tolerance = risk_tolerance * (1 + cash_ratio)
    
    # Base position size using a simplified Kelly Criterion
    base_position_size = min(1.0, adjusted_risk_tolerance / volatility_forecast)
    
    # Adjust position size based on sentiment analysis
    sentiment_multiplier = 1 + sentiment_score if abs(sentiment_score) > 0.2 else 1
    adjusted_position_size = base_position_size * sentiment_multiplier
    
    # Market Regime Adjustment
    regime_multipliers = {0: 1.5, 1: 1.0, 2: 0.5}
    adjusted_position_size *= regime_multipliers.get(market_regime, 1.0)
    
    # Calculate the dollar amount to invest based on the adjusted position size and portfolio value
    max_investment = portfolio_value * max_position_pct
    investable_amount = min(available_cash, portfolio_value * adjusted_position_size, max_investment)
    
    # Calculate the trade size in shares
    trade_size = investable_amount / price
    
    # Ensure the trade meets the minimum value
    if trade_size < min_shares:
        if available_cash >= min_trade_value:
            trade_size = min_shares
        else:
            return 1.11  # Not enough cash for minimum trade
    
    # Ensure we don't exceed available cash
    max_affordable_shares = math.floor((available_cash / price) * 10000) / 10000  # Round down to 4 decimal places
    trade_size = min(trade_size, max_affordable_shares)
    
    # Final check to ensure we're not making a trade less than min_trade_value
    if trade_size < min_shares:
        return 1.11
    if trade_size < 1.11:
        return 1.11
    return trade_size

def detect_market_regime(df):
    """
    Detect the market regime (Bullish, Bearish, Sideways) based on technical indicators like Moving Averages,
    Bollinger Bands, and RSI.
    
    :param df: DataFrame containing historical stock data and technical indicators
    :return: int, 0 for Bullish, 1 for Sideways, 2 for Bearish
    """
    # Calculate necessary indicators if not already in the DataFrame
    if 'EMA_50' not in df.columns:
        df['EMA_50'] = df['close_price'].ewm(span=50, adjust=False).mean()
    if 'EMA_200' not in df.columns:
        df['EMA_200'] = df['close_price'].ewm(span=200, adjust=False).mean()
    if 'BBU_20_2.0' not in df.columns or 'BBL_20_2.0' not in df.columns:
        df['BBM_20_2.0'] = df['close_price'].rolling(window=20).mean()
        df['BBU_20_2.0'] = df['BBM_20_2.0'] + 2 * df['close_price'].rolling(window=20).std()
        df['BBL_20_2.0'] = df['BBM_20_2.0'] - 2 * df['close_price'].rolling(window=20).std()
    if 'RSI_14' not in df.columns:
        delta = df['close_price'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
    
    latest = df.iloc[-1]

    # Rules for Bullish, Bearish, and Sideways Regimes
    is_bullish = latest['close_price'] > latest['EMA_50'] and latest['EMA_50'] > latest['EMA_200'] and 30 < latest['RSI_14'] < 70
    is_bearish = latest['close_price'] < latest['EMA_50'] and latest['EMA_50'] < latest['EMA_200'] and latest['RSI_14'] > 70
    is_sideways = (abs(latest['close_price'] - latest['BBM_20_2.0']) / latest['close_price']) < 0.01 and latest['RSI_14'] >= 30 and latest['RSI_14'] <= 70

    if is_bullish:
        return 0  # Bullish
    elif is_bearish:
        return 2  # Bearish
    elif is_sideways:
        return 1  # Sideways
    else:
        # If no clear trend, default to Sideways
        return 1  # Sideways

def _main_open_positions():
    logon=_1_init()
    try:
        full_watchlist = merged_df = open_positions = fetch_open_positions()
        all_data = []
        #if (current_vix > (mid-(mid*.1)) ):
        for index, row in merged_df.iterrows():
            try:
                # Extract common data fields
                instrument_id = row.get('object_id') or row.get('instrument_id')
                instrument = row.get('id')
                quantity = float(row.get('open_positions') or row.get('quantity'))
                average_buy_price = float(row.get('price') or row.get('average_buy_price'))
                # Fetch additional data
                stock_quote = rs.robinhood.get_stock_quote_by_id(instrument_id)
                last_trade_price = float(stock_quote.get('last_trade_price'))
                previous_close = float(stock_quote.get('previous_close'))
                rh_symbol = str(stock_quote.get('symbol'))
                get_stock_recommendation(rh_symbol, "hour", "3month", logon, quantity, average_buy_price, previous_close, 0.05)
            except Exception as e:
                logging.error(f"Error processing row {index}: {e}")
    except Exception as e:
        logging.error(f"An error occurred in main_open_positions: {e}")
    return

# Placeholder for functions to get current cash and portfolio value
def get_current_cash():
    # Implement this function to return current cash available
    return float(rs.robinhood.profiles.load_account_profile(info='crypto_buying_power'))

def get_portfolio_value():
    # Implement this function to return the current portfolio value 
    return float(rs.robinhood.profiles.load_account_profile(info='portfolio_cash'))


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    # Initialize the BackgroundScheduler with the specified timezone
    central_tz = pendulum.timezone('America/Chicago')
    scheduler = BackgroundScheduler(timezone=central_tz)
    scheduler.add_job(cancel_all_stockOrders, trigger='cron', day_of_week='*', hour='22', minute='11')
    scheduler.add_job(_main_open_positions, trigger='cron', day_of_week='mon-fri', hour='6,11,14', minute='11')

    # Start the scheduler
    scheduler.start()
    return



_1_init()
# Usage example
filename = '/etc/allPositions.csv'

# Write the initial DataFrame to CSV
initial_df = pd.DataFrame(rs.robinhood.get_open_stock_positions())
write_dataframe_to_csv(initial_df, filename)


if __name__ == '__main__':
    main()

from time import sleep

try:
    # Simulate application activity
    while True:
        sleep(60)
        logging.info(f"Handler Alive. ")
except Exception as e: print(e)

