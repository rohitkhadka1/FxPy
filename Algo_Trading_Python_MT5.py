import MetaTrader5 as mt5
from dotenv import load_dotenv 
import pandas as pd 
import numpy as np
import os 

load_dotenv()
login = 76848203
password = os.environ.get("MT5_PASSWORD")
path = r"C:\Users\Public\Desktop\MetaTrader 5 EXNESS.lnk"
servidor = "Exness-MT5Trial5"

# Initialize MT5 with error checking
mt5.initialize(login=login, password=password, server=servidor, path=path)
# print("MT5 initialization failed, error code:", mt5.last_error())


print("Login Successful")

# Use consistent symbol name
symbol = "EUR/USD"  # or "EURUSDm" - check which one is correct for your broker

# Get symbol info to check if it exists and get proper tick size
symbol_info = mt5.symbol_info(symbol)
if symbol_info is None:
    print(f"Symbol {symbol} not found")
    mt5.shutdown()
    exit()

# Get rates with error checking
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 864)
if rates is None:
    print("Failed to get rates, error code:", mt5.last_error())
    mt5.shutdown()
    exit()

rates_frame = pd.DataFrame(rates)
print("Rates retrieved successfully")
print(f"Number of rates: {len(rates_frame)}")

# Convert time to datetime
rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')

# Calculate max and min values
max_value = rates_frame['high'].max() 
min_value = rates_frame['low'].min()

print(f"Price range: {min_value} - {max_value}")

# Check if symbol allows limit orders
if not symbol_info.trade_mode & mt5.SYMBOL_TRADE_MODE_FULL:
    print("Trading is not allowed for this symbol")
    mt5.shutdown()
    exit()

# Buy limit order (at minimum price)
buy_request = {
    "action": mt5.TRADE_ACTION_PENDING,
    "symbol": symbol,
    "type": mt5.ORDER_TYPE_BUY_LIMIT,
    "price": min_value,
    "volume": 1.0,
    "comment": "Buy Limit Order",
    "type_filling": mt5.ORDER_FILLING_RETURN,  # Changed from IOC
    "magic": 234000,  # Add magic number for order identification
}

# Send buy order
buy_result = mt5.order_send(buy_request)
if buy_result.retcode != mt5.TRADE_RETCODE_DONE:
    print(f"Buy order failed, error code: {buy_result.retcode}")
    print(f"Error comment: {buy_result.comment}")
else:
    print(f"Buy order placed successfully, ticket: {buy_result.order}")

# Sell limit order (at maximum price)
sell_request = {
    "action": mt5.TRADE_ACTION_PENDING,
    "symbol": symbol,
    "type": mt5.ORDER_TYPE_SELL_LIMIT,
    "price": max_value,
    "volume": 1.0,
    "comment": "Sell Limit Order",
    "type_filling": mt5.ORDER_FILLING_RETURN,  # Changed from IOC
    "magic": 234000,  # Add magic number for order identification
}

# Send sell order
sell_result = mt5.order_send(sell_request)
if sell_result.retcode != mt5.TRADE_RETCODE_DONE:
    print(f"Sell order failed, error code: {sell_result.retcode}")
    print(f"Error comment: {sell_result.comment}")
else:
    print(f"Sell order placed successfully, ticket: {sell_result.order}")

# Properly shutdown MT5
mt5.shutdown()