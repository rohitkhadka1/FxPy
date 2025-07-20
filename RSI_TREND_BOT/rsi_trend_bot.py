import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime
import os
import csv
import logging
from dotenv import load_dotenv
load_dotenv("E:\FxPy\.env")

# === LOGGING SETUP ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)

# === CONFIGURATION ===
symbol = "BTCUSDm"  # Use "BTCUSDm" for MetaTrader 5
timeframe = mt5.TIMEFRAME_M5
lot = 0.01  # Reduced lot size for safety
rsi_period = 14
rsi_oversold = 30
rsi_overbought = 70
risk_pips = 20
pip_value = 0.0001 if "JPY" not in symbol else 0.01
sl_distance = risk_pips * pip_value
tp_distance = sl_distance * 2
check_interval = 300  # 5 minutes in seconds

# Account credentials (use environment variables for security)
account = int(os.getenv('MT5_ACCOUNT'))
password = os.getenv('MT5_PASSWORD')
server = os.getenv('MT5_SERVER')

# === INITIALIZE MT5 ===
def initialize_mt5():
    """Initialize MetaTrader 5 connection"""
    if not mt5.initialize():
        logging.error(f"MT5 initialization failed: {mt5.last_error()}")
        return False
    
    if not mt5.login(account, password, server):
        logging.error("Failed to connect to MetaTrader account!")
        mt5.shutdown()
        return False
    
    # Check symbol availability
    if not mt5.symbol_select(symbol, True):
        logging.error(f"Failed to select {symbol}: {mt5.last_error()}")
        # Try to get available symbols for debugging
        symbols = mt5.symbols_get()
        if symbols:
            available_symbols = [s.name for s in symbols[:20]]
            logging.info(f"Available symbols (first 20): {available_symbols}")
        return False
    
    logging.info(f"MT5 initialized successfully. Connected to account {account}")
    return True

# === SIGNAL GENERATION ===
def calculate_rsi(prices, period=14):
    """Calculate RSI using pandas for better performance"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_rsi_signal():
    """Get RSI-based trading signal"""
    try:
        # Get more data to ensure accurate RSI calculation
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, rsi_period * 2)
        
        if rates is None:
            logging.error(f"Failed to get rates: {mt5.last_error()}")
            return "hold"
        
        if len(rates) < rsi_period + 2:
            logging.warning("Not enough historical data for RSI calculation")
            return "hold"
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Calculate RSI
        df['rsi'] = calculate_rsi(df['close'], rsi_period)
        
        # Get current and previous RSI values
        current_rsi = df['rsi'].iloc[-1]
        prev_rsi = df['rsi'].iloc[-2]
        current_price = df['close'].iloc[-1]
        
        if pd.isna(current_rsi) or pd.isna(prev_rsi):
            logging.warning("RSI calculation returned NaN values")
            return "hold"
        
        logging.info(f"Price: {current_price:.5f}, Previous RSI: {prev_rsi:.2f}, Current RSI: {current_rsi:.2f}")
        
        # Generate signals based on RSI crossovers
        if prev_rsi <= rsi_oversold and current_rsi > rsi_oversold:
            logging.info("RSI oversold crossover detected - BUY signal")
            return "buy"
        elif prev_rsi >= rsi_overbought and current_rsi < rsi_overbought:
            logging.info("RSI overbought crossover detected - SELL signal")
            return "sell"
        else:
            return "hold"
            
    except Exception as e:
        logging.error(f"Error in get_rsi_signal: {e}")
        return "hold"

# === POSITION MANAGEMENT ===
def check_existing_positions():
    """Check if there are existing positions for the symbol"""
    positions = mt5.positions_get(symbol=symbol)
    return len(positions) if positions else 0

def close_all_positions():
    """Close all open positions for the symbol"""
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return True
    
    for position in positions:
        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": position.ticket,
            "deviation": 20,
            "magic": 123456,
            "comment": "Close by RSI Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(close_request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Failed to close position {position.ticket}: {result.retcode}")
            return False
        else:
            logging.info(f"Closed position {position.ticket}")
    
    return True

# === TRADE EXECUTION ===
def place_order(order_type):
    """Place a market order with stop loss and take profit"""
    try:
        # Ensure symbol is selected and get current prices
        if not mt5.symbol_select(symbol, True):
            logging.error(f"Failed to select symbol {symbol}")
            return None
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logging.error(f"Failed to get tick data for {symbol}")
            return None
        
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logging.error(f"Failed to get symbol info for {symbol}")
            return None
        
        # Calculate prices
        if order_type == "buy":
            price = tick.ask
            sl = price - sl_distance
            tp = price + tp_distance
            order_type_mt5 = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            sl = price + sl_distance
            tp = price - tp_distance
            order_type_mt5 = mt5.ORDER_TYPE_SELL
        
        # Round prices to symbol's digits
        price = round(price, symbol_info.digits)
        sl = round(sl, symbol_info.digits)
        tp = round(tp, symbol_info.digits)
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type_mt5,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 123456,
            "comment": "RSI Trend Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(f"Order placed successfully: {order_type} at {price}")
            log_trade_to_csv(datetime.now(), symbol, order_type, price, sl, tp, result.retcode)
            return result
        else:
            logging.error(f"Order failed with code: {result.retcode}")
            log_trade_to_csv(datetime.now(), symbol, order_type, price, sl, tp, result.retcode)
            return None
            
    except Exception as e:
        logging.error(f"Error in place_order: {e}")
        return None

# === LOGGING ===
def log_trade_to_csv(timestamp, symbol, action, entry_price, sl, tp, result_code):
    """Log trade details to CSV file"""
    filename = "trades_log.csv"
    file_exists = os.path.isfile(filename)
    
    try:
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Timestamp", "Symbol", "Action", "EntryPrice", "StopLoss", "TakeProfit", "ResultCode"])
            writer.writerow([timestamp, symbol, action, entry_price, sl, tp, result_code])
    except Exception as e:
        logging.error(f"Error writing to CSV: {e}")

# === MAIN BOT LOGIC ===
def run_bot():
    """Main bot execution logic"""
    try:
        signal = get_rsi_signal()
        logging.info(f"Generated signal: {signal.upper()}")
        
        current_positions = check_existing_positions()
        logging.info(f"Current positions: {current_positions}")
        
        # Only trade if no existing positions
        if current_positions == 0 and signal in ["buy", "sell"]:
            result = place_order(signal)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"Trade executed successfully: {signal.upper()}")
            else:
                logging.error("Trade execution failed")
        else:
            if current_positions > 0:
                logging.info("Position already open - no new trades")
            else:
                logging.info("Signal is HOLD - no trade executed")
                
    except Exception as e:
        logging.error(f"Error in run_bot: {e}")

def main():
    """Main function to run the trading bot"""
    # Initialize MT5
    if not initialize_mt5():
        logging.error("Failed to initialize MT5. Exiting.")
        return
    
    logging.info(f"Starting RSI Trend Bot for {symbol}")
    logging.info(f"RSI Period: {rsi_period}, Oversold: {rsi_oversold}, Overbought: {rsi_overbought}")
    logging.info(f"Risk: {risk_pips} pips, Lot size: {lot}")
    logging.info("Bot running... Press Ctrl+C to stop")
    
    try:
        while True:
            run_bot()
            logging.info(f"Sleeping for {check_interval} seconds...")
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        logging.info("Bot stopped by user.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        mt5.shutdown()
        logging.info("MT5 shutdown complete.")

if __name__ == "__main__":
    main()