import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime
import os
import csv

# === CONFIGURATION ===
symbol = "BTCUSDm"  # Use "BTCUSDm" for MetaTrader 5
timeframe = mt5.TIMEFRAME_M5
lot = 0.5
rsi_period = 14
rsi_oversold = 30
rsi_overbought = 70
risk_pips = 20
pip_value = 0.0001 if "JPY" not in symbol else 0.01
sl_distance = risk_pips * pip_value
tp_distance = sl_distance * 2


# === INITIALIZE MT5 ===
if not mt5.initialize():
    print("Initialization failed:", mt5.last_error())
    quit()

# Fixed: Use the configured symbol instead of hardcoded "BTCUSD"
if not mt5.symbol_select(symbol, True):
    print(f"Failed to select {symbol}:", mt5.last_error())
    # Try to get available symbols for debugging
    symbols = mt5.symbols_get()
    if symbols:
        print(f"Available symbols (first 10): {[s.name for s in symbols[:10]]}")
    quit()

# === SIGNAL GENERATION ===
def get_rsi_signal():
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, rsi_period + 2)
    if rates is None:
        print("Failed to get rates:", mt5.last_error())
        return "hold"
    elif len(rates) < rsi_period + 2:   # Ensure we have enough data
        print("Not enough data")
        return "hold"
    else:
        print(f"Rates retrieved: {len(rates)} entries")
        print(f"Last 5 rates: {rates[-5:]}")

    df = pd.DataFrame(rates)
    df['close'] = df['close']

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()

    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    prev_rsi = df['rsi'].iloc[-2]
    curr_rsi = df['rsi'].iloc[-1]

    print(f"Previous RSI: {prev_rsi:.2f}, Current RSI: {curr_rsi:.2f}")

    # if prev_rsi < rsi_oversold and curr_rsi > rsi_oversold:
    if curr_rsi:
        return "buy"
    # elif prev_rsi > rsi_overbought and curr_rsi < rsi_overbought:
    #     return "sell"
    else:
        return "hold"


# === TRADE EXECUTION ===
def place_order(order_type):
    if not mt5.symbol_info(symbol).visible:
        mt5.symbol_select(symbol, True)

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"Failed to get tick data for {symbol}")
        return None
        
    price = tick.ask if order_type == "buy" else tick.bid

    sl = price - sl_distance if order_type == "buy" else price + sl_distance
    tp = price + tp_distance if order_type == "buy" else price - tp_distance

    order_type_mt5 = mt5.ORDER_TYPE_BUY if order_type == "buy" else mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type_mt5,
        "price": price,
        "sl": round(sl, 5),
        "tp": round(tp, 5),
        "deviation": 10,
        "magic": 123456,
        "comment": "RSI Trend Bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    log_trade_to_csv(datetime.now(), symbol, order_type, price, sl, tp, result.retcode if result else "Failed")
    return result


# === LOGGING ===
def log_trade_to_csv(timestamp, symbol, action, entry_price, sl, tp, result_code):
    filename = "trades_log.csv"
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Symbol", "Action", "EntryPrice", "StopLoss", "TakeProfit", "ResultCode"])
        writer.writerow([timestamp, symbol, action, entry_price, sl, tp, result_code])


# === MAIN BOT LOGIC ===
def run_bot():
    signal = get_rsi_signal()
    print(f"{datetime.now()} - Signal: {signal.upper()}")

    # Only one trade at a time
    if mt5.positions_total() == 0 and signal in ["buy", "sell"]:
        result = place_order(signal)
        if result:
            print(f"Trade executed: {result}")
        else:
            print("Trade execution failed")
    else:
        print("No trade executed (signal = hold or position already open).")


# === LOOP EVERY 5 MINUTES ===
try:
    print(f"Starting RSI Trend Bot for {symbol}")
    print(f"RSI Period: {rsi_period}, Oversold: {rsi_oversold}, Overbought: {rsi_overbought}")
    print(f"Risk: {risk_pips} pips, Lot size: {lot}")
    print("Bot running... Press Ctrl+C to stop")
    
    while True:
        run_bot()
        time.sleep(10)  # 5 minutes (300 seconds)
except KeyboardInterrupt:
    print("Bot stopped by user.")
finally:
    mt5.shutdown()
    print("MT5 shutdown complete.")