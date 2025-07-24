import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta
import time
import logging
from dotenv import load_dotenv
import os
load_dotenv()
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MT5ChartPatternTrader:
    """
    MetaTrader 5 Chart Pattern Detection and Automated Trading System
    """
    
    def __init__(self, account: int, password: str, server: str, 
                 lookback_period: int = 50, min_pattern_length: int = 10):
        self.account = account
        self.password = password
        self.server = server
        self.lookback_period = lookback_period
        self.min_pattern_length = min_pattern_length
        self.is_connected = False
        self.active_positions = {}
        
        # Risk management settings
        self.max_risk_per_trade = 0.02  # 2% per trade
        self.max_daily_trades = 10
        self.daily_trade_count = 0
        self.last_trade_date = datetime.now().date()
        
        # Connect to MT5
        self.connect_mt5()
    
    def connect_mt5(self) -> bool:
        """Connect to MetaTrader 5"""
        try:
            if not mt5.initialize():
                logger.error("MetaTrader 5 initialization failed")
                return False
            
            # Login to account
            if not mt5.login(account = self.account, password=self.password, server=self.server):
                logger.error(f"Failed to login to account {self.account}")
                return False
            
            self.is_connected = True
            logger.info(f"Successfully connected to MT5 account {self.account}")
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        if not self.is_connected:
            return {}
        
        account_info = mt5.account_info()
        if account_info is None:
            return {}
        
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free,
            'profit': account_info.profit,
            'currency': account_info.currency
        }
    
    def get_symbol_data(self, symbol: str, timeframe: int, count: int = 100) -> pd.DataFrame:
        """Get historical price data for a symbol"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None:
                logger.error(f"Failed to get data for {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting symbol data: {e}")
            return pd.DataFrame()
    
    def detect_trend(self, prices: np.array, window: int = 20) -> str:
        """Detect current trend direction"""
        if len(prices) < window:
            return "neutral"
        
        recent_prices = prices[-window:]
        slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        
        if slope > 0.0001:
            return "uptrend"
        elif slope < -0.0001:
            return "downtrend"
        else:
            return "neutral"
    
    def find_support_resistance(self, prices: np.array, window: int = 5) -> Tuple[np.array, np.array]:
        """Find support and resistance levels"""
        highs, _ = find_peaks(prices, distance=window)
        lows, _ = find_peaks(-prices, distance=window)
        
        if len(highs) == 0:
            resistance_levels = np.array([])
        else:
            resistance_levels = prices[highs]
        
        if len(lows) == 0:
            support_levels = np.array([])
        else:
            support_levels = prices[lows]
        
        return support_levels, resistance_levels
    
    def detect_ascending_triangle(self, highs: np.array, lows: np.array, prices: np.array) -> Dict:
        """Detect ascending triangle pattern"""
        if len(highs) < 3 or len(lows) < 3:
            return {"detected": False}
        
        # Check if highs are relatively flat (resistance)
        high_slope = np.polyfit(range(len(highs)), highs, 1)[0] if len(highs) > 1 else 0
        # Check if lows are ascending (support)
        low_slope = np.polyfit(range(len(lows)), lows, 1)[0] if len(lows) > 1 else 0
        
        if abs(high_slope) < 0.0001 and low_slope > 0.0001:
            current_trend = self.detect_trend(prices)
            resistance_level = np.max(highs)
            support_level = np.min(lows)
            
            return {
                "detected": True,
                "pattern": "ascending_triangle",
                "signal": "bullish_continuation" if current_trend == "uptrend" else "neutral",
                "resistance": resistance_level,
                "support": support_level,
                "entry_price": resistance_level * 1.001,
                "target": resistance_level * 1.05,
                "stop_loss": support_level * 0.98
            }
        return {"detected": False}
    
    def detect_descending_triangle(self, highs: np.array, lows: np.array, prices: np.array) -> Dict:
        """Detect descending triangle pattern"""
        if len(highs) < 3 or len(lows) < 3:
            return {"detected": False}
        
        low_slope = np.polyfit(range(len(lows)), lows, 1)[0] if len(lows) > 1 else 0
        high_slope = np.polyfit(range(len(highs)), highs, 1)[0] if len(highs) > 1 else 0
        
        if abs(low_slope) < 0.0001 and high_slope < -0.0001:
            current_trend = self.detect_trend(prices)
            resistance_level = np.max(highs)
            support_level = np.min(lows)
            
            return {
                "detected": True,
                "pattern": "descending_triangle",
                "signal": "bearish_continuation" if current_trend == "downtrend" else "neutral",
                "resistance": resistance_level,
                "support": support_level,
                "entry_price": support_level * 0.999,
                "target": support_level * 0.95,
                "stop_loss": resistance_level * 1.02
            }
        return {"detected": False}
    
    def detect_double_top(self, prices: np.array) -> Dict:
        """Detect double top pattern"""
        peaks, _ = find_peaks(prices, distance=5)
        if len(peaks) < 2:
            return {"detected": False}
        
        # Find two highest peaks
        peak_values = prices[peaks]
        if len(peak_values) < 2:
            return {"detected": False}
        
        sorted_indices = np.argsort(peak_values)[-2:]
        top_peaks = peaks[sorted_indices]
        
        if len(top_peaks) < 2:
            return {"detected": False}
        
        peak1, peak2 = sorted(top_peaks)
        peak1_val, peak2_val = prices[peak1], prices[peak2]
        
        # Check if peaks are similar in height (within 2%)
        if abs(peak1_val - peak2_val) / max(peak1_val, peak2_val) < 0.02:
            valley_idx = np.argmin(prices[peak1:peak2]) + peak1
            valley_val = prices[valley_idx]
            
            current_trend = self.detect_trend(prices)
            if current_trend == "uptrend":
                return {
                    "detected": True,
                    "pattern": "double_top",
                    "signal": "bearish_reversal",
                    "resistance": max(peak1_val, peak2_val),
                    "support": valley_val,
                    "entry_price": valley_val * 0.999,
                    "target": valley_val * 0.95,
                    "stop_loss": max(peak1_val, peak2_val) * 1.02
                }
        
        return {"detected": False}
    
    def detect_double_bottom(self, prices: np.array) -> Dict:
        """Detect double bottom pattern"""
        troughs, _ = find_peaks(-prices, distance=5)
        if len(troughs) < 2:
            return {"detected": False}
        
        trough_values = prices[troughs]
        if len(trough_values) < 2:
            return {"detected": False}
        
        sorted_indices = np.argsort(trough_values)[:2]
        bottom_troughs = troughs[sorted_indices]
        
        if len(bottom_troughs) < 2:
            return {"detected": False}
        
        trough1, trough2 = sorted(bottom_troughs)
        trough1_val, trough2_val = prices[trough1], prices[trough2]
        
        if abs(trough1_val - trough2_val) / max(trough1_val, trough2_val) < 0.02:
            peak_idx = np.argmax(prices[trough1:trough2]) + trough1
            peak_val = prices[peak_idx]
            
            current_trend = self.detect_trend(prices)
            if current_trend == "downtrend":
                return {
                    "detected": True,
                    "pattern": "double_bottom",
                    "signal": "bullish_reversal",
                    "resistance": peak_val,
                    "support": min(trough1_val, trough2_val),
                    "entry_price": peak_val * 1.001,
                    "target": peak_val * 1.05,
                    "stop_loss": min(trough1_val, trough2_val) * 0.98
                }
        
        return {"detected": False}
    
    def detect_head_and_shoulders(self, prices: np.array) -> Dict:
        """Detect head and shoulders pattern"""
        peaks, _ = find_peaks(prices, distance=3)
        if len(peaks) < 3:
            return {"detected": False}
        
        # Take the most recent 3 peaks
        if len(peaks) >= 3:
            recent_peaks = peaks[-3:]
            
            left_shoulder = recent_peaks[0]
            head = recent_peaks[1]
            right_shoulder = recent_peaks[2]
            
            ls_val = prices[left_shoulder]
            head_val = prices[head]
            rs_val = prices[right_shoulder]
            
            # Check pattern: head higher than both shoulders
            if (head_val > ls_val and head_val > rs_val and 
                abs(ls_val - rs_val) / max(ls_val, rs_val) < 0.05):
                
                # Find neckline
                left_valley_start = max(0, left_shoulder - 5)
                left_valley_end = min(len(prices), head)
                right_valley_start = head
                right_valley_end = min(len(prices), right_shoulder + 5)
                
                if left_valley_end > left_valley_start:
                    left_valley = np.argmin(prices[left_valley_start:left_valley_end]) + left_valley_start
                else:
                    left_valley = left_shoulder
                
                if right_valley_end > right_valley_start:
                    right_valley = np.argmin(prices[right_valley_start:right_valley_end]) + right_valley_start
                else:
                    right_valley = right_shoulder
                
                neckline = (prices[left_valley] + prices[right_valley]) / 2
                
                current_trend = self.detect_trend(prices)
                if current_trend == "uptrend":
                    return {
                        "detected": True,
                        "pattern": "head_and_shoulders",
                        "signal": "bearish_reversal",
                        "resistance": head_val,
                        "support": neckline,
                        "entry_price": neckline * 0.999,
                        "target": neckline * 0.92,
                        "stop_loss": head_val * 1.02
                    }
        
        return {"detected": False}
    
    def scan_patterns(self, symbol: str, timeframe: int) -> Dict:
        """Scan for all patterns on a given symbol"""
        df = self.get_symbol_data(symbol, timeframe, self.lookback_period)
        if df.empty:
            return {}
        
        prices = df['close'].values
        support_levels, resistance_levels = self.find_support_resistance(prices)
        
        patterns = {}
        
        # Check each pattern
        patterns["ascending_triangle"] = self.detect_ascending_triangle(
            resistance_levels, support_levels, prices)
        patterns["descending_triangle"] = self.detect_descending_triangle(
            resistance_levels, support_levels, prices)
        patterns["double_top"] = self.detect_double_top(prices)
        patterns["double_bottom"] = self.detect_double_bottom(prices)
        patterns["head_and_shoulders"] = self.detect_head_and_shoulders(prices)
        
        # Return only detected patterns
        detected = {k: v for k, v in patterns.items() if v.get("detected", False)}
        
        return detected
    
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk management"""
        account_info = self.get_account_info()
        if not account_info:
            return 0.1  # Default small size
        
        balance = account_info['balance']
        risk_amount = balance * self.max_risk_per_trade
        
        # Get symbol info for proper lot size calculation
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return 0.1
        
        # Calculate pip value and risk per pip
        pip_size = symbol_info.point * 10 if 'JPY' not in symbol else symbol_info.point * 100
        risk_pips = abs(entry_price - stop_loss) / pip_size
        
        if risk_pips > 0:
            pip_value = symbol_info.trade_tick_value
            lot_size = risk_amount / (risk_pips * pip_value * symbol_info.trade_contract_size)
            
            # Round to allowed lot size
            min_lot = symbol_info.volume_min
            max_lot = symbol_info.volume_max
            lot_step = symbol_info.volume_step
            
            lot_size = max(min_lot, min(max_lot, round(lot_size / lot_step) * lot_step))
            return lot_size
        
        return 0.1
    
    def place_order(self, symbol: str, order_type: str, volume: float, price: float, 
                   stop_loss: float, take_profit: float, comment: str = "") -> bool:
        """Place a trading order"""
        try:
            # Reset daily trade count if new day
            current_date = datetime.now().date()
            if current_date != self.last_trade_date:
                self.daily_trade_count = 0
                self.last_trade_date = current_date
            
            # Check daily trade limit
            if self.daily_trade_count >= self.max_daily_trades:
                logger.warning("Daily trade limit reached")
                return False
            
            # Prepare request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": 10,
                "magic": 12345,  # Unique identifier for our EA
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed: {result.retcode} - {result.comment}")
                return False
            
            self.daily_trade_count += 1
            logger.info(f"Order placed successfully: {order_type} {volume} lots of {symbol}")
            
            # Store position info
            self.active_positions[result.order] = {
                "symbol": symbol,
                "type": order_type,
                "volume": volume,
                "entry_price": price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "time": datetime.now(),
                "pattern": comment
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return False
    
    def execute_trading_signals(self, symbols: List[str], timeframe: int = mt5.TIMEFRAME_H1):
        """Execute trading based on detected patterns"""
        for symbol in symbols:
            try:
                # Get current price
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    continue
                
                current_price = (tick.bid + tick.ask) / 2
                
                # Scan for patterns
                patterns = self.scan_patterns(symbol, timeframe)
                
                for pattern_name, pattern_data in patterns.items():
                    signal = pattern_data['signal']
                    entry_price = pattern_data['entry_price']
                    target = pattern_data['target']
                    stop_loss = pattern_data['stop_loss']
                    
                    # Check if pattern conditions are met
                    if signal in ['bullish_continuation', 'bullish_reversal']:
                        if current_price >= entry_price:
                            volume = self.calculate_position_size(symbol, entry_price, stop_loss)
                            if volume > 0:
                                self.place_order(
                                    symbol=symbol,
                                    order_type="BUY",
                                    volume=volume,
                                    price=tick.ask,
                                    stop_loss=stop_loss,
                                    take_profit=target,
                                    comment=f"{pattern_name}_BUY"
                                )
                    
                    elif signal in ['bearish_continuation', 'bearish_reversal']:
                        if current_price <= entry_price:
                            volume = self.calculate_position_size(symbol, entry_price, stop_loss)
                            if volume > 0:
                                self.place_order(
                                    symbol=symbol,
                                    order_type="SELL",
                                    volume=volume,
                                    price=tick.bid,
                                    stop_loss=stop_loss,
                                    take_profit=target,
                                    comment=f"{pattern_name}_SELL"
                                )
                
                # Small delay between symbols
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
    
    def get_open_positions(self) -> pd.DataFrame:
        """Get all open positions"""
        positions = mt5.positions_get()
        if positions is None:
            return pd.DataFrame()
        
        df = pd.DataFrame(list(positions), columns=positions[0]._asdict().keys())
        return df
    
    def monitor_positions(self):
        """Monitor and manage open positions"""
        positions = self.get_open_positions()
        
        if positions.empty:
            return
        
        for _, position in positions.iterrows():
            # Add any position management logic here
            # e.g., trailing stops, partial closures, etc.
            pass
    
    def run_trading_bot(self, symbols: List[str], timeframe: int = mt5.TIMEFRAME_H1, 
                       scan_interval: int = 300):  # 5 minutes
        """Run the automated trading bot"""
        logger.info("Starting automated trading bot...")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Timeframe: {timeframe}")
        logger.info(f"Scan interval: {scan_interval} seconds")
        
        while True:
            try:
                if not self.is_connected:
                    logger.warning("Connection lost, attempting to reconnect...")
                    self.connect_mt5()
                    time.sleep(10)
                    continue
                
                logger.info("Scanning for patterns...")
                
                # Execute trading signals
                self.execute_trading_signals(symbols, timeframe)
                
                # Monitor existing positions
                self.monitor_positions()
                
                # Display account summary
                account_info = self.get_account_info()
                if account_info:
                    logger.info(f"Balance: {account_info['balance']:.2f}, "
                              f"Equity: {account_info['equity']:.2f}, "
                              f"Profit: {account_info['profit']:.2f}")
                
                # Wait for next scan
                time.sleep(scan_interval)
                
            except KeyboardInterrupt:
                logger.info("Trading bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def disconnect(self):
        """Disconnect from MT5"""
        mt5.shutdown()
        logger.info("Disconnected from MetaTrader 5")

# Example usage
def main():
    """Example of how to use the trading system"""
    
    # Your MT5 account credentials
    ACCOUNT = int(os.environ.get("MT5_ACCOUNT"))  # Your account number
    PASSWORD = os.environ.get("MT5_PASSWORD")  # Your password
    SERVER = os.environ.get("MT5_SERVER")  # Your broker's server
    
    # Symbols to trade
    SYMBOLS = ["EURUSDm", "GBPUSDm", "USDJPYm", "AUDUSDm"]
    
    # Initialize trader
    trader = MT5ChartPatternTrader(
        account=ACCOUNT,
        password=PASSWORD,
        server=SERVER,
        lookback_period=100
    )
    
    if trader.is_connected:
        try:
            # Run the trading bot
            trader.run_trading_bot(
                symbols=SYMBOLS,
                timeframe=mt5.TIMEFRAME_H1,  # 1-hour timeframe
                scan_interval=300  # Scan every 5 minutes
            )
        finally:
            trader.disconnect()
    else:
        print("Failed to connect to MetaTrader 5")

if __name__ == "__main__":
    main()