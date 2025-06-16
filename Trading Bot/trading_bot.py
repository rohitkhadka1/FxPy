import MetaTrader5 as mt5
import time
from mt5_trade_functions import market_order, close_all_positions

if __name__ == '__main__':

    # open your own trading account: https://icmarkets.com/?camp=60457
    login = 51465418
    password = 'JPFfsLBB'
    server = 'ICMarketsEU-Demo'

    mt5.initialize()
    mt5.login(login, password, server)

    """
    Trading bot logic
    
    1) Opens a Trade on EURUSD 1 lot
    2) Strategy waits 5 seconds and then closes the position
    """

    symbol = 'EURUSD'
    volume = 1.0
    order_type = 'buy'  # values 'buy' or 'sell'

    market_order(symbol, volume, order_type)
    time.sleep(5)
    close_all_positions('all')  # accepts values 'buy', 'sell' or 'all'

