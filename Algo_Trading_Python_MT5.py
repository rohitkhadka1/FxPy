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


mt5.initialize(login = login, password = password, server = servidor, path = path)
print("Login Successful")

rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M5, 0 , 864)
rates_frame = pd.DataFrame(rates)

print(rates)
print(type(rates))


rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit = 's')

max_value = rates_frame['high'].max() 
min_value = rates_frame['low'].min()

request = {
    "action": mt5.TRADE_ACTION_PENDING,
    "symbol": "EURUSD",
    "type": mt5.ORDER_TYPE_BUY_LIMIT,
    "price": min_value,
    "volume": 1.0,
    "comment":"Algoooooo",
    "type_filling": mt5.ORDER_FILLING_IOC
}

mt5.order_send(request)

request = {
        "action":mt5.TRADE_ACTION_PENDING,
        "symbol":'EURUSD',
        "type" : mt5.ORDER_TYPE_SELL_LIMIT,
        "price": max_value,
        "volume":1.0,
        "comment":'Test_code',
        "type_filling":mt5.ORDER_FILLING_IOC
}

mt5.order_send(request)