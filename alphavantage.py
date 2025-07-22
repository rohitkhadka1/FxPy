from alpha_vantage.foreignexchange import ForeignExchange
from pprint import pprint
import os
from dotenv import load_dotenv
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
cc = ForeignExchange(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
# There is no metadata in this call
data, _ = cc.get_currency_exchange_rate(from_currency='NPR',to_currency='USD')
pprint(data)