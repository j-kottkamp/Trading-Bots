import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
import time


alpaca_api_key = 'PK3Q3CGZCG982APIX526'
alpaca_secret_key = 'hMPmAremeAEHGEE6xb2QJZ3rr2eeYFJBTzG470uG'
alpaca_base_url = 'https://paper-api.alpaca.markets'  # Für Testzwecke Paper Trading API
alpaca_api = tradeapi.REST(alpaca_api_key, alpaca_secret_key, alpaca_base_url, api_version='v2')

SYMBOL = "NDAQ"
STARTING_BALANCE = 10000
PCT_THRESH = 20.7937775384066
RANGE_THRESH = 0.9563698594615758

#define starting variables
START = datetime.now() - timedelta(days=3*365.25)
END = datetime.now()
YEARS = (END - START).days/ 365.25

price = yf.download(SYMBOL, start=START, end=END)
price = price.drop(['Adj Close'], axis=1)

#daily range
price['Range'] = price.High - price.Low
#distance between close and daily low
price['Dist'] = abs(price.Close - price.Low)
#% distance between close and low
price['Pct'] = (price.Dist / price.Range) * 100

#identify entries and allocate trading fees
price['Long'] = np.logical_and((price.Pct < PCT_THRESH), (price.Range > RANGE_THRESH))

position_open = False

def place_stock_order(symbol, qty, side, type='market', time_in_force='gtc'):
    try:
        order = alpaca_api.submit_order(
            symbol=SYMBOL,
            qty=qty,
            side=side,
            type=type,
            time_in_force=time_in_force
        )
        print(f">>>Stock order {side} {qty} of {symbol} placed successfully.")
    except Exception as e:
        print(f"Error placing stock order: {e}")

if price['Long'].iloc[-1] == True and not position_open:
    print(f"System located Dip\nPosition: Long")
    place_stock_order(SYMBOL, '1', 'buy', type='market', time_in_force='gtc')
    position_open = True
elif price['Long'].iloc[-1] == False and position_open:
    print(f"No Dip located\nClosing all Positions")
    place_stock_order(SYMBOL, '1', 'sell', type='market', time_in_force='gtc')
    position_open = False
elif price['Long'].iloc[-1] == True and position_open:
    print(f"Dip located\nStaying in Trade")
elif price['Long'].iloc[-1] == False and not position_open:
    print(f"No Dip located\nNo open Positions")

price.tail(12)
