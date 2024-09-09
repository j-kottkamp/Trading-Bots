import yfinance as yf
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.style as style
import datetime
import talib as ta
import pandas as pd
import alpaca_trade_api as tradeapi


alpaca_api_key = 'PK3Q3CGZCG982APIX526'
alpaca_secret_key = 'hMPmAremeAEHGEE6xb2QJZ3rr2eeYFJBTzG470uG'
alpaca_base_url = 'https://paper-api.alpaca.markets'  # Für Testzwecke Paper Trading API
alpaca_api = tradeapi.REST(alpaca_api_key, alpaca_secret_key, alpaca_base_url, api_version='v2')

def is_position_open(symbol):
    try:
        position = alpaca_api.get_position(symbol)
        print(f"Position for {symbol} is open with {position.qty} shares.")
        return True
    except Exception as e:
        # Falls keine Position offen ist, wird ein Fehler ausgelöst
        print(f"No open position for {symbol}: {e}")
        return False
    
def place_stock_order(symbol, qty, side, type='market', time_in_force='gtc'):
    try:
        order = alpaca_api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=type,
            time_in_force=time_in_force
            
        )
        print(f">>>Stock order {side} {qty} of {symbol} placed successfully.")
    except Exception as e:
        print(f"Error placing stock order: {e}")
        
    
def StockDataNDAQ(start_year, end_year):
    SYMBOL = "NDAQ"

    START = datetime.datetime(start_year, 1, 1)
    END = datetime.datetime(end_year, 1, 1)
    YEARS = (END - START).days / 365.25

    priceNDAQ = yf.download(SYMBOL, start=START, end=END)
    priceNDAQ = priceNDAQ.drop(['Volume', 'Adj Close'], axis=1)
    
    return priceNDAQ

def StockDataSPY(start_year, end_year):
    SYMBOL = "SPY"

    START = datetime.datetime(start_year, 1, 1)
    END = datetime.datetime(end_year, 1, 1)
    YEARS = (END - START).days / 365.25

    priceSPY = yf.download(SYMBOL, start=START, end=END)
    priceSPY = priceSPY.drop(['Volume', 'Adj Close'], axis=1)
    
    return priceSPY


def CalculateIchiMoku(priceNDAQ):
    # Calculate Tenkan-sen (Conversion Line)
    priceNDAQ['Fast_IM'] = (priceNDAQ['High'].rolling(window=9).max() + priceNDAQ['Low'].rolling(window=9).min()) / 2
    
    # Calculate Kijun-sen (Base Line)
    priceNDAQ['Slow_IM'] = (priceNDAQ['High'].rolling(window=26).max() + priceNDAQ['Low'].rolling(window=26).min()) / 2
    
    # Calculate Senkou Span A (bullish when upper)
    priceNDAQ['Bullish_Cloud'] = ((priceNDAQ['Fast_IM'] + priceNDAQ['Slow_IM']) / 2).shift(26)
    
    # Calculate Senkou Span B (bearish when upper)
    priceNDAQ['Bearish_Cloud'] = ((priceNDAQ['High'].rolling(window=52).max() + priceNDAQ['Low'].rolling(window=52).min()) / 2).shift(26)
    
    # Calculate Chikou Span (Lagging Span)
    priceNDAQ['Lagging_Close'] = priceNDAQ['Close'].shift(-26)
    
    # Get the high and low of the lagging priceNDAQs last 5
    priceNDAQ['Lagging_High'] = priceNDAQ['Lagging_Close'].rolling(window=5).max()
    priceNDAQ['Lagging_Low'] = priceNDAQ['Lagging_Close'].rolling(window=5).min()

def CalculateBTD(priceSPY):
    #daily range
    priceSPY['Range'] = priceSPY.High - priceSPY.Low
    #distance between close and daily low
    priceSPY['Dist'] = abs(priceSPY.Close - priceSPY.Low)
    #% distance between close and low
    priceSPY['Pct'] = (priceSPY.Dist / priceSPY.Range) * 100


def SetStrategyIMC(priceNDAQ):
    priceNDAQ["Long"] = np.where(
        (priceNDAQ.Fast_IM > priceNDAQ.Slow_IM) & 
        (priceNDAQ.Lagging_Low > priceNDAQ.Close), True, False
    )
    
    priceNDAQ["Short"] = np.where(
        (priceNDAQ.Fast_IM < priceNDAQ.Slow_IM) & 
        (priceNDAQ.Lagging_High < priceNDAQ.Close), True, False
    )

def SetStrategyBTD(priceSPY):
    PCT_THRESH = 20.7937775384066
    RANGE_THRESH = 0.9563698594615758

    priceSPY['Long'] = np.logical_and((priceSPY.Pct < PCT_THRESH), (priceSPY.Range > RANGE_THRESH))


def  Execute_IMC(priceNDAQ):
    if priceNDAQ['Long'].iloc[-1] == True and not is_position_open("NDAQ"):
        print(f"Ichimoku Cloud found Signal\nPosition: Long")
        place_stock_order("NDAQ", '7', 'buy', type='market', time_in_force='gtc')
    elif priceNDAQ['Long'].iloc[-1] == True and is_position_open("NDAQ"):
        print(f"Ichimoku Cloud staying in Trade\nPosition: Long")
    elif priceNDAQ['Long'].iloc[-1] == False and not is_position_open("NDAQ"):
        print(f"Ichimoku Cloud found no Long Signal\nNo open Positions")
    elif priceNDAQ['Long'].iloc[-1] == False and is_position_open("NDAQ"):
        print(f"Ichimoku Cloud found no Signal\nClosing all Long Positions")
        place_stock_order("NDAQ", '7', 'sell', type='market', time_in_force='gtc')
        
        
    if priceNDAQ['Short'].iloc[-1] == True and not is_position_open("NDAQ"):
        print(f"Ichimoku Cloud found Signal\nPosition: Short")
        place_stock_order("NDAQ", '7', 'sell', type='market', time_in_force='gtc')
    elif priceNDAQ['Short'].iloc[-1] == True and is_position_open("NDAQ"):
        print(f"Ichimoku Cloud staying in Trade\nPosition: Short")
    elif priceNDAQ['Short'].iloc[-1] == False and not is_position_open("NDAQ"):
        print(f"Ichimoku Cloud found no Short Signal\nNo open Positions")
    elif priceNDAQ['Short'].iloc[-1] == False and is_position_open("NDAQ"):
        print(f"Ichimoku Cloud found no Signal\nClosing all Short Positions")
        place_stock_order("NDAQ", '7', 'buy', type='market', time_in_force='gtc')
        
def Execute_BTD(priceSPY):
    if priceSPY['Long'].iloc[-1] == True and not is_position_open("SPY"):
        print(f"System located Dip\nPosition: Long")
        place_stock_order("SPY", '1', 'buy', type='market', time_in_force='gtc')
    elif priceSPY['Long'].iloc[-1] == False and is_position_open("SPY"):
        print(f"No Dip located\nClosing all Positions")
        place_stock_order("SPY", '1', 'sell', type='market', time_in_force='gtc')
    elif priceSPY['Long'].iloc[-1] == True and is_position_open("SPY"):
        print(f"Dip located\nStaying in Trade")
    elif priceSPY['Long'].iloc[-1] == False and not is_position_open("SPY"):
        print(f"No Dip located\nNo open Positions")
        

def main():
    priceNDAQ = StockDataNDAQ(2000, 2099)
    CalculateIchiMoku(priceNDAQ)
    SetStrategyIMC(priceNDAQ)
    Execute_IMC(priceNDAQ)
    
    priceSPY = StockDataSPY(2000, 2099)
    CalculateBTD(priceSPY)
    SetStrategyBTD(priceSPY)
    Execute_BTD(priceSPY)
    
    


if __name__ == '__main__':
    main()
