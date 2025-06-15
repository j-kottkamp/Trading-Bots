import yfinance as yf
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.style as style
import datetime
import talib as ta
import pandas as pd


def StockData(start_year, end_year):
    global STARTING_BALANCE
    global YEARS
    
    SYMBOL = "SPY"
    STARTING_BALANCE = 10000

    START = datetime.datetime(start_year, 1, 1)
    END = datetime.datetime(end_year, 1, 1)
    YEARS = (END - START).days / 365.25

    price = yf.download(SYMBOL, start=START, end=END)
    price = price.drop(['Volume', 'Adj Close'], axis=1)
    
    return price

    
def CalculateHT(price):
    price['DC_Period'] = ta.HT_DCPERIOD(price.Close)
    price['DC_Phase'] = ta.HT_DCPHASE(price.Close)
    price['Inphase'], price['Quadrature'] = ta.HT_PHASOR(price.Close)
    price['Sine'], price['Leadsine'] = ta.HT_SINE(price.Close)
    price['Trendmode'] = ta.HT_TRENDMODE(price.Close)
    return price['DC_Period'], price['DC_Phase'], price['Inphase'], price['Quadrature'], price['Sine'], price['Leadsine'], price['Trendmode']


def SetStrategy(price):
    price['Trend'] = np.where(price['Trendmode'], 'Trend', 'Zyklus')
    
    V_conditions = [
        price['DC_Period'] < 20,
        price['DC_Period'].between(20, 30),
        price['DC_Period'] > 30,
    ]
    V_choices = [
        'High',
        'Medium',
        'Low',
    ]
    price['Volatility'] = np.select(V_conditions, V_choices, default='Unknown')
    
    price['Sinewave'] = np.where(price.Sine > price.Leadsine, 'Long', 'Short')
    
    P_conditions = [
        price['DC_Phase'].between(0, 20),
        price['DC_Phase'].between(20, 170),
        price['DC_Phase'].between(170, 200),
    ]
    P_choices = [
        'Long Entry',
        'Long Hold',
        'Long Exit',
    ]
    price['Phase'] = np.select(P_conditions, P_choices, default='Unknown')
    
    df = pd.DataFrame({'Phase': price['Phase'], 'Volatility': price['Volatility']})
    print(df.tail(50))
    
    ZyklusTrend = price.Trend == "Zyklus"
    long_entry = price.loc[ZyklusTrend, 'Phase'] == 'Long Entry'
    long_exit = price.loc[ZyklusTrend, 'Phase'] == 'Long Exit'
    
    price['Long'] = np.where(price.Phase.isin(['Long Hold', 'Long Entry', 'Long Exit']), True, False)    
def PlotHT(price):
    fig, axs = plt.subplots(7, figsize=(12, 20))
    indicators = ['DC_Period', 'DC_Phase', 'Inphase', 'Quadrature', 'Sine', 'Trendmode', 'Close']
    titles = ['Dominant Cycle Period', 'Dominant Cycle Phase', 'In-Phase', 'Quadrature', 'Sine', 'Trendmode', 'Close']
    
    for i, indicator in enumerate(indicators):
        axs[i].plot(price.index, price[indicator])
        axs[i].set_title(titles[i])
        axs[i].set_xlabel('Date')
        axs[i].set_ylabel(indicator)
    
    plt.tight_layout()
    plt.show()
  
  
def PlotReturn(price):
    style.use('dark_background')
    plt.figure(figsize=(12, 6))
    plt.plot(price.Bench_Bal, label='Benchmark')
    plt.plot(price.Sys_Bal, label='System', color="g")
    plt.legend()
    plt.show()
    
    
def main():
    price = StockData(2000, 2025)
    
    CalculateHT(price)
    
    SetStrategy(price)
    
    CalculateBench(price)
    
    PlotReturn(price)
    
    PlotHT(price)


if __name__ == '__main__':
    main()
