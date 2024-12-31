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
    
    return price



    
def CalculateIndicators(price):
    price['Rsi'] = ta.RSI(price['Adj Close'].values.flatten(), timeperiod = 14)

def SetStrategy(price):
    price["Long"] = np.where(price.Rsi < 70, True, False)
    
def CalculateReturn(price):
    # Benchmark Performance
    price['Return'] = price['Adj Close'] / price['Adj Close'].shift(1)
    price.Return.iat[0] = 1
    price['Bench_Bal'] = STARTING_BALANCE * price.Return.cumprod()
    price['Bench_Peak'] = price.Bench_Bal.cummax()
    price['Bench_DD'] = price.Bench_Bal - price.Bench_Peak
    
    bench_dd = round(((price.Bench_DD / price.Bench_Peak).min() * 100), 2)
    bench_return = round(((price.Bench_Bal.iloc[-1]/price.Bench_Bal.iloc[0]) - 1) * 100, 2)
    bench_cagr = round(((((price.Bench_Bal.iloc[-1]/price.Bench_Bal.iloc[0])**(1/YEARS))-1)*100), 2)
    
    # System Performance
    price["Sys_Return"] = np.where(price.Long.shift(1) == True, price.Return, 1)
    price['Sys_Bal'] = (STARTING_BALANCE * price.Sys_Return.cumprod())
    price['Sys_Peak'] = price.Sys_Bal.cummax()
    price['Sys_DD'] = price.Sys_Bal - price.Sys_Peak
    
    sys_return = round(((price.Sys_Bal.iloc[-1]/price.Sys_Bal.iloc[0]) - 1) * 100, 2)
    sys_dd = round(((price.Sys_DD / price.Sys_Peak).min()) * 100, 2)
    sys_cagr = round(((((price.Sys_Bal.iloc[-1]/price.Sys_Bal.iloc[0])**(1/YEARS))-1)*100), 2)
    sys_in_market = round((price.Long.value_counts().loc[True] / len(price)) * 100)
    sys_win = price.Sys_Return[price.Sys_Return > 1.0].count()
    sys_loss = price.Sys_Return[price.Sys_Return < 1.0].count()
    sys_winrate = round(sys_win / (sys_win + sys_loss) * 100, 2)
    
    # Sharpe Ratios
    daily_bench_ret = price['Return'].dropna() - 1
    bench_return_mean = daily_bench_ret.mean()
    bench_return_std = daily_bench_ret.std()
    sharpe_ratio_bench = (bench_return_mean / bench_return_std) * np.sqrt(252)
    
    daily_sys_ret = price['Sys_Return'].dropna() - 1
    sys_return_mean = daily_sys_ret.mean()
    sys_return_std = daily_sys_ret.std()
    sharpe_ratio_sys = (sys_return_mean / sys_return_std) * np.sqrt(252)


    
    print(f'Benchmark Total return: {bench_return}%')
    print(f'Benchmark CAGR: {bench_cagr}')
    print(f'Benchmark DD: {bench_dd}%')
    print(f'Benchmark Sharpe Ratio: {sharpe_ratio_bench:.2f}')
    print('')
    print(f'System Total return: {sys_return}%')
    print(f'System CAGR: {sys_cagr}')
    print(f'System DD: {sys_dd}%')
    print(f'System Sharpe Ratio: {sharpe_ratio_sys:.2f}')
    print(f'Time in Market: {sys_in_market}%')
    print(f'Trades Won: {sys_win}')
    print(f'Trades Loss: {sys_loss}')
    print(f'Winrate: {sys_winrate}%\n')
        
    
    
    
  
def PlotReturn(price):
    style.use('dark_background')
    plt.figure(figsize=(12, 6))
    plt.plot(price.Bench_Bal, label='Benchmark')
    plt.plot(price.Sys_Bal, label='System', color="g")
    plt.legend()
    plt.show()
    
    
def main():
    price = StockData(2000, 2025)
    
    CalculateIndicators(price)
    
    SetStrategy(price)
    
    CalculateReturn(price)
    
    PlotReturn(price)
    


if __name__ == '__main__':
    main()
