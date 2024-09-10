import yfinance as yf
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.style as style
import datetime
import talib as ta
import pandas as pd
import optuna


def StockData(start_year, end_year):
    global STARTING_BALANCE
    global YEARS
    
    SYMBOL = "QQQ"
    STARTING_BALANCE = 1000

    START = datetime.datetime(start_year, 1, 1)
    END = datetime.datetime(end_year, 1, 1)
    YEARS = (END - START).days / 365.25

    price = yf.download(SYMBOL, start=START, end=END)
    price = price.drop(['Volume', 'Adj Close'], axis=1)
    
    return price

    

    


def SetStrategy1(price, WINDOW_LAGGING, WINDOW_FAST, WINDOW_SLOW):
    
    # Calculate Tenkan-sen (Conversion Line)
    price['Fast_IM'] = (price['High'].rolling(window=WINDOW_FAST).max() + price['Low'].rolling(window=WINDOW_FAST).min()) / 2
    
    # Calculate Kijun-sen (Base Line)
    price['Slow_IM'] = (price['High'].rolling(window=WINDOW_SLOW).max() + price['Low'].rolling(window=WINDOW_SLOW).min()) / 2
    
    # Calculate Senkou Span A (bullish when upper)
    price['Bullish_Cloud'] = ((price['Fast_IM'] + price['Slow_IM']) / 2).shift(26)
    
    # Calculate Senkou Span B (bearish when upper)
    price['Bearish_Cloud'] = ((price['High'].rolling(window=52).max() + price['Low'].rolling(window=52).min()) / 2).shift(26)
    
    # Calculate Chikou Span (Lagging Span)
    price['Lagging_Close'] = price['Close'].shift(-26)
    
    # Get the high and low of the lagging prices last 5
    price['Lagging_High'] = price['Lagging_Close'].rolling(window=WINDOW_LAGGING).max()
    price['Lagging_Low'] = price['Lagging_Close'].rolling(window=WINDOW_LAGGING).min()
    
    
    # here strategy------->
    price["Long"] = np.where(
        (price.Fast_IM > price.Slow_IM) & 
        (price.Lagging_Low.shift(-26) > price.Close) & 
        (price.Close > price.Bearish_Cloud), True, False
    )
    
    price["Short"] = np.where(
        (price.Fast_IM < price.Slow_IM) & 
        (price.Lagging_High.shift(-26) < price.Close) &
        (price.Close < price.Bullish_Cloud), True, False
    )
    # <-------- copy from backtesting 
    
    
    # Benchmark Performance
    price['Return'] = price.Close / price.Close.shift(1)
    price.Return.iat[0] = 1
    price['Bench_Bal'] = STARTING_BALANCE * price.Return.cumprod()
    price['Bench_Peak'] = price.Bench_Bal.cummax()
    price['Bench_DD'] = price.Bench_Bal - price.Bench_Peak
    
    bench_dd = round(((price.Bench_DD / price.Bench_Peak).min() * 100), 2)
    bench_return = round(((price.Bench_Bal.iloc[-1]/price.Bench_Bal.iloc[0]) - 1) * 100, 2)
    bench_cagr = round(((((price.Bench_Bal.iloc[-1]/price.Bench_Bal.iloc[0])**(1/YEARS))-1)*100), 2)
    
    # System Performance
    price["Sys_Return"] = np.where(price.Long.shift(1) == True, price.Return, 
                          np.where(price.Short.shift(1) == True, 1 / price.Return, 1))

    price['Sys_Bal'] = (STARTING_BALANCE * price.Sys_Return.cumprod())
    price['Sys_Peak'] = price.Sys_Bal.cummax()
    price['Sys_DD'] = price.Sys_Bal - price.Sys_Peak
    
    sys_return = round(((price.Sys_Bal.iloc[-1]/price.Sys_Bal.iloc[0]) - 1) * 100, 2)
    sys_dd = round(((price.Sys_DD / price.Sys_Peak).min()) * 100, 2)
    sys_cagr = round(((((price.Sys_Bal.iloc[-1]/price.Sys_Bal.iloc[0])**(1/YEARS))-1)*100), 2)
    sys_win = price.Sys_Return[price.Sys_Return > 1.0].count()
    sys_loss = price.Sys_Return[price.Sys_Return < 1.0].count()
    sys_winrate = round(sys_win / (sys_win + sys_loss) * 100, 2)
    
    # Calculate average return of winning trades
    winning_trades = price.Sys_Return[price.Sys_Return > 1.0]
    avg_win_return = (winning_trades - 1).mean() * 100

    # Calculate average return of losing trades
    losing_trades = price.Sys_Return[price.Sys_Return < 1.0]
    avg_loss_return = (losing_trades - 1).mean() * 100

    # Calculate average return of all trades
    avg_trade_return = (price.Sys_Return - 1).mean() * 100
    # Sharpe Ratios
    daily_bench_ret = price['Return'].dropna() - 1
    bench_return_mean = daily_bench_ret.mean()
    bench_return_std = daily_bench_ret.std()
    sharpe_ratio_bench = (bench_return_mean / bench_return_std) * np.sqrt(252)
    
    daily_sys_ret = price['Sys_Return'].dropna() - 1
    sys_return_mean = daily_sys_ret.mean()
    sys_return_std = daily_sys_ret.std()
    sharpe_ratio_sys = (sys_return_mean / sys_return_std) * np.sqrt(252)

    return sharpe_ratio_sys
    

  
    
def bayesianOpt(price):
    def objective(trial):
        WINDOW_LAGGING = trial.suggest_int('WINDOW_LAGGING', 1, 100)
        WINDOW_FAST = trial.suggest_int('WINDOW_FAST', 1, 100)
        WINDOW_SLOW = trial.suggest_int('WINDOW_SLOW', 1, 100)
        

        # Run the backtesting function with the suggested hyperparameters
        return SetStrategy1(price, WINDOW_LAGGING, WINDOW_FAST, WINDOW_SLOW)

    study = optuna.create_study(direction='maximize')  # Optimize to maximize the value
    study.optimize(objective, n_trials=500)

    print('Best trial:')
    trial = study.best_trial
    print('  WINDOW_LAGGING:', trial.params['WINDOW_LAGGING'])
    print('  WINDOW_FAST:', trial.params['WINDOW_FAST'])
    print('  WINDOW_SLOW:', trial.params['WINDOW_SLOW']) 
    print('  sharpe_ratio_sys:', trial.value)


def main():
    price = StockData(2000, 2025) 
    bayesianOpt(price)
    
    
   
if __name__ == '__main__':
    main()