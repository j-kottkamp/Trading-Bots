import yfinance as yf
import numpy as np
from matplotlib import pyplot as plt
import datetime
import optuna

#This version can do bayesian optimization

#define starting variables
SYMBOL = "SPY"
STARTING_BALANCE = 10000
PCT_THRESH = 20.7937775384066
RANGE_THRESH =  0.9563698594615758


#define starting variables
START = datetime.datetime(2021, 1, 1)
END = datetime.datetime(2024, 9, 1)
YEARS = (END - START).days/ 365.25

def backtest(PCT_THRESH, RANGE_THRESH, TAKE_PROFIT_PCT, TRAILING_STOP_PCT):
    
    price = yf.download(SYMBOL, start=START, end=END)
    
    price = price.drop(['Volume', 'Adj Close'], axis=1)

    price['Return'] = price.Close / price.Close.shift(1)
    price.Return.iat[0] = 1
    price['Bench_Bal'] = STARTING_BALANCE * price.Return.cumprod()
    price['Bench_Peak'] = price.Bench_Bal.cummax()
    price['Bench_DD'] = price.Bench_Bal - price.Bench_Peak
    bench_dd = round(((price.Bench_DD / price.Bench_Peak).min() * 100), 2)

    price['Range'] = ((price.High - price.Low) / price.Low) * 100
    price['Dist'] = abs(price.Close - price.Low)
    price['Pct'] = (price.Dist / price.Range) * 100

    price['Max_Close'] = np.nan
    price['Trailing_Stop'] = np.nan
    price['Stop_Loss'] = np.nan
    price['Take_Profit'] = np.nan

    price['Long'] = np.logical_and((price.Pct < PCT_THRESH), (price.Range > RANGE_THRESH))

    price['Max_Close'] = np.where(price.Long, price['Close'].cummax(), np.nan)
    price['Max_Close'] = price['Max_Close'].fillna(method='ffill')
    price['Trailing_Stop'] = price['Max_Close'] * (1 - TRAILING_STOP_PCT / 100)

    price['Take_Profit'] = price.Close * (1 + TAKE_PROFIT_PCT / 100)

    price['Stop_Loss_Trigger'] = np.where((price.Low < price.Trailing_Stop.shift(1)) & (price.Long.shift(1) == True), 1, 0)
    price['Take_Profit_Trigger'] = np.where((price.High > price.Take_Profit.shift(1)) & (price.Long.shift(1) == True), 1, 0)

    price['Sys_Ret'] = np.where(
        (price.Long.shift(1) == True) & (price.Open < price.Trailing_Stop.shift(1)),
        # If the market opens below the Trailing Stop-Loss (overnight gap down), sell at the opening price
        (price.Open / price.Close.shift(1)),
        np.where(
            (price.Long.shift(1) == True) & (price.Low < price.Trailing_Stop.shift(1)),
            # If Trailing Stop-Loss is hit during the day, sell at the Trailing Stop price
            (price.Trailing_Stop.shift(1) / price.Close.shift(1)),
            np.where(
                (price.Long.shift(1) == True) & (price.High > price.Take_Profit.shift(1)),
                # If Take Profit is hit during the day, sell at the Take Profit price
                (price.Take_Profit.shift(1) / price.Close.shift(1)),
                # Otherwise, apply the normal return
                np.where(price.Long.shift(1) == True, price.Return, 1)
            )
        )
    )
    price['Sys_Bal'] = (STARTING_BALANCE * price.Sys_Ret.cumprod())
    
    price['Sys_Peak'] = price.Sys_Bal.cummax()
    price['Sys_DD'] = price.Sys_Bal - price.Sys_Peak
    
    bench_cagr = round(((((price.Bench_Bal.iloc[-1]/price.Bench_Bal.iloc[0])**(1/YEARS))-1)*100), 2)
    sys_cagr = round(((((price.Sys_Bal.iloc[-1]/price.Sys_Bal.iloc[0])**(1/YEARS))-1)*100), 2)
    
    
    
    return sys_cagr
  


def bayesianOpt():
    def objective(trial):
        PCT_THRESH = trial.suggest_uniform('PCT_THRESH', 0, 100)
        RANGE_THRESH = trial.suggest_uniform('RANGE_THRESH', 0, 100)
        TAKE_PROFIT_PCT = trial.suggest_uniform('TAKE_PROFIT_PCT', 0, 100)
        TRAILING_STOP_PCT = trial.suggest_uniform('TRAILING_STOP_PCT', 0, 100)

        # Run the backtesting function with the suggested hyperparameters
        return backtest(PCT_THRESH, RANGE_THRESH, TAKE_PROFIT_PCT, TRAILING_STOP_PCT)

    study = optuna.create_study(direction='maximize')  # Optimize to maximize the value
    study.optimize(objective, n_trials=500)

    print('Best trial:')
    trial = study.best_trial
    print('  TAKE_PROFIT_PCT:', trial.params['TAKE_PROFIT_PCT'])
    print('  TRAILING_STOP_PCT:', trial.params['TRAILING_STOP_PCT'])
    print('  PCT_THRESH:', trial.params['PCT_THRESH']) 
    print('  RANGE_THRESH', trial.params['RANGE_THRESH'])
    print('  CAGR:', trial.value)

    
bayesianOpt()
