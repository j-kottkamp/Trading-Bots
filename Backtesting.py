import yfinance as yf
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.style as style
import datetime
import talib as ta

SYMBOL = "SPY"
STARTING_BALANCE = 10000

RANGE_THRESH = 0.10980877710302735
PCT_THRESH = 59.83610083957252

TAKE_PROFIT_PCT = 7.526225504672157
TRAILING_STOP_PCT = 88.50615899653582

START = datetime.datetime(2000, 1, 1)
END = datetime.datetime(2024, 9, 1)
YEARS = (END - START).days / 365.25

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

style.use('dark_background')
plt.figure(figsize=(12, 6))
plt.plot(price.Bench_Bal, label='Benchmark')
plt.plot(price.Sys_Bal, label='System', color="g")
plt.legend()
plt.show()

price['Sys_Peak'] = price.Sys_Bal.cummax()
price['Sys_DD'] = price.Sys_Bal - price.Sys_Peak

sys_dd = round(((price.Sys_DD / price.Sys_Peak).min()) * 100, 2)

bench_return = round(((price.Bench_Bal.iloc[-1]/price.Bench_Bal.iloc[0]) - 1) * 100, 2)
bench_cagr = round(((((price.Bench_Bal.iloc[-1]/price.Bench_Bal.iloc[0])**(1/YEARS))-1)*100), 2)
sys_return = round(((price.Sys_Bal.iloc[-1]/price.Sys_Bal.iloc[0]) - 1) * 100, 2)
sys_cagr = round(((((price.Sys_Bal.iloc[-1]/price.Sys_Bal.iloc[0])**(1/YEARS))-1)*100), 2)
sys_in_market = round((price.Long.value_counts().loc[True] / len(price)) * 100)
sys_win = price.Sys_Ret[price.Sys_Ret > 1.0].count()
sys_loss = price.Sys_Ret[price.Sys_Ret < 1.0].count()
sys_winrate = round(sys_win / (sys_win + sys_loss) * 100, 2)

daily_sys_ret = price['Sys_Ret'].dropna() - 1
sys_return_mean = daily_sys_ret.mean()
sys_return_std = daily_sys_ret.std()
sharpe_ratio_sys = (sys_return_mean / sys_return_std) * np.sqrt(252)

daily_bench_ret = price['Return'].dropna() - 1
bench_return_mean = daily_bench_ret.mean()
bench_return_std = daily_bench_ret.std()
sharpe_ratio_bench = (bench_return_mean / bench_return_std) * np.sqrt(252)

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

def moreInfo():
    # Zählen der gewonnenen und verlorenen Trades
    positive_trades = price.Sys_Ret[price.Sys_Ret > 1.0]
    negative_trades = price.Sys_Ret[price.Sys_Ret < 1.0]

    # Berechnung der Metriken
    total_trades = len(positive_trades) + len(negative_trades)
    gross_profit = (positive_trades - 1).sum() * STARTING_BALANCE  # Bruttogewinn in absoluten Zahlen
    gross_loss = abs((negative_trades - 1).sum()) * STARTING_BALANCE  # Bruttoverlust in absoluten Zahlen
    average_win = (positive_trades - 1).mean() * 100  # Durchschnittlicher Gewinn pro Trade in Prozent
    average_loss = (negative_trades - 1).mean() * 100  # Durchschnittlicher Verlust pro Trade in Prozent
    average_trade = (price.Sys_Ret - 1).mean() * 100  # Durchschnittlicher Gewinn/Verlust pro Trade insgesamt in Prozent
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.nan  # Profit Factor
    percentage_profitable = len(positive_trades) / total_trades * 100 if total_trades > 0 else 0

    # Ausgabe der Ergebnisse
    print(f"Total Trades: {total_trades}")
    print(f"Positive Trades: {len(positive_trades)}")
    print(f"Negative Trades: {len(negative_trades)}")
    print(f"Average Win: {average_win:.2f}%")
    print(f"Average Loss: {average_loss:.2f}%")
    print(f"Average Trade: {average_trade:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Percentage Profitable Trades: {percentage_profitable:.2f}%")

    # Plot erstellen
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    # Anzahl der positiven und negativen Trades
    ax[0, 0].bar(['Positive Trades', 'Negative Trades'], [len(positive_trades), len(negative_trades)], color=['green', 'red'])
    ax[0, 0].set_title('Number of Trades')
    ax[0, 0].set_ylabel('Number of Trades')

    # Durchschnittliche Gewinne und Verluste
    ax[0, 1].bar(['Average Win', 'Average Loss'], [average_win, average_loss], color=['green', 'red'])
    ax[0, 1].set_title('Average Gain/Loss per Trade (%)')

    # Durchschnittlicher Gewinn/Verlust pro Trade
    ax[1, 0].bar(['Average Trade'], [average_trade], color=['blue'])
    ax[1, 0].set_title('Average Trade (%)')

    # Profit Factor und Erfolgsrate
    ax[1, 1].bar(['Profit Factor', 'Percentage Profitable'], [profit_factor, percentage_profitable], color=['purple', 'orange'])
    ax[1, 1].set_title('Profit Factor & Percentage Profitable (%)')

    plt.tight_layout()
    plt.show()

def backtesting():

    #define list of ETFs to backtest
    symbols = ['^GSPC', '^DJI', '^IXIC', '^RUT', '^N225',]
    #create backtesting function
    def backtest(s):
        #download data
        price = yf.download(s, start=START, end=END)
        
        #drop redundant columns
        price = price.drop(['Volume', 'Adj Close'], axis=1)
        
        #calculate benchmark return and balance
        price['Return'] = price.Close / price.Close.shift(1)
        price.Return.iat[0] = 1
        price['Bench_Bal'] = STARTING_BALANCE * price.Return.cumprod()
        
        #calculate benchmark drawdown
        price['Bench_Peak'] = price.Bench_Bal.cummax()
        price['Bench_DD'] = price.Bench_Bal - price.Bench_Peak
        
        #calculate additional columns for strategy
        #daily range
        price['Range'] = price.High - price.Low
        #distance between close and daily low
        price['Dist'] = abs(price.Close - price.Low)
        #distance as % of range
        price['Pct'] = (price.Dist / price.Range) * 100
        
        #identify entries and allocate trading fees
        price['Long'] = np.logical_and((price.Pct < PCT_THRESH), (price.Range > 10))
        
        #calculate system return and balance
        price['Sys_Ret'] = np.where(price.Long.shift(1) == True, price.Return, 1)
        price['Sys_Bal'] = (STARTING_BALANCE * price.Sys_Ret.cumprod())
        
        #calculate system drawdown
        price['Sys_Peak'] = price.Sys_Bal.cummax()
        price['Sys_DD'] = price.Sys_Bal - price.Sys_Peak
        
        #calculate metrics
        bench_cagr = round(((((price.Bench_Bal.iloc[-1]/price.Bench_Bal.iloc[0])**(1/YEARS))-1)*100), 2)
        bench_dd = round((price.Bench_DD / price.Bench_Peak).min() * 100, 2)
        sys_cagr = round(((((price.Sys_Bal.iloc[-1]/price.Sys_Bal.iloc[0])**(1/YEARS))-1)*100), 2)
        sys_dd = round(((price.Sys_DD / price.Sys_Peak).min()) * 100, 2)
        
        return bench_cagr, sys_cagr, bench_dd, sys_dd

    # backtest multiple symbols
    bc = []
    sc = []
    best_symbol = None
    best_outperformance = 0

    for symbol in symbols:
        print(f"\nBacktesting symbol: {symbol}")
        bench_cagr, sys_cagr, bench_dd, sys_dd = backtest(symbol)
        bc.append(bench_cagr)
        sc.append(sys_cagr)
        outperformance = sys_cagr - bench_cagr
        if outperformance > best_outperformance:
            best_outperformance = outperformance
            best_symbol = symbol
        print(f"System CAGR: {sys_cagr}")
        print(f"Benchmark CAGR: {bench_cagr}")
        print(f"System DD: {sys_dd}")
        print(f"Benchmark DD: {bench_dd}")
        print(f"Outperformance: {outperformance}%")
        
    print(f"Best symbol: {best_symbol}")
    print(f"Best outperformance: {best_outperformance}%")

    # plot data
    x_indices = np.arange(len(symbols))
    width = 0.2

    plt.bar(x_indices - width / 2, bc, width = width, label = 'Benchmark', color="blue")
    plt.bar(x_indices + width / 2, sc, width = width, label = 'System', color="green")
    
    plt.xticks(ticks = x_indices, labels = symbols)

    plt.legend()

    plt.title('Backtest CAGR')
    plt.xlabel('Symbols')
    plt.ylabel('% CAGR')
    plt.tight_layout()

    plt.show()
    

