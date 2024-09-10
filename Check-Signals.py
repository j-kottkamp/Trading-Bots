import yfinance as yf
import pandas as pd

# Define entry and exit dates
signals = [
    ("2021-03-08", "2021-03-09"),
    ("2021-03-24", "2021-03-26"),
    ("2021-03-30", "2021-04-05"),
    ("2021-06-18", "2021-06-22"),
    ("2021-06-23", "2021-06-24"),
    ("2021-07-19", "2021-07-23"),
    ("2021-08-02", "2021-08-03"),
    ("2021-08-04", "2021-08-05"),
    ("2021-10-19", "2021-10-21"),
    ("2021-10-22", "2021-10-25"),
    ("2022-07-26", "2022-07-27"),
    ("2022-11-03", "2022-11-04"),
    ("2022-11-09", "2022-11-10"),
    ("2023-01-20", "2023-01-23"),
    ("2023-03-28", "2023-03-31"),
    ("2023-04-26", "2023-04-27"),
    ("2023-05-04", "2023-05-05"),
    ("2023-05-24", "2023-05-26"),
    ("2023-05-31", "2023-06-02"),
    ("2023-06-07", "2023-06-12"),
    ("2023-06-26", "2023-06-30"),
    ("2023-07-07", "2023-07-11"),
    ("2023-11-09", "2023-11-10"),
    ("2023-11-13", "2023-11-20"),
    ("2023-11-21", "2023-11-22"),
    ("2023-11-27", "2023-11-28"),
    ("2023-11-29", "2023-11-30"),
    ("2023-12-06", "2023-12-13"),
    ("2024-01-04", "2024-01-08"),
    ("2024-01-17", "2024-01-22"),
    ("2024-01-31", "2024-02-02"),
    ("2024-02-05", "2024-02-07"),
    ("2024-02-13", "2024-02-14"),
    ("2024-02-20", "2024-02-22"),
    ("2024-02-26", "2024-02-27"),
    ("2024-02-28", "2024-02-29"),
    ("2024-03-05", "2024-03-07"),
    ("2024-05-08", "2024-05-14"),
    ("2024-05-30", "2024-06-05"),
    ("2024-06-07", "2024-06-14"),
]


# Download historical SPY data
spy_data = yf.download("SPY", start="2021-01-01", end="2024-06-15")  # Covering full period

# Calculate the percentage returns for each entry and exit
returns = []
for entry, exit in signals:
    entry_price = spy_data.loc[entry, 'Close']
    exit_price = spy_data.loc[exit, 'Close']
    percentage_return = (exit_price / entry_price - 1) * 100
    returns.append(percentage_return)

# Calculate the total return of the S&P 500 from the first entry to the last exit
spy_total_return = (spy_data.loc[signals[-1][1], 'Close'] / spy_data.loc[signals[0][0], 'Close'] - 1) * 100

# Display the percentage returns for each trade and the total return
print("Returns per trade:", returns)
print("Total SPY return over the period:", spy_total_return)

# Calculate the total return of the trades
total_trade_return = sum(returns)

print("Total return of the trades:", total_trade_return)
