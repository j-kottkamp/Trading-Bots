import pandas as pd
import yfinance as yf
import time
from yahoo_fin import stock_info as si
import pandas as pd

# Fetch tickers for NASDAQ and NYSE
nasdaq_tickers = si.tickers_nasdaq()


# Combine and save to CSV
all_tickers = list(set(nasdaq_tickers))
pd.DataFrame(all_tickers, columns=["Ticker"]).to_csv("yfinance_tickers.csv", index=False)

print(f"Saved {len(all_tickers)} tickers to 'yfinance_tickers.csv'.")

def f():
    START = "2019, 1, 1"
    END = "2025, 1, 1"

    # Load tickers from a CSV file
    tickers = pd.read_csv('tickers.csv')['Symbol'].tolist()

    def fetch_yahoo_data(ticker):
        try:
            data = yf.download(ticker, start=START, end=END)
            return data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None

