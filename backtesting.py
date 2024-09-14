import catboost
import numpy as np
import pandas as pd
import yfinance as yf
import talib as ta
import matplotlib.pyplot as plt
import hyperopt
import pickle
import random

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from hyperopt import hp
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

SYMBOL = 'QQQ'
START = '2000-01-01'
END = '2025-01-01'

N_ESTIMATORS = 100
MAX_ITER = 200
LEARNING_RATE = 0.5
MAX_DEPTH = 7

MODEL_PATH = 'C:/Users/juliu/Code/.vscode/Projects/Python/Trading Bot/trained_model_2.0.pkl'

def download_data(symbol, start, end):
    timeframes = {
        '1d': yf.download(symbol, start=start, end=end, interval="1d"),
        '1wk': yf.download(symbol, start=start, end=end, interval="1wk"),
        '1mo': yf.download(symbol, start=start, end=end, interval="1mo")
    }
    
    for tf in timeframes:
        timeframes[tf].index = pd.to_datetime(timeframes[tf].index, format="%Y-%m-%d %H:%M:%S")
    
    return timeframes

# Rename columns for each timeframe
def rename_columns(timeframes):
    for tf, df in timeframes.items():
        df.rename(columns={col: f"{col}_{tf}" for col in df.columns if col != "Adj Close"}, inplace=True)
        df.drop(["Adj Close"], axis=1, inplace=True)



def prepare_data(timeframes):
    # Merge DataFrames
    merged_df = pd.merge(timeframes['1d'], timeframes['1wk'], on='Date', how='outer')
    merged_df = pd.merge(merged_df, timeframes['1mo'], on='Date', how='outer')
    
    merged_df.ffill(inplace=True)

    # # Drop unnecessary columns
    # drop_columns = ['Close_1d', 'Open_1d', 'High_1d', 'Low_1d', 'Volume_1d',
    #                 'Close_1wk', 'Open_1wk', 'High_1wk', 'Low_1wk', 'Volume_1wk',
    #                 'Close_1mo', 'Open_1mo', 'High_1mo', 'Low_1mo', 'Volume_1mo']
    # merged_df.drop(drop_columns, axis=1, inplace=True)
    
    return merged_df


def calculate_indicators(df):
    ema_periods = [9, 20, 50, 100, 200]
    for tf in ['1d']:
        for period in ema_periods:
            col_name = f"EMA_{tf}_{period}"
            df[col_name] = ta.EMA(df[f"Close_{tf}"], timeperiod=period)

    for tf in ['1d']:
        # Calculate RSI
        df[f"RSI_{tf}"] = ta.RSI(df[f"Close_{tf}"], timeperiod=14)

        # Calculate ATR
        df[f"ATR_{tf}"] = ta.ATR(df[f"High_{tf}"], df[f"Low_{tf}"], df[f"Close_{tf}"], timeperiod=14)

        # Calculate MACD
        df[f"MACD_{tf}"], df[f"MACD_SIGNAL_{tf}"], df[f"MACD_HIST_{tf}"] = ta.MACD(df[f"Close_{tf}"], fastperiod=12, slowperiod=26, signalperiod=9)

        # Calculate PSAR
        df[f'SAR_{tf}'] = ta.SAR(df[f'High_{tf}'], df[f'Low_{tf}'], acceleration=0.02, maximum=0.2)

        # Calculate ADX
        df[f"ADX_{tf}"] = ta.ADX(df[f"High_{tf}"], df[f"Low_{tf}"], df[f"Close_{tf}"], timeperiod=14)

        # Calculate OBV
        df[f"OBV_{tf}"] = ta.OBV(df[f"Close_{tf}"], df[f"Volume_{tf}"])

        # Calculate VAR
        df[f"VAR_{tf}"] = ta.VAR(df[f"Close_{tf}"], timeperiod=5)

        # Calculate TSF
        df[f"TSF_{tf}"] = ta.TSF(df[f"Close_{tf}"], timeperiod=14)

        # Calculate HT_DCPHASE
        df[f"HT_DCPHASE_{tf}"] = ta.HT_DCPHASE(df[f"Close_{tf}"])

        # Calculate HT_DCPeriod
        df[f"HT_DCPeriod_{tf}"] = ta.HT_DCPERIOD(df[f"Close_{tf}"])

def define_model(df):
    # Define features
    df['Target'] = np.where(df.Close_1d < df.Close_1d.shift(-1), True, False)
    X = df.drop(['Open_1wk', 'High_1wk', 'Low_1wk', 'Close_1wk', 'Volume_1wk',
                 'Open_1mo', 'High_1mo', 'Low_1mo', 'Close_1mo', 'Volume_1mo'], axis=1)  # features
    y = df['Target']  # target variable
    
    # Create train and test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67, shuffle=False)

    # Train model
    model_params = {
        'gb': {'learning_rate': LEARNING_RATE, 'max_iter': MAX_ITER, 'max_depth': MAX_DEPTH},
        'cb': {'learning_rate': LEARNING_RATE, 'depth': MAX_DEPTH, 'iterations': MAX_ITER},
        'rf': {'n_estimators': N_ESTIMATORS, 'max_depth': MAX_DEPTH}
    }
    #train_model(X_train, y_train, model_params)
     
    return X, y, X_train, X_test, y_train, y_test


def train_model(X_train, y_train, model_params):
    # Define base models
    estimators = [
        ('gb', HistGradientBoostingClassifier(**model_params['gb'])),
        ('cb', catboost.CatBoostClassifier(**model_params['cb'], task_type="GPU")),
        ('rf', RandomForestClassifier(**model_params['rf']))
    ]

    # Define stacking model
    model = StackingClassifier(estimators=estimators, final_estimator=catboost.CatBoostClassifier(task_type="GPU"))

    # Train stacking model
    model.fit(X_train, y_train)
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
        
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model
    
def test_model_once(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(y_pred, y_test)
    print(f"Accuracy: {accuracy:.2f}")
    
def test_and_train_score(model, X, y, X_test, y_test, n=10, initial_investment=10000):
    # Initialize the scoring system
    best_score = float('-inf')  # Initially set to a very low number
    best_model = None
    total_profit = 0

    # Split the data into n segments for time-based cross-validation
    segment_size = len(X) // n
    
    for i in range(n):
        # Define the training and test sets based on the current segment
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size if i < n - 1 else len(X)

        X_train_segment = X.iloc[:start_idx]
        y_train_segment = y.iloc[:start_idx]

        X_test_segment = X.iloc[start_idx:end_idx]
        y_test_segment = y.iloc[start_idx:end_idx]
        
        X_test_segment.ffill()
        y_test_segment.ffill()
        
        # Check if the segment has at least one sample
        if len(X_train_segment) == 0 or len(X_test_segment) == 0:
            print(f"Skipping segment {i+1}/{n} due to insufficient samples.")
            continue

        # Train the model on the training segment
        model.fit(X_train_segment, y_train_segment)

        # Make predictions on the test segment
        y_pred = model.predict(X_test_segment)

        # Calculate the score based on price changes
        price_changes = X_test_segment['Close_1d'].diff().shift(-1)  # Shift to align with next day's price
        profit = np.nansum(np.where(y_pred == True, price_changes, -price_changes))  # Buy when True, otherwise sell
        
        total_profit += profit
        print(f"Iteration {i+1}/{n}, Profit: {profit:.2f}")

        # Update the best model if current profit is better
        if total_profit > best_score:
            best_score = total_profit
            best_model = model

    # Calculate the percentage return on investment
    percentage_return = (total_profit / initial_investment) * 100
    print(f"Best total profit after {n} iterations: {percentage_return:.2f}%")

    # Save the best model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

        

   

def main():
    timeframes = download_data(SYMBOL, START, END)
    rename_columns(timeframes)
    df = prepare_data(timeframes)
    calculate_indicators(df)
    X, y, X_train, X_test, y_train, y_test = define_model(df)
    model = load_model()
    test_and_train_score(model, X, y, X_test, y_test)
    
    
if __name__ == '__main__':
    main()
