from django.shortcuts import render
import yfinance as yf
from django.http import JsonResponse
import random
import pandas as pd
import numpy as np

openings = ["TSLA", "MSFT", "AAPL", "NVDA", "NFLX", "BA", "SPY", "RHM"]

def index(request):
    return render(request, 'index.html')

def get_opening_data(request):
    ticker = request.GET.get('ticker')
    if not ticker:
        ticker = random.choice(openings)
    
    stock_data = yf.download(ticker, start="2023-01-01", end="2024-01-01")
    
    data = {
        'ticker': ticker,
        'dates': stock_data.index.strftime('%Y-%m-%d').tolist(),
        'close_prices': stock_data['Close'].values.tolist()
    }
    return JsonResponse(data)

def calculate_metrics(data):
    close_prices = data['close_prices']
    dates = pd.to_datetime(data['dates'])

    # Calculate daily returns
    returns = np.diff(close_prices) / close_prices[:-1]
    avg_return = np.mean(returns)
    total_return = (close_prices[-1] - close_prices[0]) / close_prices[0]
    volatility = np.std(returns)

    # Calculate maximum drawdown
    cumulative_returns = np.cumprod(1 + returns) - 1
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    # Calculate CAGR
    num_years = (dates[-1] - dates[0]).days / 365.25
    CAGR = (close_prices[-1] / close_prices[0]) ** (1 / num_years) - 1

    # Assuming a risk-free rate of 0.02 (2%) for Sharpe ratio
    risk_free_rate = 0.02
    sharpe_ratio = (avg_return - risk_free_rate) / volatility * np.sqrt(252)

    # Calmar Ratio
    calmar_ratio = CAGR / abs(max_drawdown) if max_drawdown != 0 else np.nan

    metrics = {
        "ticker": data['ticker'],
        "average_return": avg_return,
        "total_return": total_return,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "CAGR": CAGR,
        "sharpe_ratio": sharpe_ratio,
        "calmar_ratio": calmar_ratio
    }

    return metrics

def get_metrics(request):
    ticker = request.GET.get('ticker')
    if not ticker:
        return JsonResponse({'error': 'Ticker symbol is required'}, status=400)
    
    stock_data = yf.download(ticker, start="2023-01-01", end="2024-01-01")
    
    data = {
        'ticker': ticker,
        'dates': stock_data.index.strftime('%Y-%m-%d').tolist(),
        'close_prices': stock_data['Close'].values.tolist()
    }
    
    # Calculate metrics separately
    metrics = calculate_metrics(data)
    return JsonResponse(metrics)

def data_page(request):
    return render(request, 'plot.html')

def home(request):
    return render(request, 'home.html')

