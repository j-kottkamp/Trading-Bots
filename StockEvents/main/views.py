from django.shortcuts import render
import yfinance as yf
from django.http import JsonResponse
import random

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

def fetch_data(request):
    if request.method == 'POST':
            ticker = request.POST.get('ticker')
            if ticker:
                stock_data = yf.download(ticker, start="2023-10-01", end="2024-01-01")
                
                data = {
                    'dates': stock_data.index.strftime('%Y-%m-%d').tolist(),
                    'close_prices': stock_data['Close'].values.tolist()
                }
                return JsonResponse(data)

def data_page(request):
    return render(request, 'plot.html')

def home(request):
    return render(request, 'home.html')

