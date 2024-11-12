from django.shortcuts import render
import yfinance as yf
from django.http import JsonResponse

def index(request):
    return render(request, 'index.html')

def get_historical_data(request):
    ticker = 'MSFT'
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

