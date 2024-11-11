from django.shortcuts import render
import yfinance as yf

def index(request):
    return render(request, 'index.html')

def plot_graph(request):
    data = {
        'x': [1, 2, 3, 4],
        'y': [10, 15, 13, 17],
    }
    return render(request, 'plot.html', {'data': data})

def home(request):
    return render(request, 'home.html')