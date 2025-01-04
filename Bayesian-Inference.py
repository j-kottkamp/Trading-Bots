import yfinance as yf
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import matplotlib.pyplot as plt
import matplotlib.style as style
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

def GetData(symbol, start_date, end_date, forecasting_length):
  

    data = yf.download(symbol, start=start_date, end=end_date)
    
    data['Returns'] = data['Adj Close'].pct_change().dropna()
    data['Returns'] = data['Returns'] * 1000
    
    data['LogReturns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1)).dropna()
    data['LogReturns'] = data['LogReturns'] * 1000

    # frequency
    data = data.asfreq('D').ffill()
    data.index = pd.to_datetime(data.index)
    
    return data

def OptArima(data, forecasting_length):
    model_arima = auto_arima(data['LogReturns'].dropna(), seasonal=True, m=12, trace=True, suppress_warnings=True)
    arima_forecast = model_arima.predict(n_periods=forecasting_length)
    
    return arima_forecast

def Arima(data, forecasting_length):
    model_arima = ARIMA(data['LogReturns'].dropna(), order=(3, 0, 3))
    fitted_arima = model_arima.fit()

    arima_forecast = fitted_arima.forecast(steps=forecasting_length)
    
    return arima_forecast

def Egarch(data, forecasting_length):
    # GARCH-Modell zur Vorhersage der Volatilität
    model_egarch = arch_model(data['LogReturns'].dropna(), vol='EGARCH', p=1, o=1, q=1)
    fitted_egarch = model_egarch.fit(update_freq=5, disp="off")

    # GARCH-Volatilitäts-Prognose für die nächsten 5 Tage
    egarch_forecast = fitted_egarch.forecast(horizon=forecasting_length)
    egarch_volatility = np.sqrt(egarch_forecast.variance.values[-1, :])  # Standardabweichung
    
    return egarch_volatility

def Garch(data, forecasting_length):
    # GARCH-Modell zur Vorhersage der Volatilität
    model_garch = arch_model(data['LogReturns'].dropna(), vol='Garch', p=1, q=1)
    fitted_garch = model_garch.fit()

    # GARCH-Volatilitäts-Prognose für die nächsten 5 Tage
    garch_forecast = fitted_garch.forecast(horizon=forecasting_length)
    garch_volatility = np.sqrt(garch_forecast.variance.values[-1, :])  # Standardabweichung
    
    return garch_volatility

def CombArimaGarch(data, forecasting_length, arima_forecast, garch_volatility):
    last_price = data['Adj Close'].iloc[-1]

    predicted_prices = []
    for r, vol in zip(arima_forecast, garch_volatility):
        predicted_price = last_price * np.exp(r / 1000 + (0.5 * (vol / 1000) ** 2))
        print(r / 1000, vol / 1000)
        predicted_prices.append(predicted_price)

    # Extract numeric valus
    predicted_prices = [price.item() for price in predicted_prices]
    forecast_dates = pd.bdate_range(start=data.index[-1], periods=forecasting_length + 1, freq='B')[1:]
    
    return predicted_prices, forecast_dates

def PrintForecast(predicted_prices, forecast_dates):
    print("Prognostizierte Preise für die nächsten 5 Tage:")
    for date, price in zip(forecast_dates, predicted_prices):
        # Format the date to exclude the time component
        formatted_date = date.strftime('%Y-%m-%d')
        # Print the date and corresponding price
        print(f"- {formatted_date}: {price:.2f}")

def PlotForecast(data, predicted_prices, forecast_dates, symbol):
    data['Returns'] = np.exp(data['LogReturns'] / 1000) - 1  # Umrechnung von log-Renditen in Prozent
    
    style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10,6))
    plt.plot(data.index, data['Adj Close'], label='Tägliche Preise')
    plt.plot(forecast_dates, predicted_prices, label='Prognostizierte Preise', linestyle='--', color='red')
    plt.legend()
    plt.title(f'Preisprognose für {symbol} mit ARIMA und GARCH')
    plt.xlabel('Datum')
    plt.ylabel('Preis')
    plt.show()

def PlotGarch(fitted_egarch):
    """
    Plottet diagnostische Grafiken für das EGARCH-Modell.
    """
    fitted_egarch.plot(annualize='D')
    plt.show()

def Switch(data, forecast, IsGarch, IsArima):  
    garch_functions = {
        0: Garch,
        1: Egarch
    }
    
    arima_functions = {
        0: Arima,
        1: OptArima
    }
    
    garch_forecast = garch_functions[IsGarch](data, forecast)
    
    arima_forecast = arima_functions[IsArima](data, forecast)
    
    return garch_forecast, arima_forecast

    
def main():
    Ticker = 'SPY'
    start = '2023-01-01'
    end = '2024-12-09'
    forecast = 5
    Garch = 0
    Arima = 0
    
    data = GetData(Ticker, start, end, forecast)
    
    garch_forecast, arima_forecast = Switch(data, forecast, Garch, Arima)
    
    predicted_prices, forecast_dates = CombArimaGarch(data, forecast, arima_forecast, garch_forecast)
    
    PrintForecast(predicted_prices, forecast_dates)

    PlotForecast(data, predicted_prices, forecast_dates, Ticker)
    
    

if __name__ == "__main__":
    main()
