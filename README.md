# ONON-DECK
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# 1. Obtener datos desde 2024
start_date = "2024-01-01"
end_date = "2025-03-14"  # Hasta la fecha actual

# Descargar datos de ONON y DECK (usamos una lista de tickers para manejar MultiIndex)
tickers = ["ONON", "DECK"]
data_raw = yf.download(tickers, start=start_date, end=end_date)

# Verificar las columnas disponibles
print("Columnas disponibles:")
print(data_raw.columns)

# Extraer precios de cierre ('Close') del MultiIndex
if ('Adj Close', 'ONON') in data_raw.columns:
    on_prices = data_raw['Adj Close']['ONON']
    deck_prices = data_raw['Adj Close']['DECK']
    print("Usando 'Adj Close'")
else:
    on_prices = data_raw['Close']['ONON']
    deck_prices = data_raw['Close']['DECK']
    print("Usando 'Close' en lugar de 'Adj Close'")

# Crear DataFrame combinado
data = pd.DataFrame({'ONON': on_prices, 'DECK': deck_prices}).dropna()

# 2. Prueba de estacionariedad (Augmented Dickey-Fuller Test)
def adf_test(series, title=''):
    result = adfuller(series.dropna())
    print(f'ADF Test para {title}:')
    print(f'Estadístico ADF: {result[0]}')
    print(f'p-valor: {result[1]}')
    print('Valores críticos:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')
    if result[1] < 0.05:
        print("Resultado: La serie es estacionaria\n")
    else:
        print("Resultado: La serie NO es estacionaria\n")

adf_test(data['ONON'], 'ONON')
adf_test(data['DECK'], 'DECK')

# Diferenciar las series si no son estacionarias
data_diff = data.diff().dropna()
adf_test(data_diff['ONON'], 'ONON Diferenciada')
adf_test(data_diff['DECK'], 'DECK Diferenciada')

# 3. Prueba de cointegración
score, p_value, _ = coint(data['ONON'], data['DECK'])
print('Prueba de Cointegración entre ONON y DECK:')
print(f'P-valor: {p_value}')
if p_value < 0.05:
    print("Resultado: Las series están cointegradas\n")
else:
    print("Resultado: Las series NO están cointegradas\n")

# 4. Modelos AR, ARMA y ARIMA para ONON
ar_model_on = AutoReg(data['ONON'], lags=1).fit()
print("Modelo AR(1) para ONON:")
print(ar_model_on.summary())

arma_model_on = ARIMA(data_diff['ONON'], order=(1, 0, 1)).fit()
print("\nModelo ARMA(1,1) para ONON:")
print(arma_model_on.summary())

arima_model_on = ARIMA(data['ONON'], order=(1, 1, 1)).fit()
print("\nModelo ARIMA(1,1,1) para ONON:")
print(arima_model_on.summary())

# 5. Forecasting para ONON
forecast_steps = 20
forecast_ar = ar_model_on.predict(start=len(data), end=len(data)+forecast_steps-1)
forecast_arma = arma_model_on.forecast(steps=forecast_steps)
forecast_arima = arima_model_on.forecast(steps=forecast_steps)

last_date = data.index[-1]
forecast_index = pd.date_range(start=last_date, periods=forecast_steps+1, freq='B')[1:]

# 6. Visualización
plt.figure(figsize=(12, 6))
plt.plot(data['ONON'], label='Datos Reales ONON')
plt.plot(forecast_index, forecast_ar, label='Forecast AR(1)', linestyle='--')
plt.plot(forecast_index, forecast_arima, label='Forecast ARIMA(1,1,1)', linestyle='--')
plt.title('Forecasting de ONON')
plt.legend()
plt.show()

# Repetir para DECK
ar_model_deck = AutoReg(data['DECK'], lags=1).fit()
arma_model_deck = ARIMA(data_diff['DECK'], order=(1, 0, 1)).fit()
arima_model_deck = ARIMA(data['DECK'], order=(1, 1, 1)).fit()

forecast_ar_deck = ar_model_deck.predict(start=len(data), end=len(data)+forecast_steps-1)
forecast_arma_deck = arma_model_deck.forecast(steps=forecast_steps)
forecast_arima_deck = arima_model_deck.forecast(steps=forecast_steps)

plt.figure(figsize=(12, 6))
plt.plot(data['DECK'], label='Datos Reales DECK')
plt.plot(forecast_index, forecast_ar_deck, label='Forecast AR(1)', linestyle='--')
plt.plot(forecast_index, forecast_arima_deck, label='Forecast ARIMA(1,1,1)', linestyle='--')
plt.title('Forecasting de DECK')
plt.legend()
plt.show()
