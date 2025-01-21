import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data

import plotly.graph_objects as go

plt.style.use('fivethirtyeight')

ticker = "AAPL"
start = dt.datetime(2000, 1, 1)
end = dt.datetime(2024, 1, 1)

stockData = yf.download(ticker, start, end)
stockData.reset_index(inplace=True)

stockData.to_csv("aapl.csv")
data01 = pd.read_csv("aapl.csv")


figure = go.Figure(data=[go.Candlestick(x = data01['Date'],
                                        open = data01['Open'],
                                        high = data01['High'],
                                        low = data01['Low'],
                                        close = data01['Close'])])
figure.update_layout(xaxis_rangeslider_visible=True)

stockData = stockData.drop(['Date'], axis=1)
print(stockData.head())

plt.figure(figsize=(12,6))
plt.plot(stockData['Close'], label = f'{ticker} Closing Price', linewidth = 2)
plt.title(f'{ticker} Closing prices over time')
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
plt.plot(stockData['Open'], label = f'{ticker} Opening Price', linewidth = 2)
plt.title(f'{ticker} Opening prices over time')
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
plt.plot(stockData['High'], label = f'{ticker} High Price', linewidth = 2)
plt.title(f'{ticker} High prices over time')
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
plt.plot(stockData['Volume'], label = f'{ticker} Volume Price', linewidth = 2)
plt.title(f'{ticker} Volume prices over time')
plt.legend()
plt.show()
