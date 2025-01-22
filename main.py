import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

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



movAvg100 = stockData.Close.rolling(100).mean()
print(movAvg100)
movAvg200 = stockData.Close.rolling(200).mean()
print(movAvg200)

plt.figure(figsize=(12,6))
plt.plot(stockData['Close'], label = f'{ticker} Close Price', linewidth = 2)
plt.plot(movAvg100, label = f'{ticker} Moving Average 100 Price', linewidth = 2)
plt.plot(movAvg200, label = f'{ticker} Moving Average 200 Price', linewidth = 2)
plt.legend()
plt.show()

expMovAvg100 = stockData.Close.ewm(span=100, adjust=False).mean()
expMovAvg200 = stockData.Close.ewm(span=200, adjust=False).mean()

plt.figure(figsize=(12,6))
plt.plot(stockData['Close'], label = f'{ticker} Close Price', linewidth = 2)
plt.plot(movAvg100, label = f'{ticker} Exp. Moving Average 100 Price', linewidth = 2)
plt.plot(movAvg200, label = f'{ticker} Exp. Moving Average 200 Price', linewidth = 2)
plt.legend()
plt.show()

#train and testing
dataTraining = pd.DataFrame(stockData['Close'][0:int(len(stockData)*.70)])
dataTesting = pd.DataFrame(stockData['Close'][0:int(len(stockData)*.70): int(len(stockData))])
print(dataTraining.shape)
print(dataTesting.shape)

scaler = MinMaxScaler(feature_range=(0,1))
dataTrainingArray = scaler.fit_transform(dataTraining)
print(dataTrainingArray)

xTrain = []
yTrain = []

for i in range(100, dataTrainingArray.shape[0]):
    xTrain.append(dataTrainingArray[i-100:i])
    yTrain.append(dataTrainingArray[i,0])

xTrain = np.array(xTrain) 
yTrain = np.array(yTrain)

print(xTrain.shape)

#Building the model
model = Sequential()

model.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (xTrain.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.4))

model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))

print(model.summary())

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(xTrain, yTrain, epochs = 50)

# testing the model
last100Days = dataTraining.tail(100)

finalDf = pd.concat([last100Days, dataTesting], ignore_index=True)

inputData = scaler.transform(finalDf)

xTest = []
yTest = []

for i in range(100, inputData.shape[0]):
    xTest.append(inputData[i-100:i])
    yTest.append(inputData[i,0])

xTest = np.array(xTest) 
yTest = np.array(yTest)

yPredicted = model.predict(xTest)


scaleFactor = 1/0.03383118

yPredicted = yPredicted * scaleFactor
yTest = yTest * scaleFactor

plt.figure(figsize=(12,6))
plt.plot(yTest, label = 'Orignal Price', linewidth = 1)
plt.plot(yPredicted, label = 'Predicted Price', linewidth = 1)
plt.legend()
plt.show()

model.save('stockModel.h5')