#imports
import pandas as pd
# stock data
import yfinance as yf
# model trainer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import XGBRegressor
# model accuracy tester
from sklearn.metrics import precision_score


# get stock info from ticker
def get_stock_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    ticker = ticker.history(period="max")

ticker = yf.Ticker("AAPL")
ticker = ticker.history(period="max")

# show graph
# ticker.plot.line(y="Close", use_index=True)
# remove irrelevent columns
del ticker["Dividends"]
del ticker["Stock Splits"]

# add tomorrow column which shows the next day's closing price by shifting the data by 1
ticker["Target"] = ticker["Close"].shift(-1)

# add target column to see if tomorrows price is higher than current
# ticker["Target"] = (ticker["Tomorrow"] > ticker["Close"]).astype(int)

# remove irrelevent data before 1990
ticker = ticker.loc["1990-01-01":].copy()

# initialze model based on estimator decison trees, sample splits overfitting data the higher the less accurate, random state prevents model changes per run
# model = RandomForestClassifier(n_estimators=2000, min_samples_split=25, random_state=1)
# model = XGBClassifier(n_estimators=6000, learning_rate=0.009, max_depth=5, random_state=1)
model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=1)
# model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=True)

# train model
# trains without last 100 rows
# train = ticker.iloc[:-100]
# tests with last 100 rows
# test = ticker.iloc[-100:]

# Use predictor columns to try and predict target value
predictors = ["Close", "Volume", "Open", "High", "Low"]

# model training function
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    # test model accuracy from  predictors
    predictions = model.predict(test[predictors])
    # predictions[predictions >= .6] = 1
    # predictions[predictions < .6] = 0
    predictions = pd.Series(predictions, index=test.index, name="Predictions")
    #combine targets and predidtions
    combined = pd.concat([test["Target"], predictions], axis=1)
    return combined

# backtest model
# each trading year is 250 days, trains model with 10 years of data, trains model per year
def backtest(data, model, predictors, start=2500, step=250):
    allPredictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        allPredictions.append(predictions) 
    return pd.concat(allPredictions)

# predictions = backtest(ticker, model, predictors)
# predictions["Predictions"].value_counts()

horizons = [2, 5, 60, 250, 1000]
newPredictors = []

for horizon in horizons:
    rollingAvg = ticker.rolling(horizon).mean()

    ratioCol = f"Close_Ratio_{horizon}"
    ticker[ratioCol] = ticker["Close"] / rollingAvg["Close"]

    trendCol = f"Trend_{horizon}"
    ticker[trendCol] = ticker.shift(1).rolling(horizon).sum()["Target"]

    newPredictors += [ratioCol, trendCol]
# Moving Averages
ticker["SMA_50"] = ticker["Close"].rolling(window=50).mean()
ticker["SMA_200"] = ticker["Close"].rolling(window=200).mean()

# Relative Strength Index (RSI)
delta = ticker["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
ticker["RSI_14"] = 100 - (100 / (1 + rs))

# MACD (12-day EMA - 26-day EMA)
ema_12 = ticker["Close"].ewm(span=12, adjust=False).mean()
ema_26 = ticker["Close"].ewm(span=26, adjust=False).mean()
ticker["MACD"] = ema_12 - ema_26

# Bollinger Bands (Upper & Lower)
std_20 = ticker["Close"].rolling(window=20).std()
ticker["Bollinger_Upper"] = ticker["SMA_50"] + (std_20 * 2)
ticker["Bollinger_Lower"] = ticker["SMA_50"] - (std_20 * 2)

# Add new predictors
newPredictors += ["SMA_50", "SMA_200", "RSI_14", "MACD", "Bollinger_Upper", "Bollinger_Lower"]

ticker = ticker.dropna()
predictions = backtest(ticker, model, newPredictors)

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(predictions["Target"], predictions["Predictions"])
print(f"Mean Absolute Error (MAE): {mae:.2f}")

latest_data = ticker.iloc[-1:]  # Get the last row of data
tomorrow_pred = model.predict(latest_data[newPredictors])
print(f"Predicted closing price for tomorrow: {tomorrow_pred[0]:.2f}")