import streamlit as st 
import pandas as pd
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go

# Function to fetch stock data
def get_stock_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period="5y")  # Get last 5 years of data
    df["Tomorrow"] = df["Close"].shift(-1)  # Target variable
    return df.dropna()

# Function to train model and predict tomorrow’s price
def train_and_predict(df):
    predictors = ["Close", "Volume", "Open", "High", "Low"]
    
    model = XGBRegressor(n_estimators=500, learning_rate=0.009, max_depth=5, random_state=1)
    train = df.iloc[:-1]  # Use all data except the last row
    test = df.iloc[-1:]  # Last row for tomorrow's prediction
    
    model.fit(train[predictors], train["Tomorrow"])
    predicted_price = model.predict(test[predictors])[0]
    
    # Add predictions to DataFrame
    df["Predicted"] = model.predict(df[predictors])  # Predict for all past data
    return df, predicted_price

# Streamlit UI
st.title("Jinan's Stock Price Predictor")
ticker_input = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):", "AAPL")

if st.button("Predict"):
    df = get_stock_data(ticker_input)
    df, predicted_price = train_and_predict(df)

    st.subheader(f"Predicted Closing Price for {ticker_input} Tomorrow: **${predicted_price:.2f}**")

    # Interactive Plot with Plotly
    st.subheader("Stock Price Chart")
    fig = go.Figure()
    
    # Actual Prices
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], 
                             mode="lines", name="Actual Price", 
                             line=dict(color="blue")))
    
    # Predicted Prices
    fig.add_trace(go.Scatter(x=df.index, y=df["Predicted"], 
                             mode="lines", name="Predicted Price", 
                             line=dict(color="red", dash="dot")))

    # Customize layout
    fig.update_layout(title=f"{ticker_input} Stock Prices: Actual vs Predicted",
                      xaxis_title="Date",
                      yaxis_title="Stock Price",
                      hovermode="x",
                      template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)

    # Calculate MAE
    mae = mean_absolute_error(df["Close"], df["Predicted"])
    st.write(f"Mean Absolute Error (MAE): **${mae:.2f}**")
