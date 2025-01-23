import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go


# Initilize start and end date range for stock
start = "2000-01-01"
end = date.today().strftime("%Y-%m-%d")

st.title("Jinan Stock Prediction")

tickers = ("AAPL", "GOOG", "FB", "AMZN", "MSFT")
# Dropdown menu for stocks
selectedStocks = st.selectBox("Select Stock", tickers)

# Slider to select prediction range
years = st.slider("Years of prediction:", 1, 4)
period = years*365