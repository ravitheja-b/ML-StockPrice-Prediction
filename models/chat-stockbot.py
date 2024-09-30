from forecast import stock_prediction_LSTM, stock_prediction_arima, stock_prediction_decisiontree, \
    stock_prediction_linearregression, stock_prediction_randomforest
from io import StringIO
import yfinance as yf
import altair as alt
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import openai
import os
from dotenv import load_dotenv, find_dotenv
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
import streamlit as st
import math
from yahoo_fin import stock_info
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from openai import OpenAI

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")


def fetch_stock_price(ticker_new):
    stock = yf.Ticker(ticker_new)
    hist = stock.history(period="1d")
    return hist['Close'].iloc[-1]


def fetch_stock_data(tickers, period='5y'):
    stock_data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        stock_data[ticker] = hist
    return stock_data


def view_price_trend_and_chat(csv_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Convert the date column to datetime type
    df['Date'] = pd.to_datetime(df['Date'])

    # Create a line chart using Altair
    chart = alt.Chart(df).mark_line().encode(
        x='Date',
        y='Close',
        tooltip=['Date', 'Close']
    ).properties(
        width=800,
        height=500
    ).interactive()

    # Display the chart using Streamlit
    st.write("View Price Trend:")
    st.write(chart)

    # Create Pandas DataFrame agent for chatting
    agent_chat = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )

    # Display DataFrame for chatting
    st.write("Chat with CSV Data:")
    st.write("Start chatting with the CSV data:")
    user_input = st.text_input("You:", "")

    if st.button("Send"):
        user_input_lower = user_input.lower()
        if "prediction" in user_input_lower or "predict" in user_input_lower:
            with st.spinner("Predicting stock price..."):
                prediction = stock_prediction_arima()
                ticker_new = "HDFCBANK.NS"
                price = fetch_stock_price(ticker_new)
                st.write(f"The :blue[current] price of {ticker_new} is: INR :green[{price:.2f}]")
        elif "trend" in user_input_lower or "graph" in user_input_lower:
            st.image("ARIMA-prediction.png")
        else:
            response = agent_chat(user_input)
            st.write("Chatbot:", response)



def get_response(user_prompt):
    response = OpenAI().chat.completions.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt},
        ],
        stream=True
    )

    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield content


# Function to get a streaming response from OpenAI


def stock_price_checker_app():
    stock_list = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'FB', 'NFLX', 'DMART.NS', "HDFC.NS", "SBIN.NS",
                  "HDFCBANK.NS"]

    def get_live_stock_price(stock_symbol):
        try:
            live_price = stock_info.get_live_price(stock_symbol)
            return f"The current price of {stock_symbol} is INR {live_price:.2f}"
        except Exception as e:
            return f"Error fetching price for {stock_symbol}: {str(e)}"

    # Streamlit app setup
    st.sidebar.title("Live Stock Price Checker")

    # Get user input
    user_input = st.sidebar.text_input("Enter stock:", "")

    if st.sidebar.button("Retrieve"):
        found_stock = False
        # Check if the user input contains any stock symbols from the list
        for stock in stock_list:
            if stock in user_input:
                found_stock = True
                live_price_message = get_live_stock_price(stock)
                st.sidebar.success(live_price_message)

        if not found_stock:
            st.sidebar.info("No valid stock symbols found in your input.")

    # Optional: Display valid stock symbols
    # st.sidebar.write("Valid stock symbols are:", ", ".join(stock_list))


def main():
    stock_price_checker_app()
    st.sidebar.header("Stock Dataset Downloader")
    tickers_input = st.sidebar.text_area("Enter stock tickers (comma separated)", "AAPL, MSFT, GOOG")
    period = st.sidebar.selectbox("Select period", ["1y", "5y", "10y"], index=1)

    if st.sidebar.button("Download Data"):
        tickers = [ticker.strip() for ticker in tickers_input.split(",")]
        stock_data = fetch_stock_data(tickers, period)

        for ticker, data in stock_data.items():
            st.write(f"### {ticker} Stock Data")
            st.dataframe(data)
            csv = data.to_csv().encode('utf-8')
            st.download_button(
                label=f"Download {ticker} data as CSV",
                data=csv,
                file_name=f'{ticker}_{period}.csv',
                mime='text/csv',
            )

    st.title("Stock Price Prediction App")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        st.write("CSV file uploaded successfully!")
        view_price_trend_and_chat(uploaded_file)

    st.sidebar.title("Ask AI")
    user_prompt = st.sidebar.text_input("Chat with AI :")
    if user_prompt and st.sidebar.button("Get response"):
        st.sidebar.write_stream(get_response(user_prompt))


if __name__ == "__main__":
    main()
