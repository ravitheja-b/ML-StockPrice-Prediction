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
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from tensorflow.keras.metrics import MeanAbsoluteError
import tensorflow.keras.backend as K

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

uploaded_file = None


def stock_prediction_LSTM():
    # Read the CSV file
    try:
        df = pd.read_csv('HDFCBANK.NS_5y.csv')
        # st.write(df.head())
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty or has no columns to parse.")
        return

    # Convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_convert(None)

    # Prepare data for the LSTM model
    df.set_index('Date', inplace=True)
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = min_max_scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    # Split into train and test sets
    train_size = int(len(dataset) * 0.7)
    train, test = dataset[:train_size], dataset[train_size:]

    # Convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=15):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    x_train, y_train = create_dataset(train, look_back=15)
    x_test, y_test = create_dataset(test, look_back=15)

    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

    model = Sequential()
    model.add(LSTM(20, input_shape=(1, 15)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2)

    trainPredict = model.predict(x_train)
    testPredict = model.predict(x_test)
    # Invert predictions
    trainPredict = min_max_scaler.inverse_transform(trainPredict)
    trainY = min_max_scaler.inverse_transform([y_train])
    testPredict = min_max_scaler.inverse_transform(testPredict)
    testY = min_max_scaler.inverse_transform([y_test])

    # Calculate performance metrics
    trainMSE = mean_squared_error(trainY[0], trainPredict[:, 0])
    trainRMSE = np.sqrt(trainMSE)
    testMSE = mean_squared_error(testY[0], testPredict[:, 0])
    testRMSE = np.sqrt(testMSE)
    mae = mean_absolute_error(testY[0], testPredict[:, 0])
    r2 = r2_score(testY[0], testPredict[:, 0])
    mape = np.mean(np.abs((testY[0] - testPredict[:, 0]) / testY[0])) * 100
    std_dev = np.std(testY[0] - testPredict[:, 0])
    confidence_interval = 1.96 * std_dev  # 95% confidence interval

    st.write(f'Train Mean Squared Error (MSE): {trainMSE:.2f}')
    st.write(f'Train Root Mean Squared Error (RMSE): {trainRMSE:.2f}')
    st.write(f'Test Mean Squared Error (MSE): {testMSE:.2f}')
    st.write(f'Test Root Mean Squared Error (RMSE): {testRMSE:.2f}')
    st.write(f'Mean Absolute Error (MAE): {mae:.2f}')
    st.write(f'R^2 Score: {r2:.2f}')
    st.write(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')
    st.write(f'95% Confidence Interval: ±{confidence_interval:.2f}')

    # Plot baseline and predictions
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[15:len(trainPredict) + 15, :] = trainPredict
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (15 * 2) + 1:len(dataset) - 1, :] = testPredict

    plt.figure(figsize=(14, 7))
    plt.plot(min_max_scaler.inverse_transform(dataset), label='Actual Prices')
    plt.plot(trainPredictPlot, label='Train Predictions')
    plt.plot(testPredictPlot, label='Test Predictions')
    plt.title('HDFC Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    st.pyplot()

    # Load additional data for prediction
    COMPANY = 'HDFC'
    START_DATE = dt.datetime(2019, 5, 27)
    END_DATE = dt.datetime(2020, 12, 30)
    START_DATE_TEST = END_DATE

    def load_data(company, start, end):
        dataframe = df.copy()
        dataframe = dataframe.loc[(dataframe.index > start) & (dataframe.index < end), :]
        return dataframe

    data = load_data(company=COMPANY, start=START_DATE, end=END_DATE)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    prediction_days = 30
    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    checkpointer = ModelCheckpoint(filepath='weights_best.keras', verbose=2, save_best_only=True)

    model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=[checkpointer])

    test_data = load_data(company=COMPANY, start=START_DATE_TEST, end=dt.datetime.now())
    actual_prices = test_data['Close'].values
    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    x_test = []

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    plt.figure(figsize=(14, 7))
    plt.plot(actual_prices, color='black', label=f"Actual {COMPANY} price")
    plt.plot(predicted_prices, color='green', label=f"Predicted {COMPANY} price")
    plt.title(f"{COMPANY} Share Price")
    plt.xlabel("Time")
    plt.ylabel(f"{COMPANY} Share Price")
    plt.legend()
    st.pyplot()

    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs + 1), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)

    st.write("\nModel Summary and Explanation:")
    st.write(
        f"Our LSTM model has a Root Mean Squared Error (RMSE) of {testRMSE:.2f} on the test set, indicating the average magnitude of error in predicting the stock prices.")
    st.write(
        f"The Mean Squared Error (MSE) is {testMSE:.2f}, which represents the average of the squares of the errors or deviations.")
    st.write(
        f"The Mean Absolute Error (MAE) is {mae:.2f}, showing the average error magnitude. The R² score is {r2:.2f}, explaining how well our predictions approximate the actual values.")
    st.write(
        f"The Mean Absolute Percentage Error (MAPE) is {mape:.2f}%, indicating the average percentage error between our predictions and the actual values.")
    st.write(
        f"The 95% confidence interval is ±{confidence_interval:.2f}, providing a range within which the actual stock prices are expected to fall.")

    st.write(
        "\nThese metrics demonstrate the reliability of our model in predicting stock prices. Visualizing the real vs. predicted prices shows that our model captures the general trend, boosting confidence in its predictive capability.")

    # Return and st.write the latest predicted stock value
    predicted_value = prediction[0][0]
    st.write(f"The latest predicted stock value is: :green[{predicted_value:.2f}]")

    return predicted_value


def stock_prediction_arima():
    # Step 1: Read CSV data and load it as a pandas DataFrame
    def read_csv_data(file_path):
        data = pd.read_csv(file_path, parse_dates=True, index_col='Date')
        return data

    # Step 2: Build ARIMA model for stock price prediction
    def build_arima_model(data, order=(5, 1, 0)):
        model = ARIMA(data, order=order)
        model_fit = model.fit()
        return model_fit

    # Step 3: Evaluate model reliability
    def evaluate_model(model_fit, data):
        residuals = model_fit.resid
        adf_test = adfuller(residuals)
        p_value = adf_test[1]

        if p_value < 0.05:
            #st.write("The residuals are stationary. The model is reliable.")
            st.write("")
        else:
            st.write("The residuals are not stationary. The model may not be reliable.")

        # Calculate RMSE, MAE, MAPE, and R-squared
        predicted = model_fit.predict(start=1, end=len(data))
        rmse = np.sqrt(mean_squared_error(data[1:], predicted[1:]))
        mae = mean_absolute_error(data[1:], predicted[1:])
        mape = np.mean(np.abs((data[1:] - predicted[1:]) / data[1:])) * 100
        r2 = r2_score(data[1:], predicted[1:])

        # st.write(f"Root Mean Squared Error (RMSE): {rmse}")
        # st.write(f"Mean Absolute Error (MAE): {mae}")
        # st.write(f"Mean Absolute Percentage Error (MAPE): {mape}%")
        # st.write(f"R-squared (R²): {r2}")

    # Step 4: Plot predicted stock price
    def plot_predicted_stock_price(model_fit, data, steps=30):
        forecast = model_fit.get_forecast(steps=steps)
        forecast_index = pd.date_range(start=data.index[-1], periods=steps + 1, closed='right')
        forecast_mean = forecast.predicted_mean
        forecast_conf_int = forecast.conf_int()

        plt.figure(figsize=(12, 6))
        plt.plot(data, label='Observed')
        plt.plot(forecast_index, forecast_mean, color='red', label='Forecast')
        plt.fill_between(forecast_index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='pink')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.title('Stock Price Prediction using ARIMA')
        plt.legend()
        #st.pyplot()
        plt.savefig("ARIMA-prediction.png")

    # Step 5: Model Summary and Explanation
    # def model_summary(model_fit):
    #     st.write(model_fit.summary())
    #     st.write("\nModel Explanation:")
    #     st.write(
    #         "The ARIMA model is composed of three parts: AR (AutoRegressive), I (Integrated), and MA (Moving Average).")
    #     st.write("AR part involves regressing the variable on its own lagged (prior) values.")
    #     st.write("I part involves differencing the variable to make it stationary.")
    #     st.write(
    #         "MA part involves modeling the error term as a linear combination of error terms occurring contemporaneously and at various times in the past.")

    # Step 6: Return and st.write the predicted stock value with confidence level
    def predict_stock_value(model_fit, steps=1):
        forecast = model_fit.get_forecast(steps=steps)
        predicted_value = forecast.predicted_mean.iloc[-1]
        conf_int = forecast.conf_int().iloc[-1]
        confidence_level = 0.95
        st.write(f"Predicted stock value: :orange[{predicted_value}]")
        st.write(f"Confidence Interval: {conf_int.values}")
        st.write(f"Confidence Level: {confidence_level * 100}%")
        #return predicted_value, conf_int

    # Main function to run all steps
    def main(file_path):
        data = read_csv_data(file_path)

        # Select the 'Close' column for univariate time series
        data_close = data['Close']

        model_fit = build_arima_model(data_close)
        evaluate_model(model_fit, data_close)
        plot_predicted_stock_price(model_fit, data_close)
        #model_summary(model_fit)
        predict_stock_value(model_fit)

    # Example usage
    file_path = 'C:\\Users\\bulip\\PycharmProjects\\Streamlit-bot\\HDFCBANK.NS_5y_new.csv'
    main(file_path)


def stock_prediction_decisiontree():
    # Load and preview the data
    hdfc = pd.read_csv('HDFCBANK.NS_5y.csv)

    # # Plot the closing price
    # sns.set()
    # plt.figure(figsize=(10, 4))
    # plt.title("HDFC's Stock Price")
    # plt.xlabel("Days")
    # plt.ylabel("Close Price INR")
    # plt.plot(hdfc["Close"])
    # # plt.show()
    # st.pyplot()

    # Prepare data
    hdfc = hdfc[["Close"]]
    futureDays = 25
    hdfc["Prediction"] = hdfc[["Close"]].shift(-futureDays)
    # st.write(hdfc.head())
    # st.write(hdfc.tail())

    # Features and labels
    x = np.array(hdfc.drop(["Prediction"], axis=1))[:-futureDays]
    y = np.array(hdfc["Prediction"])[:-futureDays]

    # Split into training and testing sets
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=42)

    # Decision Tree Regressor model
    tree = DecisionTreeRegressor().fit(xtrain, ytrain)

    # Future data
    xfuture = hdfc.drop(["Prediction"], axis=1)[:-futureDays]
    xfuture = xfuture.tail(futureDays)
    xfuture = np.array(xfuture)
    # st.write(xfuture)

    # Predictions
    treePrediction = tree.predict(xfuture)

    # st.write("Decision Tree prediction =", treePrediction)

    # Calculate performance metrics for Tree  model
    predict_y = tree.predict(xtest)
    prediction_score = tree.score(xtest, ytest)
    mse = mean_squared_error(ytest, predict_y)
    mae = np.mean(np.abs(predict_y - ytest))
    r2 = r2_score(ytest, predict_y)
    mape = np.mean(np.abs((ytest - predict_y) / ytest)) * 100

    st.write('\nModel Evaluation Metrics:')
    st.write(f'Prediction Score (R^2): {prediction_score:.2f}')
    st.write(f'Mean Squared Error (MSE): {mse:.2f}')
    st.write(f'Mean Absolute Error (MAE): {mae:.2f}')
    st.write(f'R^2 Score: {r2:.2f}')
    st.write(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')

    # Model summary and explanation
    st.write("\nModel Summary and Explanation:")
    st.write("The Decision model has been trained to predict the stock prices based on the given features.")
    st.write(f"R^2 Score of {r2:.2f} indicates how well the predictions approximate the actual data.")
    st.write(
        f"The Mean Squared Error (MSE) is {mse:.2f}, which shows the average squared difference between the predicted and actual values.")
    st.write(f"The Mean Absolute Error (MAE) of {mae:.2f} indicates the average magnitude of errors in predictions.")
    st.write(
        f"Mean Absolute Percentage Error (MAPE) of {mape:.2f}% represents the average percentage error between predicted and actual values.")

    # Calculate confidence intervals for Tree predictions
    n = len(xtest)
    p = x.shape[1]
    alpha = 0.05
    t_value = stats.t.ppf(1 - alpha / 2, n - p - 1)
    se = np.sqrt(np.sum((ytest - predict_y) ** 2) / (n - p - 1))
    confidence_interval = t_value * se
    lower_bound = treePrediction - confidence_interval
    upper_bound = treePrediction + confidence_interval

    # Plot the predictions
    predictions = treePrediction
    valid = hdfc[x.shape[0]:]
    valid["Predictions"] = predictions

    plt.figure(figsize=(10, 6))
    plt.title("HDFC's Stock Price Prediction Model (Decision Tree Regressor Model)")
    plt.xlabel("Days")
    plt.ylabel("Close Price INR")
    plt.plot(hdfc["Close"], label='Original')
    plt.plot(valid["Close"], label='Valid')
    plt.plot(valid["Predictions"], label='Predictions')
    plt.fill_between(range(len(valid)), lower_bound, upper_bound, color='gray', alpha=0.2, label='Confidence Interval')
    plt.legend()
    # plt.show()
    st.pyplot()

    # Return and st.write the latest predicted stock value
    latest_predicted_value = treePrediction[-1]
    latest_lower_bound = lower_bound[-1]
    latest_upper_bound = upper_bound[-1]
    st.write(f"The latest predicted stock value is: :green[{latest_predicted_value:.2f}]")
    st.write(f"With a 95% confidence interval of: ({latest_lower_bound:.2f}, {latest_upper_bound:.2f})")

    # Return the predicted stock value
    def stock_decisiontree():
        return latest_predicted_value

    predicted_value = stock_decisiontree()
    st.write(f"Predicted Stock Value: :green[{predicted_value:.2f}]")


def stock_prediction_linearregression():
    # Import the data
    data = pd.read_csv("C:\\Users\\bulip\\PycharmProjects\\Streamlit-bot\\HDFCBANK.NS_5y.csv")

    # Drop the Date column as it's not used for prediction
    data = data.drop('Date', axis=1)

    # Split into train and test data
    data_X = data.loc[:, data.columns != 'Close']
    data_Y = data['Close']
    train_X, test_X, train_y, test_y = train_test_split(data_X, data_Y, test_size=0.25, random_state=42)

    # st.write('\n\nTraining Set')
    # st.write(train_X.head())
    # st.write(train_y.head())

    # Creating the Regressor
    regressor = LinearRegression()
    regressor.fit(train_X, train_y)

    # Make Predictions and Evaluate the results
    predict_y = regressor.predict(test_X)

    # Calculate performance metrics
    prediction_score = regressor.score(test_X, test_y)
    mse = mean_squared_error(test_y, predict_y)
    mae = mean_absolute_error(test_y, predict_y)
    r2 = r2_score(test_y, predict_y)
    mape = np.mean(np.abs((test_y - predict_y) / test_y)) * 100

    st.write('\nModel Evaluation Metrics:')
    st.write(f'Prediction Score (R^2): {prediction_score:.2f}')
    st.write(f'Mean Squared Error (MSE): {mse:.2f}')
    st.write(f'Mean Absolute Error (MAE): {mae:.2f}')
    st.write(f'R^2 Score: {r2:.2f}')
    st.write(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')

    # Model summary and explanation
    st.write("\nModel Summary and Explanation:")
    st.write("The Linear Regression model has been trained to predict the stock prices based on the given features.")
    st.write(f"R^2 Score of {r2:.2f} indicates how well the predictions approximate the actual data.")
    st.write(
        f"The Mean Squared Error (MSE) is {mse:.2f}, which shows the average squared difference between the predicted and actual values.")
    st.write(f"The Mean Absolute Error (MAE) of {mae:.2f} indicates the average magnitude of errors in predictions.")
    st.write(
        f"Mean Absolute Percentage Error (MAPE) of {mape:.2f}% represents the average percentage error between predicted and actual values.")

    # Calculate the confidence intervals for predictions
    n = len(test_X)
    p = test_X.shape[1]
    alpha = 0.05
    t_value = stats.t.ppf(1 - alpha / 2, n - p - 1)

    # Calculate the standard error of the predictions
    se = np.sqrt(np.sum((test_y - predict_y) ** 2) / (n - p - 1))
    confidence_interval = t_value * se
    lower_bound = predict_y - confidence_interval
    upper_bound = predict_y + confidence_interval

    # Plot the predicted and the expected values with confidence intervals
    plt.figure(figsize=(14, 7))
    plt.grid()
    plt.xlabel('Sample Index')
    plt.ylabel('Close Price ($)')
    plt.title('HDFC Bank Stock Prediction using Linear Regression')
    plt.plot(test_y.values, label='Actual Prices', color='blue')
    plt.plot(predict_y, label='Predicted Prices', color='orange')
    plt.fill_between(range(len(predict_y)), lower_bound, upper_bound, color='gray', alpha=0.2,
                     label='Confidence Interval')
    plt.legend()
    plt.savefig('LRPlot.png')
    # plt.show
    st.pyplot()

    # Return and st.write the latest predicted stock value
    latest_predicted_value = predict_y[-1]
    latest_lower_bound = lower_bound[-1]
    latest_upper_bound = upper_bound[-1]
    st.write(f"The latest predicted stock value is: :green[{latest_predicted_value:.2f}]")
    st.write(f"With a 95% confidence interval of: ({latest_lower_bound:.2f}, {latest_upper_bound:.2f})")


def stock_prediction_randomforest():
    # Step 1: Read CSV data and load it as a pandas DataFrame
    def read_csv_data(file_path):
        data = pd.read_csv(file_path, parse_dates=True, index_col='Date')
        return data

    # Step 2: Build a Random Forest model for stock price prediction
    def build_random_forest_model(data):
        # Assuming 'Close' is the target variable and other columns are features
        X = data.drop(columns=['Close'])
        y = data['Close']

        # Splitting data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Building the Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        return model, X_train, X_test, y_train, y_test

    # Step 3: Evaluate the model to determine its reliability
    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        r2 = r2_score(y_test, y_pred)

        st.write(f"Mean Squared Error: {mse}")
        st.write(f"Root Mean Squared Error: {rmse}")
        st.write(f"Mean Absolute Error: {mae}")
        st.write(f"Mean Absolute Percentage Error: {mape}%")
        st.write(f"R-squared: {r2}")

        return y_pred, mse, rmse, mae, mape, r2

    # Step 4: Plot the predicted stock price using the Random Forest model
    def plot_predicted_stock_price(y_test, y_pred):
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.values, label='Actual')
        plt.plot(y_pred, label='Predicted', color='red')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.title('Stock Price Prediction using Random Forest')
        plt.legend()
        # plt.show()
        st.pyplot()

    # Step 5: Model Summary and Explanation
    def model_summary(model):
        st.write("Model Summary:")
        st.write("The Random Forest model is an ensemble learning method for regression.")
        st.write(
            "It operates by constructing multiple decision trees during training and outputting the mean prediction of the individual trees.")
        st.write("This method helps to improve the predictive accuracy and control overfitting.")

    # Step 6: Return and print the predicted stock value with confidence level
    def predict_stock_value(model, X_test, y_test):
        predicted_value = model.predict(X_test[-1:])
        residuals = y_test - model.predict(X_test)
        std_dev = np.std(residuals)

        conf_interval = 1.96 * std_dev  # for approximately 95% confidence interval
        confidence_level = 95  # We are using the 95% confidence interval

        st.write(f"Predicted stock value: :green[{predicted_value[0]}]")
        st.write(f"Confidence Interval: ±{conf_interval}")
        st.write(f"Confidence Level: {confidence_level}%")

        return predicted_value[0], conf_interval

    # Main function to run all steps
    def main(file_path):
        data = read_csv_data(file_path)

        model, X_train, X_test, y_train, y_test = build_random_forest_model(data)
        y_pred, mse, rmse, mae, mape, r2 = evaluate_model(model, X_test, y_test)
        plot_predicted_stock_price(y_test, y_pred)
        model_summary(model)
        predict_stock_value(model, X_test, y_test)

    # Example usage
    file_path = 'HDFCBANK.NS_5y.csv'

    main(file_path)

