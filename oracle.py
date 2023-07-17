import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import yfinance as yf

def stock_oracle(ticker):
    # Load data from Yahoo Finance
    company = ticker
    start = dt.datetime(2020, 7, 15)
    end = dt.datetime(2021, 7, 15)
    data = yf.download(company, start=start, end=end)

    # Preprocessing
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    prediction_days = 60  # Number of days to look back for prediction
    x_train, y_train = [], []

    # Create training data using sliding window
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    # Convert training data to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Predict the next closing price

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    # Accuracy on existing data

    # Load test data from Yahoo Finance
    test_start = dt.datetime(2021, 7, 15)
    test_end = dt.datetime.now()
    test_data = yf.download(company, start=test_start, end=test_end)
    actual = test_data['Close'].values
    total = pd.concat((data['Close'], test_data['Close']))
    model_inputs = total[len(total) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    # Generate predictions on test data
    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])
    x_test = np.array(x_test)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    predict_price = model.predict(x_test)

    # Inverse scaling to get actual predicted prices
    predict_price = scaler.inverse_transform(predict_price)

    # Plot actual and predicted prices
    plt.plot(actual, color="red", label=f"Actual {company} price")
    plt.plot(predict_price, color="green", label=f"Predicted {company} price")
    plt.title(f"{company} Share Price")
    plt.xlabel("Time")
    plt.ylabel(f"{company} Share Price")
    plt.legend()
    plt.show()

    # Predict the next day's price
    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days: len(model_inputs) + 1, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    print(f"Prediction: {prediction}")
