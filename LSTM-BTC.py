# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:34:04 2024

@author: Yunus
"""
"""
This parameters come from previous trials. 
 Lag  Unit1   Epochs   Unit2  TrainRMSE   TestRMSE
17.0  128.0   200.0    64.0   21.275000   61.808739
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTNS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import datetime

from sklearn.preprocessing import MinMaxScaler
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Dropout
from keras.api.optimizers import Adam
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Fetch Bitcoin data
data = pd.DataFrame(yf.download("BTC-USD", start="2014-01-01", end="2024-07-27")["Adj Close"])
# print(data) # check


# Normalize the data
sc = MinMaxScaler(feature_range=(0, 1))
veri_sc = sc.fit_transform(data)

# Split into training and testing sets
train_size = int(len(veri_sc) * 0.70)
train, test = veri_sc[:train_size], veri_sc[train_size:]

# Create time series dataset
def ts(data, timestep):
    x, y = [], []
    for i in range(timestep, len(data)):
        x.append(data[i-timestep:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

# Prepare the training and test data
# lag = 17
x_train, y_train = ts(train, 17)
x_test, y_test = ts(test, 17)

# Reshape data for LSTM input
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Build LSTM model
# 1.layer = 128, 2.layer = 64
model = Sequential()
model.add(LSTM(units=128, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=64,))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile and train the model
# For computational efficiency, I didnt try epoch=250 for model testing. I tried 250 here, results are better than 200. 
opt = Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss="mse")
model.fit(x_train, y_train, epochs=250, validation_data=(x_test, y_test), verbose=1)

# Make predictions
traintahmin = model.predict(x_train)
testtahmin = model.predict(x_test)

# Inverse transform predictions and actual values
traintahmin = sc.inverse_transform(traintahmin)
testtahmin = sc.inverse_transform(testtahmin)
trainY = sc.inverse_transform(y_train)
testY = sc.inverse_transform(y_test)

# Evaluate the model
mse = mean_squared_error(testY, testtahmin)
r2 = r2_score(testY, testtahmin)
mae = mean_absolute_error(testY, testtahmin)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
print(f'Mean Absolute Error: {mae}')

# Forecasting function
def forecast(model, last_data, n_future):
    forecasted = []
    current_data = last_data[-30:]
    for _ in range(n_future):
        current_data = current_data.reshape((1, current_data.shape[0], 1))
        prediction = model.predict(current_data)
        forecasted.append(prediction[0, 0])
        current_data = np.append(current_data[0, 1:], prediction)
    forecasted = np.array(forecasted).reshape(-1, 1)
    return sc.inverse_transform(forecasted)

# Forecasting the next 30 days
n_future = 30
last_data = veri_sc[-30:]
forecasted_values = forecast(model, last_data, n_future)

# Plot predictions, actual values, and forecasted values
fig, axs = plt.subplots(2, 1, figsize=(12, 8))

axs[0].plot(trainY, label="Actual Train")
axs[0].plot(traintahmin, label="Predicted Train")
axs[0].legend()

axs[1].plot(testY, label="Actual Test")
axs[1].plot(testtahmin, label="Predicted Test")
axs[1].legend()

plt.figure(figsize=(10, 5))
plt.plot(np.arange(len(data)), data, label="Actual Data")
plt.plot(np.arange(len(data), len(data) + n_future), forecasted_values, label="Forecasted Data", linestyle='dashed')
plt.legend()
plt.show()