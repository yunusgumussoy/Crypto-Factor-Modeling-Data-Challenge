# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 14:52:55 2024

@author: Yunus
"""

# pip install yfinance pandas scikit-learn xgboost matplotlib

import yfinance as yf
import pandas as pd
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Download Bitcoin data
btc_data = yf.download("BTC-USD", start="2015-01-01", end="2024-07-27")

# Prepare the data
btc_data['Date'] = btc_data.index
btc_data = btc_data[['Date', 'Close']]
btc_data = btc_data.reset_index(drop=True)

# Feature Engineering
btc_data['day'] = btc_data['Date'].dt.day
btc_data['month'] = btc_data['Date'].dt.month
btc_data['year'] = btc_data['Date'].dt.year

# Lag features
for lag in range(1, 8):
    btc_data[f'lag_{lag}'] = btc_data['Close'].shift(lag)

# Drop missing values
btc_data = btc_data.dropna()

# Prepare X and y
X = btc_data[['day', 'month', 'year'] + [f'lag_{lag}' for lag in range(1, 8)]]
y = btc_data['Close']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01, max_depth=6)

# Train the model
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse}")

# Prepare data for the next 30 days forecast
last_date = btc_data['Date'].iloc[-1]
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 31)]

future_df = pd.DataFrame(future_dates, columns=['Date'])
future_df['day'] = future_df['Date'].dt.day
future_df['month'] = future_df['Date'].dt.month
future_df['year'] = future_df['Date'].dt.year

# Initialize lag features with the last known prices
for lag in range(1, 8):
    future_df[f'lag_{lag}'] = btc_data['Close'].iloc[-lag]

# Make predictions for each day in the future
future_predictions = []
for i in range(len(future_df)):
    # Scale the features
    future_features = future_df[['day', 'month', 'year'] + [f'lag_{lag}' for lag in range(1, 8)]].iloc[i].values.reshape(1, -1)
    future_features_scaled = scaler.transform(future_features)
    
    # Predict the next day price
    next_price = model.predict(future_features_scaled)[0]
    future_predictions.append(next_price)
    
    # Update lag features
    for lag in range(1, 8):
        if i+1 < lag:
            future_df.at[i+1, f'lag_{lag}'] = btc_data['Close'].iloc[-lag + i + 1]
        else:
            future_df.at[i+1, f'lag_{lag}'] = future_predictions[i - lag + 1]

# Print future predictions
for i, price in enumerate(future_predictions):
    print(f"Day {i+1}: {price}")

# Plot actual vs predicted
plt.figure(figsize=(14, 7))
plt.plot(btc_data['Date'], btc_data['Close'], label='Actual Prices')
plt.plot(btc_data['Date'].iloc[-len(y_test):], y_pred, label='Predicted Prices')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Bitcoin Price')
plt.title('Bitcoin Price Prediction using XGBoost')
plt.show()

# Plot future predictions
plt.figure(figsize=(14, 7))
plt.plot(future_dates, future_predictions, label='Future Predictions')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Bitcoin Price')
plt.title('Future Bitcoin Price Prediction using XGBoost')
plt.show()

