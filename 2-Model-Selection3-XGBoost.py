# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 04:45:59 2024

@author: Yunus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

# Data Access
df = pd.read_excel('C:/Users/yunus/Downloads/Data Deneme/Custom-Dataset.xlsx')

# Separate the target variable
target = "BTC_Adj Close"
y = df[[target]]
X = df.drop(columns=[target])

# Normalize the features and the target variable separately
sc_X = MinMaxScaler(feature_range=(0, 1))
sc_y = MinMaxScaler(feature_range=(0, 1))

X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)

# Convert y_train and y_test to 1D arrays for XGBoost
y_train = y_train.flatten()
y_test = y_test.flatten()

# Define XGBoost model
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Fit the model
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

# Predict on test data
predictions = model.predict(X_test)

# Inverse transform the predictions and actual values to original scale
y_test_inv = sc_y.inverse_transform(y_test.reshape(-1, 1))
predictions_inv = sc_y.inverse_transform(predictions.reshape(-1, 1))

# Evaluate the model
mse = mean_squared_error(y_test_inv, predictions_inv)
r2 = r2_score(y_test_inv, predictions_inv)
mae = mean_absolute_error(y_test_inv, predictions_inv)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
print(f'Mean Absolute Error: {mae}')

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Actual')
plt.plot(predictions_inv, label='Predicted')
plt.title('Actual vs Predicted')
plt.xlabel('Time')
plt.ylabel('BTC Adj Close')
plt.legend()
plt.show()
