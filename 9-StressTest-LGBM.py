# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 02:38:21 2024

@author: Yunus
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load and prepare data
df = pd.read_excel('C:/Users/yunus/Downloads/Data Deneme/Custom-Dataset.xlsx')

# Clean feature names to remove special characters
df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)

# print(df.head())

X = df.drop(columns=['BTC_AdjClose']) 
y = df['BTC_AdjClose']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = lgb.LGBMRegressor()
model.fit(X_train, y_train)

# Simulate a market crash by reducing price-related features by 30%
X_test_crash = X_test.copy()
price_features = ['BTC_Open', 'BTC_High', 'BTC_Low']  # Adjust as per dataset
X_test_crash[price_features] *= 0.7

# Predict using the crash scenario
predictions_crash = model.predict(X_test_crash)

# Evaluate the performance
mse_crash = mean_squared_error(y_test, predictions_crash)
r2_crash = r2_score(y_test, predictions_crash)
mae_crash = mean_absolute_error(y_test, predictions_crash)

print(f'Mean Squared Error under Crash Scenario: {mse_crash}')
print(f'R^2 Score under Crash Scenario: {r2_crash}')
print(f'Mean Absolute Error under Crash Scenario: {mae_crash}')

# Compare with baseline performance
predictions_baseline = model.predict(X_test)
mse_baseline = mean_squared_error(y_test, predictions_baseline)
r2_baseline = r2_score(y_test, predictions_baseline)
mae_baseline = mean_absolute_error(y_test, predictions_baseline)

print(f'Baseline Mean Squared Error: {mse_baseline}')
print(f'Baseline R^2 Score: {r2_baseline}')
print(f'Baseline Mean Absolute Error: {mae_baseline}')

