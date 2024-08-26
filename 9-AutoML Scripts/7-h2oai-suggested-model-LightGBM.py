# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 17:05:14 2024

@author: Yunus
"""

import pandas as pd
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

# Predict and evaluate
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
print(f'Mean Absolute Error: {mae}')

# Optional: Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title('Actual vs Predicted')
plt.xlabel('Time')
plt.ylabel('BTC Adj Close')
plt.legend()
plt.show()
