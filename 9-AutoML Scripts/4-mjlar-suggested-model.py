# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:28:58 2024

@author: Yunus
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

# Load your dataset
df = pd.read_excel('C:/Users/yunus/Downloads/Data Deneme/Custom-Dataset.xlsx')

# Separate the target variable
target = "BTC_Adj Close"
y = df[target].values  # Ensure y is a 1D array
X = df.drop(columns=[target]).values  # Ensure X is a 2D array

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the base models for stacking
base_models = [
    ('tree', DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, min_samples_split=17)),
    ('gbm', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)),
    ('svr', SVR(kernel='linear', C=1.0))
]

# Define the final estimator
final_estimator = RidgeCV(alphas=[0.1, 1.0, 10.0])

# Create the stacking regressor
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=final_estimator
)

# Create the pipeline with a scaler and stacking model
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Apply StandardScaler to all features
    ('stacking', stacking_model)  # Apply StackingRegressor
])

# Fit the model
pipeline.fit(X_train, y_train)

# Predict on test data
predictions = pipeline.predict(X_test)

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
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title('Actual vs Predicted')
plt.xlabel('Time')
plt.ylabel('BTC Adj Close')
plt.legend()
plt.show()
