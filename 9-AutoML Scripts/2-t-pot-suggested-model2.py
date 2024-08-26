# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:36:43 2024

@author: Yunus
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Custom Estimator to handle ElasticNetCV and RidgeCV
class ElasticNetWithRidgeCV(BaseEstimator, TransformerMixin):
    def __init__(self, alpha_ridge=1.0, l1_ratio=1.0):
        self.alpha_ridge = alpha_ridge
        self.l1_ratio = l1_ratio

    def fit(self, X, y):
        self.elasticnet = ElasticNetCV(l1_ratio=self.l1_ratio, tol=0.001)
        self.elasticnet.fit(X, y)
        self.ridge = RidgeCV(alphas=[self.alpha_ridge])
        self.ridge.fit(X, y)
        return self

    def predict(self, X):
        return self.ridge.predict(X)

# Load your dataset
df = pd.read_excel('C:/Users/yunus/Downloads/Data Deneme/Custom-Dataset.xlsx')

# Separate the target variable
target = "BTC_Adj Close"
y = df[target].values  # Ensure y is a 1D array
X = df.drop(columns=[target]).values  # Ensure X is a 2D array

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the model pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Apply MinMaxScaler to all features
    ('model', ElasticNetWithRidgeCV(alpha_ridge=1.0, l1_ratio=1.0))  # Custom model with ElasticNetCV and RidgeCV
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
