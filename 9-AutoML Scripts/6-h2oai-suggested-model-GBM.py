# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 19:27:54 2024

@author: Yunus
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_excel('C:/Users/yunus/Downloads/Data Deneme/Custom-Dataset.xlsx')
X = df.drop(columns=['BTC_Adj Close'])
y = df['BTC_Adj Close']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Gradient Boosting Regressor model
model = GradientBoostingRegressor(
    n_estimators=100,  # Number of boosting stages to be run
    learning_rate=0.1,  # Step size shrinkage used in update to prevent overfitting
    max_depth=3,  # Maximum depth of the individual estimators
    min_samples_split=2,  # Minimum number of samples required to split an internal node
    min_samples_leaf=1,  # Minimum number of samples required to be at a leaf node
    subsample=1.0,  # Fraction of samples used for fitting the individual base learners
    random_state=42  # To ensure reproducibility
)
model.fit(X_train, y_train)

# Predict on the test data
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
print(f'Mean Absolute Error: {mae}')

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(y_test.reset_index(drop=True), label='Actual')
plt.plot(predictions, label='Predicted')
plt.title('Actual vs Predicted')
plt.xlabel('Time')
plt.ylabel('BTC Adj Close')
plt.legend()
plt.show()
