# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 05:57:52 2024

@author: Yunus
"""

# pip install tpot

# Import necessary libraries
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Load your dataset
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

# Initialize TPOT
tpot = TPOTRegressor(verbosity=2, generations=5, population_size=50, random_state=42)

# Fit TPOT to the training data
tpot.fit(X_train, y_train.ravel())  # Flatten y_train to avoid shape issues

# Evaluate the model on the test set
print(f"Test R^2 Score: {tpot.score(X_test, y_test)}")

# Export the best model pipeline
tpot.export('best_model_pipeline.py')

# Predict on the test data
predictions = tpot.predict(X_test)

# Inverse transform the predictions and actual values to the original scale
y_test_inv = sc_y.inverse_transform(y_test)
predictions_inv = sc_y.inverse_transform(predictions.reshape(-1, 1))

# Evaluate the model performance
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

mse = mean_squared_error(y_test_inv, predictions_inv)
r2 = r2_score(y_test_inv, predictions_inv)
mae = mean_absolute_error(y_test_inv, predictions_inv)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
print(f'Mean Absolute Error: {mae}')

# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Actual')
plt.plot(predictions_inv, label='Predicted')
plt.title('Actual vs Predicted')
plt.xlabel('Time')
plt.ylabel('BTC Adj Close')
plt.legend()

# Save the plot to a file
plt.savefig('t-pot-actual_vs_predicted.png')

plt.show()
