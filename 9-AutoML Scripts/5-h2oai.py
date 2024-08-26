# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 06:58:46 2024

@author: Yunus
"""

# Import necessary libraries
import h2o
from h2o.automl import H2OAutoML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# Initialize the H2O cluster
h2o.init()

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

# Ensure y_scaled is a 2D array
y_scaled = y_scaled.reshape(-1, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)

# Convert to DataFrame and then to H2O Frame
train = pd.DataFrame(X_train, columns=X.columns)
train[target] = y_train
train_h2o = h2o.H2OFrame(train)

test = pd.DataFrame(X_test, columns=X.columns)
test[target] = y_test
test_h2o = h2o.H2OFrame(test)

# Specify the target and predictors
x = list(train.columns[:-1])  # List of all columns except the target
y = target

# Initialize and run H2O AutoML
aml = H2OAutoML(max_runtime_secs=3600,  # Set time limit (in seconds)
                seed=42,
                stopping_metric='RMSE',
                sort_metric='RMSE')
aml.train(x=x, y=y, training_frame=train_h2o)

# View the AutoML Leaderboard
lb = aml.leaderboard
print(lb)

# Predict on test data
preds = aml.leader.predict(test_h2o)

# Convert predictions and actual values back to original scale
y_test_inv = sc_y.inverse_transform(y_test)
preds_inv = sc_y.inverse_transform(preds.as_data_frame().values.reshape(-1, 1))

# Evaluate the model performance
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

mse = mean_squared_error(y_test_inv, preds_inv)
r2 = r2_score(y_test_inv, preds_inv)
mae = mean_absolute_error(y_test_inv, preds_inv)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
print(f'Mean Absolute Error: {mae}')

# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Actual')
plt.plot(preds_inv, label='Predicted')
plt.title('Actual vs Predicted')
plt.xlabel('Time')
plt.ylabel('BTC Adj Close')
plt.legend()

# Save the plot to a file
plt.savefig('h2oai-actual_vs_predicted.png')

plt.show()

# Shutdown H2O cluster
h2o.shutdown(prompt=False)

