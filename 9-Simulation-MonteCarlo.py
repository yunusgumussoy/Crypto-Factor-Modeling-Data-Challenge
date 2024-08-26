import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load and prepare data
df = pd.read_excel('C:/Users/yunus/Downloads/Data Deneme/Custom-Dataset.xlsx')

# Clean feature names to remove special characters
df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)

# Separate features and target variable
X = df.drop(columns=['BTC_AdjClose'])
y = df['BTC_AdjClose']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = lgb.LGBMRegressor()
model.fit(X_train, y_train)

# Baseline model predictions and metrics
baseline_predictions = model.predict(X_test)
baseline_mse = mean_squared_error(y_test, baseline_predictions)
baseline_r2 = r2_score(y_test, baseline_predictions)
baseline_mae = mean_absolute_error(y_test, baseline_predictions)

print(f"Baseline Mean Squared Error: {baseline_mse}")
print(f"Baseline R-Squared Score: {baseline_r2}")
print(f"Baseline Mean Absolute Error: {baseline_mae}")

# Define the number of simulations
n_simulations = 10000

# Define the distribution for the interest rate (e.g., mean=5%, std=0.5%)
interest_rate_mean = 0.05
interest_rate_std = 0.005

# Generate random interest rate scenarios
interest_rate_scenarios = np.random.normal(interest_rate_mean, interest_rate_std, n_simulations)

# Initialize lists to store metrics for each simulation
mse_results = []
r2_results = []
mae_results = []

# Simulate the model predictions under different interest rate scenarios
for rate in interest_rate_scenarios:
    X_test_simulated = X_test.copy()
    X_test_simulated['FederalFundsEffectiveRate'] = rate  # Add the simulated feature
    
    # Predict with the simulated test set
    predictions = model.predict(X_test_simulated)
    
    # Calculate metrics for the simulated model
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    
    # Append metrics to lists
    mse_results.append(mse)
    r2_results.append(r2)
    mae_results.append(mae)

# Analyze the distribution of MSE, R-Squared, and MAE results
mse_mean = np.mean(mse_results)
mse_std = np.std(mse_results)
mse_var = np.percentile(mse_results, 5)  # Value at Risk (5th percentile)

r2_mean = np.mean(r2_results)
r2_std = np.std(r2_results)

mae_mean = np.mean(mae_results)
mae_std = np.std(mae_results)

print(f"Simulated Mean MSE: {mse_mean}")
print(f"Simulated Standard Deviation of MSE: {mse_std}")
print(f"Simulated Value at Risk (5th percentile): {mse_var}")

print(f"Simulated Mean R-Squared: {r2_mean}")
print(f"Simulated Standard Deviation of R-Squared: {r2_std}")

print(f"Simulated Mean MAE: {mae_mean}")
print(f"Simulated Standard Deviation of MAE: {mae_std}")
