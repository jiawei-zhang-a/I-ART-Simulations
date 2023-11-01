import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import Simulation as Generator

beta_coef = 0.1
DataGen = Generator.DataGenerator(N=1000, strata_size=10, beta_11=beta_coef, beta_12=beta_coef, beta_21=beta_coef,
                                  beta_22=beta_coef, beta_23=beta_coef, beta_31=beta_coef, beta_32=beta_coef,
                                  MaskRate=0.5, Single=1, verbose=1, Missing_lambda=None)

X, Z, U, Y, M, S = DataGen.GenerateData()

# Reshape Z, Y, and M to be 1-dimensional
Z = Z.ravel()
Y = Y.ravel()
M = M.ravel()

# Convert X into a DataFrame and rename its columns
X_df = pd.DataFrame(X, columns=[f'X_{i}' for i in range(X.shape[1])])

# Now, create the combined DataFrame
df = pd.concat([X_df, pd.Series(Z, name='Z'), pd.Series(Y, name='Y'), pd.Series(M, name='M')], axis=1)

feature_columns = [f'X_{i}' for i in range(X.shape[1])] + ['Z']

# Split data based on the missing indicator
train = df[df['M'] == 0]
predict_set = df[df['M'] == 1]

X_train = train[feature_columns]
Y_train = train['Y']
X_predict = predict_set[feature_columns]

# Train a BayesianRidge model
bayesian_ridge = linear_model.BayesianRidge()
bayesian_ridge.fit(X_train, Y_train)
ridge_pred = bayesian_ridge.predict(X_predict)

# Train an XGBoost model
xgb = XGBRegressor(objective='reg:squarederror')
xgb.fit(X_train, Y_train)
xgb_pred = xgb.predict(X_predict)

# Calculate the correlation and MSE for both models
ridge_mse = mean_squared_error(predict_set['Y'], ridge_pred)
xgb_mse = mean_squared_error(predict_set['Y'], xgb_pred)

ridge_corr = np.corrcoef(predict_set['Y'], ridge_pred)[0, 1]
xgb_corr = np.corrcoef(predict_set['Y'], xgb_pred)[0, 1]

print(f"BayesianRidge MSE: {ridge_mse}, Correlation: {ridge_corr}")
print(f"XGBoost MSE: {xgb_mse}, Correlation: {xgb_corr}")



plt.figure(figsize=(14, 7))

# Plot true values
plt.plot(predict_set['Y'].values, 'b-', label="True Y", alpha=0.6)

# Plot BayesianRidge predictions
plt.plot(ridge_pred, 'r-', label="BayesianRidge Predicted Y", alpha=0.6)

# Plot XGBoost predictions
plt.plot(xgb_pred, 'g-', label="XGBoost Predicted Y", alpha=0.6)

plt.title("Predicted vs True Y")
plt.xlabel("Sample Index")
plt.ylabel("Y Value")
plt.legend()
plt.show()


# Sort the true Y values and associated predictions
sorted_indices = np.argsort(predict_set['Y'].values)
sorted_y = predict_set['Y'].values[sorted_indices]
sorted_ridge_pred = ridge_pred[sorted_indices]
sorted_xgb_pred = xgb_pred[sorted_indices]

plt.figure(figsize=(14, 7))

# Plot true values
plt.plot(sorted_y, 'b-', label="True Y", alpha=0.6)

# Plot BayesianRidge predictions
plt.plot(sorted_ridge_pred, 'r-', label="BayesianRidge Predicted Y", alpha=0.6)

# Plot XGBoost predictions
plt.plot(sorted_xgb_pred, 'g-', label="XGBoost Predicted Y", alpha=0.6)

plt.title("Predicted vs Sorted True Y")
plt.xlabel("Sample Index")
plt.ylabel("Y Value")
plt.legend()
plt.show()
