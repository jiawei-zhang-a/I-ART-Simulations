import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # Explicitly require this experimental feature
from sklearn.impute import IterativeImputer  # Now you can import normally from sklearn.impute
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import Simulation as Generator  # Assuming this is the module with your data generating function
from scipy.stats import spearmanr

beta_coef = 0.1
DataGen = Generator.DataGenerator(N=1000, strata_size=10, beta_11=beta_coef, beta_12=beta_coef, beta_21=beta_coef,
                                  beta_22=beta_coef, beta_23=beta_coef, beta_31=beta_coef, beta_32=beta_coef,
                                  MaskRate=0.5, Single=1, verbose=1, Missing_lambda=None)

X, Z, U, Y, M, S = DataGen.GenerateData()

def MICE(X, Z, U, Y, M, S):
    print(pd.DataFrame(X).describe())

    # Masking the Y with Missing indicator M
    Y_True = pd.DataFrame(Y, columns=['Y'])
    Y = np.ma.masked_array(Y, mask=M).filled(np.nan)
    Y_df = pd.DataFrame(Y, columns=['Y'])

    # Convert X into a DataFrame and rename its columns
    X_df = pd.DataFrame(X, columns=[f'X_{i}' for i in range(X.shape[1])])

    # Convert Z into a DataFrame
    Z_df = pd.DataFrame(Z, columns=['Z'])

    # Now, create the combined DataFrame
    df_train = pd.concat([Z_df, X_df, Y_df], axis=1)

    # ... [further processing] ...

    # Let's say you want to use iterative imputer here:
    max_iter = 10 # You would set this to your preferred number of iterations.

    # Define the imputer using Bayesian Ridge as the estimator
    bayesian_ridge_imputer = IterativeImputer(estimator=linear_model.BayesianRidge(), max_iter=max_iter, random_state=0)

    # Define the imputer using XGBoost as the estimator
    xgb_imputer = IterativeImputer(estimator=XGBRegressor(n_jobs=1), max_iter=max_iter, random_state=0)

    df = pd.DataFrame()
    # Fit the imputer on the DataFrame excluding the 'Y' column as a feature
    # The imputer expects the features and the column to be imputed in the same DataFrame
    df['Y_imputed_bayesian'] = bayesian_ridge_imputer.fit_transform(df_train)[:, -1]
    df['Y_imputed_xgb'] = xgb_imputer.fit_transform(df_train)[:, -1]
    Y_True['Y'].to_csv("Y_True.csv")
    # fill df['Y'] 
    df['Y'] = Y_True['Y']

    df.to_csv("df.csv")
    # If you need to work with the imputed values, you can do so like this:
    imputed_Y_bayesian = df['Y_imputed_bayesian']
    imputed_Y_xgb = df['Y_imputed_xgb']

    # Find indices where Y was originally missing
    missing_indices = Y_df['Y'].isnull()

    # Calculate MSE and Correlation for Bayesian Ridge imputed values
    bayesian_mse = mean_squared_error(df['Y'][missing_indices], df['Y_imputed_bayesian'][missing_indices])
    bayesian_corr = np.corrcoef(df['Y'][missing_indices], df['Y_imputed_bayesian'][missing_indices])[0, 1]

    # Calculate MSE and Correlation for XGBoost imputed values
    xgb_mse = mean_squared_error(df['Y'][missing_indices], df['Y_imputed_xgb'][missing_indices])
    xgb_corr = np.corrcoef(df['Y'][missing_indices], df['Y_imputed_xgb'][missing_indices])[0, 1]

    print(f"BayesianRidge Imputed MSE: {bayesian_mse}, Correlation: {bayesian_corr}")
    print(f"XGBoost Imputed MSE: {xgb_mse}, Correlation: {xgb_corr}")

    # Calculate Spearman Correlation for Bayesian Ridge imputed values
    bayesian_spearman_corr, _ = spearmanr(df['Y'][missing_indices].rank(), df['Y_imputed_bayesian'][missing_indices].rank())

    # Calculate Spearman Correlation for XGBoost imputed values
    xgb_spearman_corr, _ = spearmanr(df['Y'][missing_indices].rank(), df['Y_imputed_xgb'][missing_indices].rank())

    # Print the results
    print(f"Bayesian Ridge Imputed Spearman Correlation: {bayesian_spearman_corr}")
    print(f"XGBoost Imputed Spearman Correlation: {xgb_spearman_corr}")


"""
# Plotting the results
plt.figure(figsize=(14, 7))

# Plot true values where originally missing
plt.plot(df['Y'][missing_indices].values, 'b-', label="True Y", alpha=0.6)

# Plot Bayesian Ridge imputed values
plt.plot(df['Y_imputed_bayesian'][missing_indices].values, 'r-', label="BayesianRidge Imputed Y", alpha=0.6)

# Plot XGBoost imputed values
plt.plot(df['Y_imputed_xgb'][missing_indices].values, 'g-', label="XGBoost Imputed Y", alpha=0.6)

plt.title("Imputed vs True Y for Missing Data")
plt.xlabel("Sample Index")
plt.ylabel("Y Value")
plt.legend()
plt.show()

plt.figure(figsize=(14, 7))

# Retrieve and sort the true and imputed values where Y was originally missing
sorted_indices = np.argsort(df['Y'][missing_indices].values)
sorted_true_y = df['Y'][missing_indices].values[sorted_indices]
sorted_bayesian_imputed_y = df['Y_imputed_bayesian'][missing_indices].values[sorted_indices]
sorted_xgb_imputed_y = df['Y_imputed_xgb'][missing_indices].values[sorted_indices]

# Plot sorted true values
plt.plot(sorted_true_y, 'b-', label="True Y", alpha=0.6)

# Plot sorted Bayesian Ridge imputed values
plt.plot(sorted_bayesian_imputed_y, 'r-', label="BayesianRidge Imputed Y", alpha=0.6)

# Plot sorted XGBoost imputed values
plt.plot(sorted_xgb_imputed_y, 'g-', label="XGBoost Imputed Y", alpha=0.6)

plt.title("Imputed vs Sorted True Y for Missing Data")
plt.xlabel("Sorted Sample Index")
plt.ylabel("Y Value")
plt.legend()
plt.show()"""

def direct(X, Z, U, Y, M, S):
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

MICE(X, Z, U, Y, M, S)
direct(X, Z, U, Y, M, S)


"""plt.figure(figsize=(14, 7))

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
plt.show()"""
