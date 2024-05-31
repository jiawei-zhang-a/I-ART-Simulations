import sys
import numpy as np
import os
import iArt
import lightgbm as lgb
import xgboost as xgb
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn import linear_model

# Parameter for the iArt.test function
L = 1
verbose = 0
random_state = 0
threshold_covariate_median_imputation = 0.0

# Define β values range 
beta_values = np.linspace(-0.5, 0.5, 1001)  

# Retrieve the job array index from SLURM
array_index = int(sys.argv[1]) - 1 

# Load the arrays from the .npz file
arrays = np.load('Data/arrays.npz')

# Accessing each array using its key
Z = arrays['Z']
X = arrays['X']
Y = arrays['Y']
S = arrays['S']
M = np.isnan(Y)  # Mask for missing values in Y

# Select the β for this task
beta = beta_values[array_index]

# Adjust Y based on β for this task (Implement the logic as needed)
Y_adjusted = Y.copy()
# Example adjustment, adjust according to your actual logic
Y_adjusted[(M == 0) & (Z == 0)] += beta

# Define the folder name
folder_name = "ConfidenceRegions"

# Check if the folder does not exist
if not os.path.exists(folder_name):
    # Create the folder
    os.makedirs(folder_name)
    print(f"The folder '{folder_name}' has been created.")
else:
    print(f"The folder '{folder_name}' already exists.")
# Run the iArt.test with the adjusted Y

# Save the result for median imputer
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
reject,p_values = iArt.test(G=median_imputer,Z=Z, X=X, Y=Y_adjusted, S=S, L=L, verbose=verbose, randomization_design='cluster', threshold_covariate_median_imputation=0.0, random_state=random_state)
result_path = f"{folder_name}/test_MedianImputation_{beta}.npy"
np.save(result_path, np.array([beta, reject,*p_values]))  # Adjust based on actual result structure

# Save the result for median imputer with covariate adjustment
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
reject,p_values = iArt.test(G=median_imputer,Z=Z, X=X, Y=Y_adjusted, S=S, L=L, verbose=verbose, randomization_design='cluster', threshold_covariate_median_imputation=0.0, random_state=random_state, covariate_adjustment='linear')
result_path = f"{folder_name}/test_MedianImputationwithCovariateAdjustment_{beta}.npy"
np.save(result_path, np.array([beta, reject,*p_values]))  # Adjust based on actual result structure

# Save the result for ridge regression
RidgeRegression = IterativeImputer(estimator=linear_model.BayesianRidge(), max_iter=3)
reject,p_values = iArt.test(G=RidgeRegression, Z=Z, X=X, Y=Y_adjusted, S=S, L=L, verbose=verbose, randomization_design='cluster', threshold_covariate_median_imputation=0.0, random_state=random_state)
result_path = f"{folder_name}/test_AlgoILinear_{beta}.npy"
np.save(result_path, np.array([beta, reject,*p_values]))  # Adjust based on actual result structure

# Save the result for ridge regression with covariate adjustment
reject,p_values = iArt.test(G=RidgeRegression,Z=Z, X=X, Y=Y_adjusted, S=S, L=L, verbose=verbose, randomization_design='cluster', threshold_covariate_median_imputation=0.0, covariate_adjustment='linear', random_state=random_state)
result_path = f"{folder_name}/test_AlgoIILinear_{beta}.npy"
np.save(result_path, np.array([beta, reject,*p_values]))  # Adjust based on actual result structure

LightGBM = IterativeImputer(estimator=lgb.LGBMRegressor(verbosity=-1), max_iter=3)
# Save the result for LightGBM
reject,p_values = iArt.test(G=LightGBM, Z=Z, X=X, Y=Y_adjusted, S=S, L=L, verbose=verbose, randomization_design='cluster', threshold_covariate_median_imputation=0.0, random_state=random_state)
result_path = f"{folder_name}/test_AlgoIBoosting_{beta}.npy"
np.save(result_path, np.array([beta, reject,*p_values]))  # Adjust based on actual result structure

# Save the result for LightGBM with covariate adjustment
reject,p_values = iArt.test(G=LightGBM, Z=Z, X=X, Y=Y_adjusted, S=S, L=L, verbose=verbose, randomization_design='cluster', threshold_covariate_median_imputation=0.0, covariate_adjustment='lightgbm', random_state=random_state)
result_path = f"{folder_name}/test_AlgoIIBoosting_{beta}.npy"
np.save(result_path, np.array([beta, reject,*p_values]))  # Adjust based on actual result structure

