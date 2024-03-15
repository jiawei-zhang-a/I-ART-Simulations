import sys
import numpy as np
import os
import iArt
import lightgbm as lgb
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer

# Parameter for the iArt.test function
file_path = "p_values.txt"
L = 1
verbose = 0
random_state = 0
threshholdForX = 0.0

# Define β values range 
beta_values = np.linspace(0, 10, 100)  

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
Y_adjusted[(M == 0) & (Z == 1)] -= beta

os.makedirs(os.path.dirname("Result"), exist_ok=True)  # Ensure the directory exists

# Run the iArt.test with the adjusted Y

# Save the result for median imputer
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
reject,_ = iArt.test(G=median_imputer,Z=Z, X=X, Y=Y_adjusted, S=S, L=L, verbose=verbose, mode='cluster', threshholdForX=threshholdForX, covariate_adjustment=1, random_state=random_state)
result_path = f"Result/test_median_{beta}.npy"
np.save(result_path, np.array([beta, reject]))  # Adjust based on actual result structure

# Save the result for ridge regression
reject,_ = iArt.test(Z=Z, X=X, Y=Y_adjusted, S=S, L=L, verbose=verbose, mode='cluster', threshholdForX=threshholdForX, covariate_adjustment=1, random_state=random_state)
result_path = f"Result/test_ridge_{beta}.npy"
np.save(result_path, np.array([beta, reject]))  # Adjust based on actual result structure

# Save the result for ridge regression with covariate adjustment
reject,_ = iArt.test(Z=Z, X=X, Y=Y_adjusted, S=S, L=L, verbose=verbose, mode='cluster', threshholdForX=threshholdForX, covariate_adjustment=3, random_state=random_state)
result_path = f"Result/test_ridge_covariate_{beta}.npy"
np.save(result_path, np.array([beta, reject]))  # Adjust based on actual result structure

LightGBM = IterativeImputer(estimator=lgb.LGBMRegressor(n_jobs = 26,verbosity=-1), max_iter=1)
# Save the result for LightGBM
reject,_ = iArt.test(G=LightGBM, Z=Z, X=X, Y=Y_adjusted, S=S, L=L, verbose=verbose, mode='cluster', threshholdForX=threshholdForX, random_state=random_state)
result_path = f"Result/test_lightgbm_{beta}.npy"
np.save(result_path, np.array([beta, reject]))  # Adjust based on actual result structure

# Save the result for LightGBM with covariate adjustment
reject,_ = iArt.test(G=LightGBM, Z=Z, X=X, Y=Y_adjusted, S=S, L=L, verbose=verbose, mode='cluster', threshholdForX=threshholdForX, covariate_adjustment=3, random_state=random_state)
result_path = f"Result/test_lightgbm_covariate_{beta}.npy"
np.save(result_path, np.array([beta, reject]))  # Adjust based on actual result structure

