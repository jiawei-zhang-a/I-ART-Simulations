import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn import linear_model
from sklearn.base import BaseEstimator, TransformerMixin
import iArtrandom as iArt1
import iArtrandom2 as iArt2
import iArtrandom3 as iArt3



# Load the arrays from the .npz file
arrays = np.load('Data/arrays.npz')

# Accessing each array using its key
Z = arrays['Z']
X = arrays['X']
Y = arrays['Y']
S = arrays['S']

# Run the iArt test
file_path = "p_values_testnoiseinG.txt"
L = 10000
verbose = 1
random_state = 0

median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
result = iArt3.test(G=median_imputer,Z=Z, X=X, Y=Y, S=S, L=L, verbose=verbose, randomization_design='cluster', threshold_covariate_median_imputation=0.0, random_state=random_state)
with open(file_path, 'a') as file:
    file.write("Median: " + str(result) + '\n')
exit()

RidgeRegression = IterativeImputer(estimator=linear_model.BayesianRidge(),max_iter=3)
result = iArt1.test(G=RidgeRegression,Z=Z, X=X, Y=Y, S=S, L=L, verbose=verbose, randomization_design='cluster', threshold_covariate_median_imputation=0.0, random_state=random_state)
with open(file_path, 'a') as file:
    file.write("RidgeRegression: " + str(result) + '\n')

LightGBM = IterativeImputer(estimator=lgb.LGBMRegressor(), max_iter=3)
result = iArt2.test(G=LightGBM,Z=Z, X=X, Y=Y, S=S, L=L, verbose=verbose, randomization_design='cluster', threshold_covariate_median_imputation=0.0, random_state=random_state)
with open(file_path, 'a') as file:
    file.write("LightGBM: " + str(result) + '\n')

