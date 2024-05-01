import numpy as np
import lightgbm as lgb
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn import linear_model
from sklearn.base import BaseEstimator, TransformerMixin
import iArt

# Load the arrays from the .npz file
arrays = np.load('Data/arrays.npz')

# Accessing each array using its key
Z = arrays['Z']
X = arrays['X']
Y = arrays['Y']
S = arrays['S']
 
 # Concatenate the Z,X,Y,S all arrays, sort by S and then split them back
combined = np.concatenate([Z, X, Y, S], axis=1)

# Sort combined array by the last column (column of S), which is now at index -1 because of concatenation
sorted_combined = combined[np.argsort(combined[:, -1])]

# Split the sorted array back into Z, X, Y, S, taking into account the number of columns in each
Z = sorted_combined[:, 0:1]  # Z has 1 column
X = sorted_combined[:, 1:8]  # X has 7 columns, so we slice from index 1 to 8
Y = sorted_combined[:, 8:9]  # Y has 1 column
S = sorted_combined[:, 9:10] # S has 1 column

# Run the iArt test
file_path = "p_values.txt"
L = 10000
verbose = 0
random_state = 0
threshholdForX = 0.0

with open(file_path, 'a') as file:
    file.write("One-sided test\n")
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
result = iArt.test(Z=Z, X=X, Y=Y, S=S,L=L,G= median_imputer, verbose=verbose,threshholdForX = threshholdForX,mode = 'cluster',random_state=random_state, covariate_adjustment=0)
with open(file_path, 'a') as file:
    file.write("median " + str(result) + '\n')

with open(file_path, 'a') as file:
    file.write("One-sided test\n")
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
result = iArt.test(Z=Z, X=X, Y=Y, S=S,L=L,G= median_imputer, verbose=verbose,threshholdForX = threshholdForX,mode = 'cluster',random_state=random_state, covariate_adjustment=1)
with open(file_path, 'a') as file:
    file.write("median LR adjusted: " + str(result) + '\n')

with open(file_path, 'a') as file:
    file.write("One-sided test\n")
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
result = iArt.test(Z=Z, X=X, Y=Y, S=S,L=L,G= median_imputer, verbose=verbose,threshholdForX = threshholdForX,mode = 'cluster',random_state=random_state, covariate_adjustment=3)
with open(file_path, 'a') as file:
    file.write("median GBM adjusted: " + str(result) + '\n')


with open(file_path, 'a') as file:
    file.write("One-sided test\n")
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
result = iArt.test(Z=Z, X=X, Y=Y, S=S,L=L,G= median_imputer, verbose=verbose,threshholdForX = threshholdForX,mode = 'cluster',random_state=random_state)
with open(file_path, 'a') as file:
    file.write("median: " + str(result) + '\n')

RidgeRegression = IterativeImputer(estimator=linear_model.BayesianRidge(), max_iter=3)
result = iArt.test(Z=Z, X=X, Y=Y, S=S,L=L,G=RidgeRegression, verbose = verbose,mode = 'cluster',threshholdForX = threshholdForX,random_state=random_state)
with open(file_path, 'a') as file:
    file.write("RidgeRegression: " + str(result) + '\n')

result = iArt.test(Z=Z, X=X, Y=Y, S=S,L=L,G=RidgeRegression, verbose=verbose,mode = 'cluster', threshholdForX = threshholdForX,covariate_adjustment=1,random_state=random_state)
with open(file_path, 'a') as file:
    file.write("RidgeRegression with covariate adjustment: " + str(result) + '\n')

LightGBM = IterativeImputer(estimator=lgb.LGBMRegressor(), max_iter=3)
result = iArt.test(Z=Z, X=X, Y=Y,G=LightGBM,S=S,L=L,threshholdForX = threshholdForX, verbose=verbose,mode = 'cluster',random_state=random_state)
with open(file_path, 'a') as file:
    file.write("LightGBM: " + str(result) + '\n')

result = iArt.test(Z=Z, X=X, Y=Y,G=LightGBM,S=S,L=L,threshholdForX = threshholdForX, verbose=verbose,mode = 'cluster', covariate_adjustment=3,random_state=random_state)
with open(file_path, 'a') as file:
    file.write("LightGBM with covariate adjustment: " + str(result) + '\n')
