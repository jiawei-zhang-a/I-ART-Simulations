import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
import iArtNormalize as iArt

# Load the arrays from the .npz file
arrays = np.load('Data/arrays.npz')

# Accessing each array using its key
Z = arrays['Z']
X = arrays['X']
Y = arrays['Y']
S = arrays['S']

# Run the iArt test
file_path = "p_values_nomalizedY.txt"
L = 10000
verbose = 0
random_state = 0
threshholdForX = 0.0

# Write the results with one-sided test
with open(file_path, 'a') as file:
    file.write("One-sided test\n")
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
result = iArt.test(Z=Z, X=X, Y=Y, S=S,L=L,G= median_imputer, verbose=verbose,threshholdForX = threshholdForX,mode = 'cluster',random_state=random_state)
with open(file_path, 'a') as file:
    file.write("median: " + str(result) + '\n')

result = iArt.test(Z=Z, X=X, Y=Y, S=S,L=L, verbose = verbose,mode = 'cluster',threshholdForX = threshholdForX,random_state=random_state)
with open(file_path, 'a') as file:
    file.write("RidgeRegression: " + str(result) + '\n')

result = iArt.test(Z=Z, X=X, Y=Y, S=S,L=L, verbose=verbose,mode = 'cluster', threshholdForX = threshholdForX,covariate_adjustment=1,random_state=random_state)
with open(file_path, 'a') as file:
    file.write("RidgeRegression with covariate adjustment: " + str(result) + '\n')

LightGBM = IterativeImputer(estimator=lgb.LGBMRegressor(n_jobs = 26,verbosity=-1), max_iter=1)
result = iArt.test(Z=Z, X=X, Y=Y,G=LightGBM,S=S,L=L,threshholdForX = threshholdForX, verbose=verbose,mode = 'cluster',random_state=random_state)
with open(file_path, 'a') as file:
    file.write("LightGBM: " + str(result) + '\n')

result = iArt.test(Z=Z, X=X, Y=Y,G=LightGBM,S=S,L=L,threshholdForX = threshholdForX, verbose=verbose,mode = 'cluster', covariate_adjustment=3,random_state=random_state)
with open(file_path, 'a') as file:
    file.write("LightGBM with covariate adjustment: " + str(result) + '\n')
