import numpy as np
import lightgbm as lgb
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn import linear_model
from sklearn.base import BaseEstimator, TransformerMixin
import iArt

# Load the arrays from the .npz file
arrays = np.load('Data/arrays_nomissing.npz')

# Accessing each array using its key
Z = arrays['Z']
X = arrays['X']
Y = arrays['Y']
S = arrays['S']
 
# Run the iArt test
file_path = "p_values_median.txt"
L = 10000
verbose = 0
random_state = 0
threshholdForX = 0.0

# For Compelete Analysis
class NoOpImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Initialization code can include parameters if needed
        pass

    def fit(self, X, y=None):
        # Nothing to do here, return self to allow chaining
        return self

    def transform(self, X):
        # Check if X is a numpy array, if not, convert it to avoid potential issues
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        # Return the data unchanged
        return X

    def fit_transform(self, X, y=None):
        # This method can often be optimized but here we'll just use fit and transform sequentially
        return self.fit(X, y).transform(X)

no_op_imputer = NoOpImputer()
"""with open(file_path, 'a') as file:
    file.write("One-sided test\n")
result = iArt.test(Z=Z, X=X, Y=Y, S=S,L=L,G= no_op_imputer, verbose=verbose,threshholdForX = threshholdForX,mode = 'cluster',random_state=random_state)
with open(file_path, 'a') as file:
    file.write("NoOp: " + str(result) + '\n')"""
with open(file_path, 'a') as file:
    file.write("One-sided test\n")
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
result = iArt.test(Z=Z, X=X, Y=Y, S=S,L=L,G= median_imputer, verbose=verbose,threshholdForX = threshholdForX,mode = 'cluster',random_state=random_state, covariate_adjustment=0)
with open(file_path, 'a') as file:
    file.write("median " + str(result) + '\n')

"""with open(file_path, 'a') as file:
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
    file.write("median GBM adjusted: " + str(result) + '\n')"""



"""with open(file_path, 'a') as file:
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
"""