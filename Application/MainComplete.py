import numpy as np
import lightgbm as lgb
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn import linear_model
from sklearn.base import BaseEstimator, TransformerMixin
import Application.iArt2 as iArt2
import pandas as pd

# Load the arrays from the .npz file
arrays = np.load('Data/arrays_Y_nomissing.npz')

# Accessing each array using its key
Z = arrays['Z']
X = arrays['X']
Y = arrays['Y']
S = arrays['S']
 
# Run the iArt test
file_path = "p_values_Noop.txt"
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



combined = np.concatenate([Z, X, Y, S], axis=1)

# Sort combined array by the last column (S) using a stable sort algorithm
sorted_combined = combined[np.argsort(combined[:, -1], kind='mergesort')]

# Split the sorted array back into Z, X, Y, S
Z = sorted_combined[:, 0:1]    # Z has 1 column
X = sorted_combined[:, 1:8]    # X has 7 columns
Y = sorted_combined[:, 8:9]    # Y has 1 column
S = sorted_combined[:, 9:10]   # S has 1 column

#pd.DataFrame(Z).to_csv('Data/Z.csv')

#print the shape of the arrays
print(Z.shape)
print(X.shape)
print(Y.shape)
print(S.shape)
# Concatenate the Z, X, Y, S arrays

no_op_imputer = NoOpImputer()
with open(file_path, 'a') as file:
    file.write("One-sided test\n")
result = iArt2.test(Z=Z, X=X, Y=Y, S=S,L=L,G= no_op_imputer, verbose=1,threshholdForX = threshholdForX,mode = 'cluster',random_state=random_state)
with open(file_path, 'a') as file:
    file.write("NoOp: " + str(result) + '\n')

"""
from scipy.stats import rankdata

def custom_rank_sum(Y, Z):
    # Y is an array of sample data
    # Z is an array or scalar to multiply with the ranks of Y
    ranked_Y = rankdata(Y)  # Compute the ranks of Y
    print(ranked_Y)
    print(ranked_Y[Z == 1])
    T = ranked_Y * Z         # Multiply ranks by Z
    return T

# Example usage:
Y = np.array([5, 1, 3, 10,10, 7])
Z = np.array([1, 1, 0, 1,1, 1.0])

print(custom_rank_sum(Y,Z))"""