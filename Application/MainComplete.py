import numpy as np
import lightgbm as lgb
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn import linear_model
from sklearn.base import BaseEstimator, TransformerMixin
import iArt

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


#print the shape of the arrays
print(Z.shape)
print(X.shape)
print(Y.shape)
print(S.shape)
# Concatenate the Z,X,Y,S all arrays, sort by S and then split them back
combined = np.concatenate([Z, X, Y, S], axis=1)

# Sort combined array by the last column (column of S), which is now at index -1 because of concatenation
sorted_combined = combined[np.argsort(combined[:, -1])]

# Split the sorted array back into Z, X, Y, S, taking into account the number of columns in each
Z = sorted_combined[:, 0:1]  # Z has 1 column
X = sorted_combined[:, 1:8]  # X has 7 columns, so we slice from index 1 to 8
Y = sorted_combined[:, 8:9]  # Y has 1 column
S = sorted_combined[:, 9:10] # S has 1 column

no_op_imputer = NoOpImputer()
with open(file_path, 'a') as file:
    file.write("One-sided test\n")
result = iArt.test(Z=Z, X=X, Y=Y, S=S,L=L,G= no_op_imputer, verbose=verbose,threshholdForX = threshholdForX,mode = 'cluster',random_state=random_state)
with open(file_path, 'a') as file:
    file.write("NoOp: " + str(result) + '\n')