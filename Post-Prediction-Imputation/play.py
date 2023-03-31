import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression

# Create a sample dataset with missing values
X = np.array([
    [1, 2, 3],
    [4, np.nan, 6],
    [7, 8, 9],
    [np.nan, 10, 11],
    [12, 13, np.nan],
])

lr = LinearRegression()

# Initialize the IterativeImputer with LinearRegression as the estimator
imputer = IterativeImputer(estimator=lr, random_state=0)

# Fit the imputer on the dataset and transform the data
X_imputed = imputer.fit_transform(X)

print("Original dataset with missing values:")
print(X)

print("Imputed dataset:")
print(X_imputed)


# Use a fixed X instead of a random one
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])

# Generate some random data for demonstration purposes
y = 2 + 3 * X + np.random.randn(10, 1)

# Create a LinearRegression object and fit the data
reg = lr.predict(X)

# Predict new values using the trained model
X_new = np.array([[0], [11]])
y_pred = reg.predict(X_new)

print(y_pred)
