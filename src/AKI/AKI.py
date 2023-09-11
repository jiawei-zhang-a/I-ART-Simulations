import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from scipy.special import expit as logistic
import sys
sys.path.append('../')
from causalpart import preptest 

def GenerateM(X, U, Y, MaskRate, Missing_lambda=None):
    U = U.reshape(-1,)
    n = X.shape[0]
    M = np.zeros(n)  # Initialize M with zeros
    M_lamda = np.zeros((n, 1))

    for i in range(n):
        sum3 = sum(p * X[i, p-1] for p in range(1, 7)) / np.sqrt(5)
        sum2 = sum(p * np.cos(X[i, p-1]) for p in range(1, 7)) / np.sqrt(5)
        M_lamda[i][0] = sum3 + sum2# + 10 * logistic(Y[i]) + U[i]

    if Missing_lambda is None:
        lambda1 = np.percentile(M_lamda, 100 * (1-MaskRate))
    else:
        lambda1 = Missing_lambda

    for i in range(n):
        sum3 = sum(p * X[i, p-1] for p in range(1, 10)) / np.sqrt(5)
        sum2 = sum(p * np.cos(X[i, p-1]) for p in range(1, 10)) / np.sqrt(5)
        
        if sum3 + sum2> lambda1: #+ 10 * logistic(Y[i]) + U[i] > lambda1:
            M[i] = 1  # Set M[i] to 1 if Y[i] will be missing
    
    return M



# Load the dataset
data = pd.read_csv('ELAIA-1_deidentified_data_10-6-2020.csv', na_values="NA")
print(data.describe())
data_missing = data.isna().sum().to_csv("t")
with open("t.txt", 'w') as f:
    f.write(str(pd.DataFrame(data_missing)))



# Extracting the Intervention and Outcome
Z = data['alert'].to_numpy().reshape(-1,1)
Y = data['composite_outcome'].to_numpy().reshape(-1,1)

# List of post-treatment variables
post_treatment_variables = ['aki_progression14', 'consult14', 'death14', 'dialysis14', 'time_to_aki_progression', 
                            'time_to_composite_outcome', 'time_to_consult', 'time_to_death', 'time_to_dialysis']

# Remove post-treatment variables from your list of covariates
covariate_columns = [col for col in data.columns if col not in ['alert', 'composite_outcome', 'id'] + post_treatment_variables]
X = data[covariate_columns].to_numpy()

# Standardizing the Covariates
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Convert the standardized data back to a DataFrame for easier handling
X_standardized = pd.DataFrame(X_standardized, columns=covariate_columns)

# Generate U
mean = 0
std = 0.5
U = np.random.normal(mean, std, len(Y)).reshape(-1, 1)

# Generate missing index for Y
MaskRate = 0.5
#GenerateM(X, U, Y, MaskRate).reshape(-1, 1)
M = np.random.choice([0, 1], size=Y.shape, p=[1-MaskRate, MaskRate]) 

# Pretest
reject, p_values = preptest(Z=Z, X=X, Y=Y, M=M,G=1, L=1000, verbose=1,oracle=True)
with open('AKI.txt', 'a') as f:
    print("p-values-oracle: ", p_values, file=f)

reject, p_values = preptest(Z=Z, X=X, Y=Y,  M=M,G='median', L=1000, verbose=1)
with open('AKI.txt', 'a') as f:
    print("p-values-median: ", p_values, file=f)

reject, p_values = preptest(Z=Z, X=X, Y=Y, M=M, G='bayesianridge', L=100, verbose=1)
with open('AKI.txt', 'a') as f:
    print("p-values-bayesianridge: ", p_values, file=f)

reject, p_values = preptest(Z=Z, X=X, Y=Y, M=M, G='bayesianridge', L=100, verbose=1, covariate_adjustment=1)
with open('AKI.txt', 'a') as f:
    print("p-values-bayesianridge-adjusted: ", p_values, file=f)

reject, p_values = preptest(Z=Z, X=X, Y=Y, M=M,G='lightgbm', L=50, verbose=1)
with open('AKI.txt', 'a') as f:
    print("p-values-lightgbm: ", p_values, file=f)

reject, p_values = preptest(Z=Z, X=X, Y=Y, M=M, G='lightgbm', L=50, verbose=1, covariate_adjustment=1)
with open('AKI.txt', 'a') as f:
    print("p-values-lightgbm-adjusted: ", p_values, file=f)


