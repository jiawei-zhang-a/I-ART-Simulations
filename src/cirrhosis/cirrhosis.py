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
        sum3 = sum(p * X[i, p-1] for p in range(1, 10)) / np.sqrt(5)
        sum2 = sum(p * np.cos(X[i, p-1]) for p in range(1, 10)) / np.sqrt(5)
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
df = pd.read_csv('cirrhosis.csv', na_values="NA")

# One-hot encoding on non-numerical columns
non_numerical_columns = ['Drug', 'Status', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
one_hot = OneHotEncoder(drop='first')
one_hot_df = pd.DataFrame(one_hot.fit_transform(df[non_numerical_columns]).toarray())
print(one_hot_df)
df = df.drop(non_numerical_columns, axis=1)
df = pd.concat([one_hot_df, df], axis=1)
df.columns = df.columns.astype(str) 

# Standardize numerical columns including 'N_Days'
numerical_columns = ['Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage', 'N_Days']
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Impute missing values with median for the X matrix
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

Z = df_imputed['0'].values  # Assuming the first column after one-hot encoding is 'Drug'
Z = Z
Y = df_imputed['Stage'].values.reshape(-1, 1)  # 'N_Days' is the outcome
X = df_imputed.drop(['Stage', '0'], axis=1).values  # Dropping 'N_Days' and the 'Drug' column to form the X matrix


# Generate U
mean = 0
std = 0.5
U = np.random.normal(mean, std, len(Y)).reshape(-1, 1)

# Generate missing index for Y
MaskRate = 0.5
M = GenerateM(X, U, Y, MaskRate).reshape(-1, 1)

# Pretest
reject, p_values = preptest(Z=Z, X=X, Y=Y,M=M, G=1, L=1000, verbose=1,oracle=True)
with open('cirrhosis.txt', 'a') as f:
    print("p-values-oracle: ", p_values, file=f)

reject, p_values = preptest(Z=Z, X=X, Y=Y,M=M, G='median', L=1000, verbose=1)
with open('cirrhosis.txt', 'a') as f:
    print("p-values-median: ", p_values, file=f)

reject, p_values = preptest(Z=Z, X=X, Y=Y,M=M, G='bayesianridge', L=1000, verbose=1)
with open('cirrhosis.txt', 'a') as f:
    print("p-values-bayesianridge: ", p_values, file=f)

reject, p_values = preptest(Z=Z, X=X, Y=Y,M=M, G='bayesianridge', L=1000, verbose=1, covariate_adjustment=1)
with open('cirrhosis.txt', 'a') as f:
    print("p-values-bayesianridge-adjusted: ", p_values, file=f)

reject, p_values = preptest(Z=Z, X=X, Y=Y,M=M, G='lightgbm', L=10, verbose=1)
with open('cirrhosis.txt', 'a') as f:
    print("p-values-lightgbm: ", p_values, file=f)

reject, p_values = preptest(Z=Z, X=X, Y=Y,M=M, G='lightgbm', L=19, verbose=1, covariate_adjustment=1)
with open('cirrhosis.txt', 'a') as f:
    print("p-values-lightgbm-adjusted: ", p_values, file=f)


