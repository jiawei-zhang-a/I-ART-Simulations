import sys
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
from sklearn.impute import SimpleImputer
import simulation as Generator
import rimo as retrain
import os
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the dataset
df = pd.read_csv('cirrhosis.csv', na_values="NA")

# 1. One-hot encoding on non-numerical columns
non_numerical_columns = [ 'Drug', 'Status','Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
one_hot = OneHotEncoder(drop='first')
one_hot_df = pd.DataFrame(one_hot.fit_transform(df[non_numerical_columns]).toarray())
df = df.drop(non_numerical_columns, axis=1)
df = pd.concat([one_hot_df,df], axis=1)

# 2. Standardize numerical columns
numerical_columns = ['N_Days', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage']
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns] )

Z = df[0].values
Y = df.drop([0], axis=1)

# 3. Impute missing values
Framework = retrain.RetrainTest()
BayesianRidge = IterativeImputer(estimator = linear_model.BayesianRidge())
medina_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
reject,p_values  = Framework.retrain_test(Z = Z,Y=Y,G=medina_imputer,L=100,verbose=1)

print("p-values: ", p_values)
print("Reject: ", reject)