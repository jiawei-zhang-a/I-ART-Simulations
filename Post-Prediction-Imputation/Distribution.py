import sys
import numpy as np
import multiprocessing
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
from sklearn.impute import SimpleImputer
import multiprocessing
import Simulation as Generator
import OneShot
import warnings
import xgboost as xgb
import os
import  pandas as pd


Nsize = 1000

print("Begin")

# Simulate data
DataGen = Generator.DataGenerator(N = Nsize, N_T = int(Nsize / 2), N_S = int(Nsize / 20), beta_11 = 0, beta_12 = 0, beta_21 = 0, beta_22 = 0, beta_23 = 0, beta_31 = 0, beta_32 = 0, MaskRate=0.3,Unobserved=0, Single=0)

X, Z, U, Y, M, S = DataGen.GenerateData()

df = pd.DataFrame(Y)
print(df.describe())

df = pd.DataFrame(X)
print(df.describe())

df = pd.DataFrame(M)
print(df.describe())

        



