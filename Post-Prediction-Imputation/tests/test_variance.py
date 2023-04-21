import sys
sys.path.append('../')

import Simulation as Generator
import  pandas as pd


Nsize = 100000

print("Begin")

# Simulate data
DataGen = Generator.DataGenerator(N = Nsize, N_T = int(Nsize / 2), N_S = int(Nsize / 20), beta_11 = 1, beta_12 = 1, beta_21 = 1, beta_22 = 1, beta_23 = 1, beta_31 = 1, beta_32 = 0, MaskRate=0.3,Unobserved=0, Single=0, verbose=1)

X, Z, U, Y, M, S = DataGen.GenerateData()

df = pd.DataFrame(Y)
print(df.describe())

df = pd.DataFrame(X)
print(df.describe())

df = pd.DataFrame(M)
print(df.describe())

print(M.sum())


        



