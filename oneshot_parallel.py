import pandas as pd
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import random
from sklearn.impute import KNNImputer
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import scipy.stats as stats
import multiprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.kernel_approximation import Nystroem

#load data
X = np.load("/Users/jiaweizhang/research/data/X.npy")
Y = np.load("/Users/jiaweizhang/research/data/Y.npy")
Z = np.load("/Users/jiaweizhang/research/data/Z.npy")
M = np.load("/Users/jiaweizhang/research/data/M.npy")
S = np.load("/Users/jiaweizhang/research/data/S.npy")

N = len(X)

#split based on strata
def split_df(df,index_S):

    # Sort the groups by the number of rows in each group
    sorted_df = df.sort_values(by = index_S, ascending=True)
    
    # Split the sorted groups into two equal-sized sets of 100 strata each
    df_set1 = sorted_df.iloc[:int(N/2),0 : index_S]
    df_set2 = sorted_df.iloc[int(N/2):N, 0 : index_S]

    #set the index of the two sets from zero to 1
    df_set1.index = range(len(df_set1))
    df_set2.index = range(len(df_set2))
    
    # Return the two sets of strata
    return df_set1, df_set2

def T(z,y):

    #the Wilcoxon rank sum test
    n = len(z)
    t = 0
    #O(N^2) version
    """
    for n in range(N):
        rank = sum(1 for n_prime in range(N) if Y[n] >= Y[n_prime])
        T += Z[n] * rank
    """

    #O(N*Log(N)) version
    my_list = []
    for i in range(n):
        my_list.append((z[i],y[i]))
    sorted_list = sorted(my_list, key=lambda x: x[1])

    #Calculate
    for i in range(n):
        t += sorted_list[i][0] * (i + 1)
    
    return t

def getT(G, df):
    
    # Get the imputed data Y and indicator Z
    df_imputed = G.transform(df)
    y = df_imputed[:, Z.shape[1] + X.shape[1]:df_imputed.shape[1]]
    z = df_imputed[:, 0]
    
    z_tiled = np.tile(z, 3)

    # Concatenate the tiled versions of Z together
    new_z = np.concatenate((z_tiled,))
    new_y = y.flatten()

    #the Wilcoxon rank sum test
    t = T(new_z,new_y)

    return t

def worker(args):
    # unpack the arguments
    X, Y_masked, S, G1, G2, t1_obs, t2_obs, L = args

    # simulate data and calculate test statistics
    t1_sim = np.zeros(L)
    t2_sim = np.zeros(L)

    for l in range(L):

        # simulate treatment indicators in parts 1 and 2
        df_sim = pd.DataFrame(np.concatenate((X, Y_masked, S), axis=1))
        
        # split the simulated data into two parts
        df1_sim, df2_sim = split_df(df_sim, index_S = X.shape[1] + Y.shape[1])

        # simulate treatment indicators in parts 1 and 2
        Z_1 = np.random.binomial(1, 0.5, df1_sim.shape[0]).reshape(-1, 1)
        Z_2 = np.random.binomial(1, 0.5, df2_sim.shape[0]).reshape(-1, 1)
        df1_sim = pd.concat([pd.DataFrame(Z_1), df1_sim], axis=1)
        df2_sim = pd.concat([pd.DataFrame(Z_2), df2_sim], axis=1)

        
        # get the test statistics in part 1
        t1_sim[l] = getT(G2, df1_sim)

        # get the test statistics in part 2
        t2_sim[l] = getT(G1, df2_sim)

        # Calculate the completeness percentage
        if l % 100 == 0:
            completeness = l / L * 100  
            print(f"Task is {completeness:.2f}% complete.")

    p1 = np.mean(t1_sim >= t1_obs, axis=0)
    p2 = np.mean(t2_sim >= t2_obs, axis=0)

    return p1, p2




def one_shot_test_parallel(Z, X, M, Y, S, G1, G2, L=10000, n_jobs=multiprocessing.cpu_count()):
    """
    A one-shot framework for testing H_0.

    Args:
    Z: 2D array of observed treatment indicators
    X: 2D array of observed covariates
    M: 2D array of observed missing indicators
    Y: 2D array of observed values for K outcomes
    G1: a function that takes (Z, X, M, Y_k) as input and returns the imputed value for outcome k
    G2: a function that takes (Z, X, M, Y_k) as input and returns the imputed value for outcome k
    L: number of Monte Carlo simulations (default is 10000)

    Returns:
    p1: 1D array of exact p-values for testing Fisher's sharp null in part 1
    p2: 1D array of exact p-values for testing Fisher's sharp null in part 2
    """
    #print train start
    print("Training start")

    # create data a whole data frame
    Y_masked = np.ma.masked_array(Y, mask=M)
    Y_masked = Y_masked.filled(np.nan)
    df = pd.DataFrame(np.concatenate((Z, X, Y_masked, S), axis=1))
    
    # randomly split the data into two parts
    df1, df2 = split_df(df, index_S = Z.shape[1] + X.shape[1] + Y.shape[1])

    # impute the missing values and calculate the observed test statistics in part 1
    G1.fit(df1)
    t1_obs = getT(G1, df1)

    # impute the miassing values and calculate the observed test statistics in part 2
    G2.fit(df2)
    t2_obs = getT(G2, df2)

    #print train end
    print("Training end")
    
    # print the number of cores
    print(f"Number of cores: {n_jobs}")


    # simulate data and calculate test statistics in parallel
    args_list = [(X, Y_masked, S, G1, G2, t1_obs, t2_obs, int(L / n_jobs))] * n_jobs
    with multiprocessing.Pool(processes=n_jobs) as pool:
        p_list = pool.map(worker, args_list)
    p1 = np.mean([p[0] for p in p_list], axis=0)
    p2 = np.mean([p[1] for p in p_list], axis=0)
    
    return p1, p2

if __name__ == '__main__':
    multiprocessing.freeze_support() # This is necessary and important, not sure why 
    
    """
    #MissForest
    missForest = IterativeImputer(estimator = RandomForestRegressor(),max_iter=10, random_state=0)
    p1, p2 = one_shot_test_parallel(Z, X, M, Y, S, G1=missForest, G2=missForest)
    print("One-shot test for Fisher's sharp null for MissForest")
    print("p-values for part 1:", p1)
    print("p-values for part 2:", p2)
    
    #KNN
    KNNimputer = KNNImputer(n_neighbors=2)
    p1, p2 = one_shot_test_parallel(Z, X, M, Y, G1=KNNimputer, G2=KNNimputer)
    print("One-shot test for Fisher's sharp null for KNN imputer")
    print("p-values for part 1:", p1)
    print("p-values for part 2:", p2)
   
    #BayesianRidge
    BayesianRidge = IterativeImputer(estimator = linear_model.BayesianRidge(),max_iter=10, random_state=0)
    p1, p2 = one_shot_test_parallel(Z, X, M, Y, S, G1=BayesianRidge, G2=BayesianRidge)
    print("One-shot test for Fisher's sharp null for BayesianRidge")
    print("p-values for part 1:", p1)
    print("p-values for part 2:", p2)

    #Nystroem Method for Kernel Approximation
    pipeline = make_pipeline(
        StandardScaler(),
        Nystroem(), 
        linear_model.Ridge()
    )
    NystroemKernel = IterativeImputer(estimator = pipeline,max_iter=10, random_state=0)
    p1, p2 = one_shot_test_parallel(Z, X, M, Y, S, G1=NystroemKernel, G2=NystroemKernel)
    print("One-shot test for Fisher's sharp null for Nystroem Kernel Approximation")
    print("p-values for part 1:", p1)
    print("p-values for part 2:", p2)
 """
    #XGBoost
    pipeline = make_pipeline(
        StandardScaler(),
        xgb.XGBRegressor()
    )
    XGBoost = IterativeImputer(estimator = pipeline,max_iter=10, random_state=0)
    p1, p2 = one_shot_test_parallel(Z, X, M, Y, S, G1=XGBoost, G2=XGBoost)
    print("One-shot test for Fisher's sharp null for XGBoost")
    print("p-values for part 1:", p1)
    print("p-values for part 2:", p2)
    """
    #Neural Network
    print("One-shot test for Fisher's sharp null for Neural Network")
    pipeline = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(100, 100, 100,100), activation='relu', alpha=0.0001, random_state=0)
    )
    NN_imputer = IterativeImputer(estimator=pipeline.named_steps['mlpregressor'], max_iter=10, random_state=0)
    p1, p2 = one_shot_test_parallel(Z, X, M, Y, S, G1=NN_imputer, G2=NN_imputer)
    print("p-values for part 1:", p1)
    print("p-values for part 2:", p2)
    """
    #test Median imputer
    median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    p1, p2 = one_shot_test_parallel(Z, X, M, Y, S, G1=median_imputer, G2=median_imputer)
    print("One-shot test for Fisher's sharp null for Median imputer")
    print("p-values for part 1:", p1)
    print("p-values for part 2:", p2)




    


