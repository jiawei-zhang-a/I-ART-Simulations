import sys
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
from sklearn.impute import SimpleImputer
import Simulation as Generator
import Retrain
import os
import lightgbm as lgb
import xgboost as xgb
import pandas as pd

"""
import psutil

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024) 
"""
beta_coef = None
task_id = 1
save_file = False
max_iter = 3
L = 1000
S_size = 10

def run(Nsize, Single, filepath, adjust, Missing_lambda,strata_size, small_size,verbose=1):

    Missing_lambda = None
    # Create an instance of the OneShot class
    Framework = Retrain.RetrainTest(N = Nsize, covariance_adjustment=adjust)

    print("Begin")

    # Simulate data
    DataGen = Generator.DataGenerator(N = Nsize, strata_size=S_size,beta_11 = beta_coef, beta_12 = beta_coef, beta_21 = beta_coef, beta_22 = beta_coef, beta_23 = beta_coef, beta_31 = beta_coef, beta_32 = beta_coef, MaskRate=0.5,Single=Single, verbose=verbose,Missing_lambda = Missing_lambda)

    X, Z, U, Y, M, S = DataGen.GenerateData()

    X = X - X.mean(axis=0)

    #LR imputer
    if adjust == 0 or adjust == 1:
        print("LR")
        BayesianRidge = IterativeImputer(estimator = linear_model.BayesianRidge(),max_iter=max_iter,random_state=0)
        p_values, reject, test_time = Framework.retrain_test(Z, X, M, Y,strata_size=strata_size, L=L,G=BayesianRidge,verbose=verbose)
        # Append p-values to corresponding lists
        values_LR = [ *p_values, reject, test_time]
        print(test_time)

        # If the folder does not exist, create it
        if not os.path.exists(filepath):
            os.makedirs(filepath)

    #XGBoost
    if adjust == 0 or adjust == 2 or adjust == 3:
        if small_size == True:
            print("Xgboost")
            XGBoost = IterativeImputer(estimator=xgb.XGBRegressor(n_jobs=1), max_iter=max_iter,random_state=0)
            p_values, reject, test_time = Framework.retrain_test(Z, X, M, Y, strata_size = strata_size,L=L, G=XGBoost, verbose=1)
            values_xgboost = [*p_values, reject, test_time]
            print(test_time)

        #LightGBM
        if small_size == False:
            print("LightGBM")
            LightGBM = IterativeImputer(estimator=lgb.LGBMRegressor(n_jobs=1,verbosity=-1), max_iter=max_iter,random_state=0)
            p_values, reject, test_time = Framework.retrain_test(Z, X, M, Y, strata_size=strata_size,L=L, G=LightGBM, verbose=verbose)
            values_lightgbm = [*p_values, reject, test_time]
            print(test_time)

    #Save the file in numpy format
    if(save_file):

        if not os.path.exists("%s/%f"%(filepath,beta_coef)):
            # If the folder does not exist, create it
            os.makedirs("%s/%f"%(filepath,beta_coef))

        # Save numpy arrays to files
        if adjust == 0 or adjust == 1:
            np.save('%s/%f/p_values_LR_%d.npy' % (filepath, beta_coef,task_id), values_LR)

        if adjust == 0 or adjust == 2 or adjust == 3:
            if small_size == False:
                np.save('%s/%f/p_values_lightGBM_%d.npy' % (filepath, beta_coef, task_id), values_lightgbm)
            if small_size == True:
                np.save('%s/%f/p_values_xgboost_%d.npy' % (filepath, beta_coef, task_id), values_xgboost)

if __name__ == '__main__':

    if len(sys.argv) == 2:
        task_id = int(sys.argv[1])
        save_file = True
    else:
        print("Please add the job number like this\nEx.python Power.py 1")
        exit()

    # Define your dictionary here based on the table you've given
    beta_to_lambda = {
        0.0: 16.177150885454697,
        0.1: 16.2694615519701,
        0.2: 16.32049345428536,
        0.3: 16.39758255463026,
        0.4: 16.488028543910794,
        0.5: 16.535288510759447,
    }
    

    for coef in np.arange(0.0,0.6 ,0.1):
        beta_coef = coef
        # Round to two decimal places to match dictionary keys
        beta_coef_rounded = round(beta_coef, 2)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            if beta_coef_rounded == 0.00:
                L = 5000
            else:
                L = 2000
            run(1000, Single = 1, filepath = "Result/HPC_power_1000_unobserved_interference" + "_single", adjust = 0, strata_size = S_size, Missing_lambda = lambda_value, small_size=False)
            run(1000, Single = 1, filepath = "Result/HPC_power_1000_unobserved_interference_adjusted_3" + "_single", adjust = 3, strata_size = S_size, Missing_lambda = lambda_value, small_size=False)
            run(1000, Single = 1, filepath = "Result/HPC_power_1000_unobserved_interference_adjusted_1" + "_single", adjust = 1, strata_size = S_size, Missing_lambda = lambda_value, small_size=False)
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")
    """
    memory_usage = get_memory_usage()
    print(f"Memory usage: {memory_usage:.2f} MB")
    exit()"""
    # Define your dictionary here based on the table you've given
    beta_to_lambda = {
        0.0: 16.12504713635269,
        1: 16.94090023084058,
        2: 17.195256529230846,
        3: 17.69671093992372,
        4: 17.990403650030704,
        5: 18.479387446480725,
    }

    for coef in np.arange(0.0,6,1):
        beta_coef = coef
        # Round to nearest integer to match dictionary keys
        beta_coef_rounded = round(beta_coef)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            if beta_coef_rounded == 0.00:
                L = 4000
            else:
                L = 1000
            run(50, Single = 1, filepath = "Result/HPC_power_50_unobserved_interference_adjusted_2" + "_single", adjust = 2, strata_size = S_size, Missing_lambda = lambda_value,small_size=True)
            run(50, Single = 1, filepath = "Result/HPC_power_50_unobserved_interference_adjusted_1" + "_single", adjust = 1, strata_size = S_size,  Missing_lambda = lambda_value,small_size=True)
            run(50, Single = 1, filepath = "Result/HPC_power_50_unobserved_interference" + "_single", adjust = 0, strata_size = S_size,  Missing_lambda = lambda_value,small_size=True)
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")
