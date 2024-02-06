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
max_iter = 1
L = 2000
S_size = 10

def run(Nsize, Single, filepath, adjust, Missing_lambda,strata_size, small_size,linear_method = 0, verbose=1):

    # Create an instance of the OneShot class
    Framework = Retrain.RetrainTest(N = Nsize, covariance_adjustment=adjust)

    print("Begin")

    # Simulate data
    DataGen = Generator.DataGenerator(N = Nsize, strata_size=S_size,beta_11 = beta_coef, beta_12 = beta_coef, beta_21 = beta_coef, beta_22 = beta_coef, beta_23 = beta_coef, beta_31 = beta_coef, beta_32 = beta_coef,linear_method = linear_method, MaskRate=0.5,Single=Single, verbose=verbose,Missing_lambda = Missing_lambda)

    X, Z, U, Y, M, S = DataGen.GenerateData()

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


    beta_to_lambda = {
        0.0: 14.40094013524747,
        0.25: 14.910720336423797,
        0.5: 15.263145139161315,
        0.75: 15.533261334832284,
        1.0: 15.692701002082288,
        1.25:15.804622540803644
    }
    for coef in np.arange(0.0,1.5,0.25):
        beta_coef = coef
        # Round to two decimal places to match dictionary keys
        beta_coef_rounded = round(beta_coef, 2)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            run(50,Single = 1, filepath = "Result/HPC_power_50_unobserved_nonlinearZ_nonlinearX" + "_single", adjust = 0, linear_method = 2,strata_size = S_size, Missing_lambda = lambda_value, small_size=True)
            run(50,Single = 1, filepath = "Result/HPC_power_50_unobserved_nonlinearZ_nonlinearX_adjusted_2" + "_single", adjust = 2, linear_method = 2,strata_size = S_size, Missing_lambda = lambda_value, small_size=True)
            run(50,Single = 1, filepath = "Result/HPC_power_50_unobserved_nonlinearZ_nonlinearX_adjusted_1" + "_single", adjust = 1, linear_method = 2,strata_size = S_size, Missing_lambda = lambda_value, small_size=True)
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")
    
    beta_to_lambda = {
        0.0: 14.454744752337417,
        0.06: 14.569378860946706,
        0.12: 14.668096269721573,
        0.18: 14.832998887528781,
        0.24: 14.902329149357278,
        0.3: 14.997437855009514
    }
    for coef in np.arange(0.0,0.36,0.06):
        beta_coef = coef
        # Round to two decimal places to match dictionary keys
        beta_coef_rounded = round(beta_coef, 2)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            run(1000,  Single = 1, filepath = "Result/HPC_power_1000_unobserved_nonlinearZ_nonlinearX" + "_single", adjust = 0, linear_method = 2,strata_size = S_size, Missing_lambda = lambda_value, small_size=False)
            run(1000,  Single = 1, filepath = "Result/HPC_power_1000_unobserved_nonlinearZ_nonlinearX_adjusted_3" + "_single", adjust = 3, linear_method = 2,strata_size = S_size, Missing_lambda = lambda_value, small_size=False)
            run(1000,  Single = 1, filepath = "Result/HPC_power_1000_unobserved_nonlinearZ_nonlinearX_adjusted_1" + "_single", adjust = 1, linear_method = 2,strata_size = S_size, Missing_lambda = lambda_value, small_size=False)
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")