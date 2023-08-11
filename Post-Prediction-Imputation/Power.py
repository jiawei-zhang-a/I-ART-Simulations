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

beta_coef = None
task_id = 1
save_file = False
max_iter = 3
L = 100
S_size = 10

def run(Nsize, Single, filepath, adjust, Missing_lambda,linear_method,strata_size, small_size,verbose=0):

    # If the folder does not exist, create it
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # Create an instance of the OneShot class
    Framework = Retrain.RetrainTest(N = Nsize, covariance_adjustment=adjust)

    print("Begin")

    # Simulate data
    DataGen = Generator.DataGenerator(N = Nsize, strata_size=S_size,beta_11 = beta_coef, beta_12 = beta_coef, beta_21 = beta_coef, beta_22 = beta_coef, beta_23 = beta_coef, beta_31 = beta_coef, beta_32 = beta_coef, MaskRate=0.5,linear_method=linear_method,Single=Single, verbose=verbose,Missing_lambda = Missing_lambda)

    X, Z, U, Y, M, S = DataGen.GenerateData()

    #Oracale imputer
    print("Oracle")
    p_values, reject, test_time = Framework.retrain_test(Z, X, M, Y,strata_size = strata_size, L=L, G = None,verbose=0)
    # Append p-values to corresponding lists
    values_oracle = [ *p_values, reject, test_time]

    #XGBoost
    if small_size == True:
        XGBoost = IterativeImputer(estimator=xgb.XGBRegressor(n_jobs=1), max_iter=max_iter)
        p_values, reject, test_time = Framework.retrain_test(Z, X, M, Y, strata_size = strata_size,L=L, G=XGBoost, verbose=1)
        values_xgboost = [*p_values, reject, test_time]

    #LightGBM
    if small_size == False:
        print("LightGBM")
        LightGBM = IterativeImputer(estimator=lgb.LGBMRegressor(n_jobs=1,verbosity=-1), max_iter=max_iter)
        p_values, reject, test_time = Framework.retrain_test(Z, X, M, Y, strata_size=strata_size,L=L, G=LightGBM, verbose=verbose)
        values_lightgbm = [*p_values, reject, test_time]

    #Save the file in numpy format
    if(save_file):

        if not os.path.exists("%s/%f"%(filepath,beta_coef)):
            # If the folder does not exist, create it
            os.makedirs("%s/%f"%(filepath,beta_coef))

        # Save numpy arrays to files
        np.save('%s/%f/p_values_oracle_%d.npy' % (filepath, beta_coef, task_id), values_oracle)
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
        0.0: 2.190585018478782,
        0.25: 2.4005415180617367,
        0.5: 2.4247574115023114,
        0.75: 2.619155952256185,
        1.0: 2.595153270291314,
        1.25: 2.744281729970429
    }
    for coef in np.arange(0,1.5,0.25):
        beta_coef = coef
        # Round to two decimal places to match dictionary keys
        beta_coef_rounded = round(beta_coef, 2)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            run(50,  Single = 1, filepath = "Result/HPC_power_50_unobserved_linearZ_linearX_adjustment" + "_single", adjust = 1, strata_size = S_size, Missing_lambda = lambda_value, linear_method = 0,small_size=True)
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")
    
    beta_to_lambda = {
        0.0: 2.035857074708517,
        0.07: 2.174988178649473,
        0.14: 2.3387202937400846,
        0.21: 2.3725864425755945,
        0.28: 2.313569659342935,
        0.35: 2.315751089091089
    }
    for coef in np.arange(0.0,0.42,0.07):
        beta_coef = coef
        # Round to two decimal places to match dictionary keys
        beta_coef_rounded = round(beta_coef, 2)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            run(1000, Single = 1, filepath = "Result/HPC_power_1000_unobserved_linearZ_linearX_adjustment" + "_single", adjust = 2, strata_size = S_size, Missing_lambda = lambda_value, linear_method = 0,small_size=False)
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")

    beta_to_lambda = {
        0.0: 14.491811119136337,
        0.8: 14.958941428362772,
        1.6: 15.403478511847414,
        2.4: 15.720791380016868,
        3.2: 15.944599814361716,
        4.0: 16.098830681267856
    }
    for coef in np.arange(0.0,4.8,0.8):
        beta_coef = coef
        # Round to two decimal places to match dictionary keys
        beta_coef_rounded = round(beta_coef, 2)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            run(50, Single = 1, filepath = "Result/HPC_power_50_unobserved_linearZ_nonlinearX_adjustment" + "_single", adjust = 1,strata_size = S_size, Missing_lambda = lambda_value,linear_method = 1, small_size=True)
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")
    
    beta_to_lambda = {
        0.0: 14.363602621057938,
        0.18: 14.603768808707162,
        0.36: 14.59942127025626,
        0.54: 14.721181497335788,
        0.72: 14.907266369483425,
        0.9: 15.043985299562044,
        1.08: 15.072236396242202
    }
    for coef in np.arange(0.0,1.26,0.18):
        beta_coef = coef
        # Round to two decimal places to match dictionary keys
        beta_coef_rounded = round(beta_coef, 2)
        print(beta_coef_rounded)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            run(1000, Single = 1, filepath = "Result/HPC_power_1000_unobserved_linearZ_nonlinearX_adjustment" + "_single", adjust = 2, strata_size = S_size, Missing_lambda = lambda_value,linear_method = 1, small_size=False)
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")
    
    beta_to_lambda = {
        0.0: 14.363917292284167,
        0.25: 14.868591582739715,
        0.5: 15.335728550072929,
        0.75: 15.485908766946375,
        1.0: 15.500897841516423,
        1.25: 15.801413524242948
    }
    for coef in np.arange(0.0,1.5,0.25):
        beta_coef = coef
        # Round to two decimal places to match dictionary keys
        beta_coef_rounded = round(beta_coef, 2)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            run(50,  Single = 1, filepath = "Result/HPC_power_50_unobserved_nonlinearZ_nonlinearX_adjustment" + "_single", adjust = 1,strata_size = S_size, Missing_lambda = lambda_value,linear_method = 2, small_size=True)
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")
    
    beta_to_lambda = {
        0.0: 14.396331546692416,
        0.06: 14.485505560340885,
        0.12: 14.539825257311756,
        0.18: 14.605380751276583,
        0.24: 14.823815159316204,
        0.3: 14.868493955429921
    }
    for coef in np.arange(0.0,0.36,0.06):
        beta_coef = coef
        # Round to two decimal places to match dictionary keys
        beta_coef_rounded = round(beta_coef, 2)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            run(1000, Single = 1, filepath = "Result/HPC_power_1000_unobserved_nonlinearZ_nonlinearX_adjustment" + "_single", adjust = 2, strata_size = S_size, Missing_lambda = lambda_value,linear_method = 2, small_size=False)
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")
    exit()
    # Define your dictionary here based on the table you've given
    beta_to_lambda = {
        0.0: 15.338280233232549,
        0.05: 15.513632949165219,
        0.1: 15.700965399935757,
        0.15: 15.778598987947303,
        0.2: 15.919273976686219,
        0.25: 16.090606547366434,
    }

    for coef in np.arange(0.0,0.3 ,0.05):
        beta_coef = coef
        # Round to two decimal places to match dictionary keys
        beta_coef_rounded = round(beta_coef, 2)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            run(1000, Single = 1, filepath = "Result/HPC_power_1000_unobserved_interference_adjusted" + "_single", adjust = 2, strata_size = S_size, Missing_lambda = lambda_value, small_size=False)
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")

    # Define your dictionary here based on the table you've given
    beta_to_lambda = {
        0.0: 15.64623838541569,
        0.2: 15.914767907195158,
        0.4: 16.139500824890415,
        0.6: 16.744323425885444,
        0.8: 16.996508871283982,
        1.0: 17.340156028716592,
    }

    for coef in np.arange(0.0,1.2,0.2):
        beta_coef = coef
        # Round to nearest integer to match dictionary keys
        beta_coef_rounded = round(beta_coef)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            run(50, Single = 1, filepath = "Result/HPC_power_50_unobserved_interference_adjusted" + "_single", adjust = 1, strata_size = S_size,  Missing_lambda = lambda_value,small_size=True)
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")
