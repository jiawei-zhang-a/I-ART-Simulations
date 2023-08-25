import sys
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
from sklearn.impute import SimpleImputer
import simulation as Generator
from causalpart import preptest
import os
import lightgbm as lgb
import xgboost as xgb

beta_coef = None
task_id = 1
save_file = False
max_iter = 3
L = 10
S_size = 10

def run(Nsize, Unobserved, Single, filepath, adjust, linear_method, Missing_lambda,strata_size, small_size,verbose=1):

    # If the folder does not exist, create it
    if not os.path.exists(filepath):
        os.makedirs(filepath)


    # Simulate data
    DataGen = Generator.DataGenerator(N = Nsize, strata_size=S_size,beta_11 = beta_coef, beta_12 = beta_coef, beta_21 = beta_coef, beta_22 = beta_coef, beta_23 = beta_coef, beta_31 = beta_coef, beta_32 = beta_coef, MaskRate=0.5,Unobserved=Unobserved, Single=Single, linear_method = linear_method,verbose=verbose,Missing_lambda = Missing_lambda)

    X, Z, U, Y, M, S = DataGen.GenerateData()

    #Median imputer
    median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    reject, p_values  = preptest(Z = Z,X = X, Y=Y, M = M, S=S,L=L,  G = 'median',verbose=verbose, covariate_adjustment = True)
    reject, p_values  = preptest(Z = Z,X = X, Y=Y, M = M, S=S,L=L,  G = 'median',verbose=verbose, covariate_adjustment = False)

    # Append p-values to corresponding lists
    values_median = [ *p_values, reject]

    #LR imputer
    BayesianRidge = IterativeImputer(estimator = linear_model.BayesianRidge(),max_iter=max_iter)
    reject, p_values  = preptest(Z = Z,X = X, Y=Y, M = M, S=S,L=L,  G = 'bayesianridge',verbose=verbose, covariate_adjustment = True)
    reject, p_values  = preptest(Z = Z,X = X, Y=Y, M = M, S=S,L=L,  G = 'bayesianridge',verbose=verbose, covariate_adjustment = False)

    # Append p-values to corresponding lists
    values_LR = [ *p_values, reject]

    #XGBoost
    if small_size == True:
        XGBoost = IterativeImputer(estimator=xgb.XGBRegressor(), max_iter=max_iter)
        reject, p_values  = preptest(Z = Z,X = X, Y=Y, M = M, S=S,L=L,  G = 'xgboost',verbose=verbose, covariate_adjustment = True)
        reject, p_values  = preptest(Z = Z,X = X, Y=Y, M = M, S=S,L=L,  G = 'xgboost',verbose=verbose, covariate_adjustment = False)

        values_xgboost = [*p_values, reject]

    #LightGBM
    if small_size == False:
        LightGBM = IterativeImputer(estimator=lgb.LGBMRegressor( verbosity=-1), max_iter=max_iter)
        reject, p_values  = preptest(Z = Z,X = X, Y=Y, M = M, S=S,L=L,  G = 'lightgbm',verbose=verbose, covariate_adjustment = True)
        reject, p_values  = preptest(Z = Z,X = X, Y=Y, M = M, S=S,L=L,  G = 'lightgbm',verbose=verbose, covariate_adjustment = False)

        values_lightgbm = [*p_values, reject]

    #Save the file in numpy format
    if(save_file):

        if not os.path.exists("%s/%f"%(filepath,beta_coef)):
            # If the folder does not exist, create it
            os.makedirs("%s/%f"%(filepath,beta_coef))

        # Save numpy arrays to files
        np.save('%s/%f/p_values_median_%d.npy' % (filepath, beta_coef, task_id), values_median)
        np.save('%s/%f/p_values_LR_%d.npy' % (filepath, beta_coef,task_id), values_LR)
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

    if os.path.exists("Result") == False:
        os.mkdir("Result")

    # Define your dictionary here based on the table you've given
    beta_to_lambda = {
        0.0: 15.338280233232549,
        0.05: 15.513632949165219,
        0.1: 15.700965399935757,
        0.15: 15.778598987947303,
        0.2: 15.919273976686219,
        0.25: 16.090606547366434,
    }

    for coef in np.arange(0.1,0.3 ,0.05):
        beta_coef = coef
        # Round to two decimal places to match dictionary keys
        beta_coef_rounded = round(beta_coef, 2)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            run(1000, Unobserved = 1, Single = 1, filepath = "Result/HPC_power_1000_unobserved_interference" + "_single", adjust = 0, linear_method = 2,strata_size = S_size, Missing_lambda = lambda_value, small_size=False)
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
            run(50, Unobserved = 1, Single = 1, filepath = "Result/HPC_power_50_unobserved_interference" + "_single", adjust = 0, linear_method = 2,strata_size = S_size,  Missing_lambda = lambda_value,small_size=True)
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")
