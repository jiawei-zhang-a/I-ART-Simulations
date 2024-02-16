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

# Do not change this parameter
beta_coef = None
task_id = 1

# Set the default values
max_iter = 3
L = 1000

def run(Nsize, filepath, adjust, Missing_lambda, strata_size = 10,small_size = 10,model = 0, verbose=1):

    # If the folder does not exist, create it
    if beta_coef == 0:
        Iter = 5000
    else:
        Iter = L 
    

    "asddasd"
    Iter = 50
    max_iter = 1
    Missing_lambda = None
    "asddasd"

    # Create an instance of the OneShot class
    Framework = Retrain.RetrainTest(N = Nsize, covariance_adjustment=adjust)

    # Simulate data
    DataGen = Generator.DataGenerator(N = Nsize, strata_size=strata_size,beta = beta_coef,model = model, MaskRate=0.5, verbose=verbose,Missing_lambda = Missing_lambda)

    X, Z, U, Y, M, S = DataGen.GenerateData()


    #LR imputer
    if adjust == 0 or adjust == 1:
        print("LR")
        BayesianRidge = IterativeImputer(estimator = linear_model.LinearRegression(),max_iter=max_iter,random_state=0)
        p_values, reject, test_time = Framework.retrain_test(Z, X, M, Y,strata_size=strata_size, L=Iter,G=BayesianRidge,verbose=verbose)
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
            p_values, reject, test_time = Framework.retrain_test(Z, X, M, Y, strata_size = strata_size,L=Iter, G=XGBoost, verbose=1)
            values_xgboost = [*p_values, reject, test_time]
            print(test_time)

        #LightGBM
        if small_size == False:
            print("LightGBM")
            LightGBM = IterativeImputer(estimator=lgb.LGBMRegressor(n_jobs=1,verbosity=-1), max_iter=max_iter,random_state=0)
            p_values, reject, test_time = Framework.retrain_test(Z, X, M, Y, strata_size=strata_size,L=Iter, G=LightGBM, verbose=verbose)
            values_lightgbm = [*p_values, reject, test_time]
            print(test_time)

    #Save the file in numpy format

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
        """
    # Model 1
    beta_to_lambda = {0.0: 2.159275141001102, 0.07: 2.165387531267955, 0.14: 2.285935405246937, 0.21: 2.258923945496463, 0.28: 2.2980720651301794, 0.35: 2.3679216299985613}
    for coef in np.arange(0.0,0.42,0.07):
        beta_coef = coef
        # Round to two decimal places to match dictionary keys
        beta_coef_rounded = round(beta_coef, 2)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            run(1000, filepath = "Result/HPC_power_1000_model1", adjust = 0, model = 1, Missing_lambda = lambda_value, small_size=False)
            run(1000, filepath = "Result/HPC_power_1000_model1_adjusted_LightGBM", adjust = 3, model = 1, Missing_lambda = lambda_value, small_size=False)
            run(1000, filepath = "Result/HPC_power_1000_model1_adjusted_LR", adjust = 1, model = 1, Missing_lambda = lambda_value, small_size=False)
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")

    beta_to_lambda = {0.0: 2.1577587265653126, 0.25: 2.2946233479956843, 0.5: 2.42339283727788, 0.75: 2.544154767644711, 1.0: 2.669166349074493, 1.25: 2.792645016605368}
    for coef in np.arange(0,1.5,0.25):
        beta_coef = coef
        # Round to two decimal places to match dictionary keys
        beta_coef_rounded = round(beta_coef, 2)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            run(50, filepath = "Result/HPC_power_50_model1", adjust = 0, model = 1, Missing_lambda = lambda_value, small_size=True)
            run(50, filepath = "Result/HPC_power_50_model1_adjusted_Xgboost", adjust = 2, model = 1, Missing_lambda = lambda_value, small_size=True)
            run(50, filepath = "Result/HPC_power_50_model1_adjusted_LR", adjust = 1, model = 1,Missing_lambda = lambda_value, small_size=True)
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")
    """
    # Model 2
    beta_to_lambda = {0.0: 14.376830203817725, 0.16: 14.492781397662549, 0.32: 14.636259203432914, 0.48: 14.790662640235277, 0.64: 14.902477227186191, 0.8: 14.995429287214796}
    for coef in np.arange(0.0,0.96,0.16):
        beta_coef = coef
        # Round to two decimal places to match dictionary keys
        beta_coef_rounded = round(beta_coef, 2)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            run(1000, filepath = "Result/HPC_power_1000_model2", adjust = 0, model = 2, Missing_lambda = lambda_value, small_size=False)
            run(1000, filepath = "Result/HPC_power_1000_model2_adjusted_LightGBM", adjust = 3, model = 2, Missing_lambda = lambda_value, small_size=False)
            run(1000, filepath = "Result/HPC_power_1000_model2_adjusted_LR", adjust = 1, model = 2, Missing_lambda = lambda_value, small_size=False)
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")

    beta_to_lambda = {0.0: 14.335704307321487, 0.8: 14.971101330156632, 1.6: 15.366375386649604, 2.4: 15.724510367662774, 3.2: 15.831265197313604, 4.0: 16.011150941155087}
    for coef in np.arange(0.0,4.8,0.8):
        beta_coef = coef
        # Round to two decimal places to match dictionary keys
        beta_coef_rounded = round(beta_coef, 2)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            run(50, filepath = "Result/HPC_power_50_model2", adjust = 0, model = 2, Missing_lambda = lambda_value, small_size=True)
            run(50, filepath = "Result/HPC_power_50_model2_adjusted_Xgboost", adjust = 2, model = 2, Missing_lambda = lambda_value, small_size=True)
            run(50, filepath = "Result/HPC_power_50_model2_adjusted_LR", adjust = 1, model = 2,Missing_lambda = lambda_value, small_size=True)
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")
          
    # Model 3
    beta_to_lambda = {0.0: 14.43119646829717, 0.06: 14.530199258897895, 0.12: 14.74631511872901, 0.18: 14.90822250678929, 0.24: 14.947558384606348, 0.3: 14.979303880359883}
    for coef in np.arange(0.0,0.36,0.06):
        beta_coef = coef
        # Round to two decimal places to match dictionary keys
        beta_coef_rounded = round(beta_coef, 2)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            run(1000, filepath = "Result/HPC_power_1000_model3", adjust = 0, model = 3, Missing_lambda = lambda_value, small_size=False)
            run(1000, filepath = "Result/HPC_power_1000_model3_adjusted_LightGBM", adjust = 3, model = 3, Missing_lambda = lambda_value, small_size=False)
            run(1000, filepath = "Result/HPC_power_1000_model3_adjusted_LR", adjust = 1, model = 3, Missing_lambda = lambda_value, small_size=False)
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")

    beta_to_lambda = {0.0: 14.390422997290685, 0.25: 14.909362702476912, 0.5: 15.219787798636258, 0.75: 15.4701421427122, 1.0: 15.625851714156521, 1.25: 15.831324938012127}
    for coef in np.arange(0.0,1.5,0.25):
        beta_coef = coef
        # Round to two decimal places to match dictionary keys
        beta_coef_rounded = round(beta_coef, 2)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            run(50, filepath = "Result/HPC_power_50_model3", adjust = 0, model = 3, Missing_lambda = lambda_value, small_size=True)
            run(50, filepath = "Result/HPC_power_50_model3_adjusted_Xgboost", adjust = 2, model = 3, Missing_lambda = lambda_value, small_size=True)
            run(50, filepath = "Result/HPC_power_50_model3_adjusted_LR", adjust = 1, model = 3,Missing_lambda = lambda_value, small_size=True)
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")
    """
    # Model 4
    beta_to_lambda = {0.0: 15.359698674885047, 0.06: 15.507224279021253, 0.12: 15.675599389006583, 0.18: 15.744503702370242, 0.24: 15.778177240810757, 0.3: 15.8935570369039}
    for coef in np.arange(0.0,0.36,0.06):
        beta_coef = coef
        # Round to two decimal places to match dictionary keys
        beta_coef_rounded = round(beta_coef, 2)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            run(1000, filepath = "Result/HPC_power_1000_model4", adjust = 0, model = 4, Missing_lambda = lambda_value, small_size=False)
            run(1000, filepath = "Result/HPC_power_1000_model4_adjusted_LightGBM", adjust = 3, model = 4, Missing_lambda = lambda_value, small_size=False)
            run(1000, filepath = "Result/HPC_power_1000_model4_adjusted_LR", adjust = 1, model = 4, Missing_lambda = lambda_value, small_size=False)
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")

    beta_to_lambda = {0.0: 15.391438190996098, 0.25: 15.871854826261341, 0.5: 16.34293913102228, 0.75: 16.45643215605396, 1.0: 16.556851722322666, 1.25: 16.760752915013537}
    for coef in np.arange(0.0,1.5,0.25):
        beta_coef = coef
        # Round to two decimal places to match dictionary keys
        beta_coef_rounded = round(beta_coef, 2)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            run(50, filepath = "Result/HPC_power_50_model4", adjust = 0, model = 4, Missing_lambda = lambda_value, small_size=True)
            run(50, filepath = "Result/HPC_power_50_model4_adjusted_Xgboost", adjust = 2, model = 4, Missing_lambda = lambda_value, small_size=True)
            run(50, filepath = "Result/HPC_power_50_model4_adjusted_LR", adjust = 1, model = 4,Missing_lambda = lambda_value, small_size=True)
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")
    
    
    # Model 6
    beta_to_lambda = {0.0: 15.52272711345184, 0.1: 15.686703500976, 0.2: 15.686402633876, 0.3: 15.787598335083226, 0.4: 15.753018503387455, 0.5: 15.73965750718643}
    for coef in np.arange(0.0,0.6 ,0.1):
        beta_coef = coef
        # Round to two decimal places to match dictionary keys
        beta_coef_rounded = round(beta_coef, 2)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            run(1000, filepath = "Result/HPC_power_1000_model6", adjust = 0, model = 6, Missing_lambda = lambda_value, small_size=False)
            run(1000, filepath = "Result/HPC_power_1000_model6_adjusted_LightGBM", adjust = 3, model = 6, Missing_lambda = lambda_value, small_size=False)
            run(1000, filepath = "Result/HPC_power_1000_model6_adjusted_LR", adjust = 1, model = 6, Missing_lambda = lambda_value, small_size=False)
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")

    beta_to_lambda = {0.0: 15.583775008005304, 3.0: 16.2044899667755, 6.0: 16.364986769719895, 9.0: 16.572385216230238, 12.0: 16.508220779651012, 15.0: 16.572190364153975}
    for coef in np.arange(0.0,18,3):
        beta_coef = coef
        # Round to two decimal places to match dictionary keys
        beta_coef_rounded = round(beta_coef, 2)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            run(50, filepath = "Result/HPC_power_50_model6", adjust = 0, model = 6, Missing_lambda = lambda_value, small_size=True)
            run(50, filepath = "Result/HPC_power_50_model6_adjusted_Xgboost", adjust = 2, model = 6, Missing_lambda = lambda_value, small_size=True)
            run(50, filepath = "Result/HPC_power_50_model6_adjusted_LR", adjust = 1, model = 6,Missing_lambda = lambda_value, small_size=True)
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")
    """