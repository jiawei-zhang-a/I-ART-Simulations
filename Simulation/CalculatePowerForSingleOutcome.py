import sys
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
import SingleOutcomeModelGenerator as Generator
import RandomizationTest as RandomizationTest
import os
from statsmodels.stats.multitest import multipletests
import lightgbm as lgb
import xgboost as xgb
import iArt as iArtMain
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


def holm_bonferroni(p_values, alpha = 0.05):
    """
    Perform the Holm-Bonferroni correction on the p-values
    """

    # Perform the Holm-Bonferroni correction
    reject, corrected_p_values, _, _ = multipletests(p_values, alpha=alpha, method='holm')

    # Check if any null hypothesis can be rejected
    any_rejected = any(reject)

    return any_rejected


# Do not change this parameter
beta_coef = None
task_id = 1

# Set the default values
max_iter = 3



def run(Nsize, filepath, Missing_lambda, adjust=0, model=0, verbose=0, small_size=True, multiple=False):
    Missing_lambda = None

    if beta_coef == 0:
        Iter = 10000
    else:
        Iter = 100

    # Simulate data
    DataGen = Generator.DataGenerator(N=Nsize, strata_size=10, beta=beta_coef, model=model, MaskRate=0.5, verbose=verbose, Missing_lambda=Missing_lambda)
    X, Z, U, Y, M, S = DataGen.GenerateData()

    Framework = RandomizationTest.RandomizationTest(N=Nsize)

    # Oracle method
    elapsed_time, t_obs, t_sim = Framework.test(Z, X, M, Y, strata_size=10, L=Iter, G=None, verbose=verbose)

    # Compute p-values and rejection decisions
    p_values = np.mean(t_sim >= t_obs, axis=0)
    reject = holm_bonferroni(p_values, alpha=0.05)

    # Create a dictionary to store all results
    results_oracle = {
        'elapsed_time': elapsed_time,
        't_obs': t_obs,
        't_sim': t_sim,
        'p_values': p_values,
        'reject': reject
    }

    # Median imputer
    print("Median")
    median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    elapsed_time, t_obs, t_sim = Framework.test_imputed(Z=Z, X=X, Y=Y,M=M, G=median_imputer,L=Iter, verbose=verbose)

    p_values = np.mean(t_sim >= t_obs, axis=0)
    reject = holm_bonferroni(p_values, alpha=0.05)

    results_median = {
        'elapsed_time': elapsed_time,
        't_obs': t_obs,
        't_sim': t_sim,
        'p_values': p_values,
        'reject': reject
    }
    print(t_sim)


    # Mask Y with M
    Y = np.ma.masked_array(Y, mask=M)
    Y = Y.filled(np.nan)

    # Linear Regression imputer
    print("LR")
    BayesianRidge = IterativeImputer(estimator=linear_model.BayesianRidge(), max_iter=max_iter)
    elapsed_time, t_obs, t_sim = iArtMain.test(Z=Z, X=X, Y=Y, S=S, G=BayesianRidge, L=Iter, verbose=verbose)

    p_values = np.mean(t_sim >= t_obs, axis=0)
    reject = holm_bonferroni(p_values, alpha=0.05)

    results_LR = {
        'elapsed_time': elapsed_time,
        't_obs': t_obs,
        't_sim': t_sim,
        'p_values': p_values,
        'reject': reject
    }

    # XGBoost
    if small_size:
        print("XGBoost")
        XGBoost = IterativeImputer(estimator=xgb.XGBRegressor(n_jobs=1), max_iter=max_iter)
        elapsed_time, t_obs, t_sim = iArtMain.test(Z=Z, X=X, Y=Y, S=S, G=XGBoost, L=Iter, verbose=verbose)

        p_values = np.mean(t_sim >= t_obs, axis=0)
        reject = holm_bonferroni(p_values, alpha=0.05)

        results_xgboost = {
            'elapsed_time': elapsed_time,
            't_obs': t_obs,
            't_sim': t_sim,
            'p_values': p_values,
            'reject': reject
        }

    # LightGBM
    else:
        print("LightGBM")
        LightGBM = IterativeImputer(estimator=lgb.LGBMRegressor(n_jobs=1, verbosity=-1), max_iter=max_iter)
        elapsed_time, t_obs, t_sim = iArtMain.test(Z=Z, X=X, Y=Y, S=S, G=LightGBM, L=Iter, verbose=verbose)

        p_values = np.mean(t_sim >= t_obs, axis=0)
        reject = holm_bonferroni(p_values, alpha=0.05)

        results_lightgbm = {
            'elapsed_time': elapsed_time,
            't_obs': t_obs,
            't_sim': t_sim,
            'p_values': p_values,
            'reject': reject
        }

    os.makedirs(f"{filepath}/{beta_coef}", exist_ok=True)

    # Save the results dictionaries to files
    np.save(f'{filepath}/{beta_coef}/results_oracle_{task_id}.npy', results_oracle)
    np.save(f'{filepath}/{beta_coef}/results_median_{task_id}.npy', results_median)
    np.save(f'{filepath}/{beta_coef}/results_LR_{task_id}.npy', results_LR)
    if small_size:
        np.save(f'{filepath}/{beta_coef}/results_xgboost_{task_id}.npy', results_xgboost)
    else:
        np.save(f'{filepath}/{beta_coef}/results_lightgbm_{task_id}.npy', results_lightgbm)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        task_id = int(sys.argv[1])
    else:
        print("Please add the job number like this\nEx.python Power.py 1")
        exit()
    # Model 1
    beta_to_lambda = {0.0: 2.159275141001102, 0.07: 2.165387531267955, 0.14: 2.285935405246937, 0.21: 2.258923945496463, 0.28: 2.2980720651301794, 0.35: 2.3679216299985613}
    for coef in np.arange(0.0,0.42,0.07):
        beta_coef = coef
        # Round to two decimal places to match dictionary keys
        beta_coef_rounded = round(beta_coef, 2)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            run(1000, filepath = "Result/HPC_power_1000_model1", adjust = 0, model = 1, Missing_lambda = lambda_value, small_size=False)
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
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")
    
    # Model 2
    beta_to_lambda = {0.0: 14.376830203817725, 0.16: 14.492781397662549, 0.32: 14.636259203432914, 0.48: 14.790662640235277, 0.64: 14.902477227186191, 0.8: 14.995429287214796}
    for coef in np.arange(0.0,0.96,0.16):
        beta_coef = coef
        # Round to two decimal places to match dictionary keys
        beta_coef_rounded = round(beta_coef, 2)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            run(1000, filepath = "Result/HPC_power_1000_model2", adjust = 0, model = 2, Missing_lambda = lambda_value, small_size=False)
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
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")
    
    # Model 4
    beta_to_lambda = {0.0: 15.359698674885047, 0.06: 15.507224279021253, 0.12: 15.675599389006583, 0.18: 15.744503702370242, 0.24: 15.778177240810757, 0.3: 15.8935570369039}
    for coef in np.arange(0.0,0.36,0.06):
        beta_coef = coef
        # Round to two decimal places to match dictionary keys
        beta_coef_rounded = round(beta_coef, 2)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            run(1000, filepath = "Result/HPC_power_1000_model4", adjust = 0, model = 4, Missing_lambda = lambda_value, small_size=False)
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
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")

    beta_to_lambda = {0.0: 15.52272711345184, 0.1: 15.686703500976, 0.2: 15.686402633876, 0.3: 15.787598335083226, 0.4: 15.753018503387455, 0.5: 15.73965750718643}
    for coef in np.arange(0.0,0.6 ,0.1):
        beta_coef = coef
        # Round to two decimal places to match dictionary keys
        beta_coef_rounded = round(beta_coef, 2)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            run(Nsize = 1000, filepath = "Result/HPC_power_1000_model6",adjust =0,  model = 6, Missing_lambda = lambda_value, small_size=False)
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")

    beta_to_lambda = {0.0: 15.583775008005304, 3.0: 16.2044899667755, 6.0: 16.364986769719895, 9.0: 16.572385216230238, 12.0: 16.508220779651012, 15.0: 16.572190364153975}
    for coef in np.arange(0.0,18,3):
        beta_coef = coef
        # Round to two decimal places to match dictionary keys
        beta_coef_rounded = round(beta_coef, 2)
        if beta_coef_rounded in beta_to_lambda:
            lambda_value = beta_to_lambda[beta_coef_rounded]
            run(Nsize = 50, filepath = "Result/HPC_power_50_model6",adjust =0,  model = 6, Missing_lambda = lambda_value, small_size=True)
        else:
            print(f"No lambda value found for beta_coef: {beta_coef_rounded}")