import sys
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
import SingleOutcomeModelGenerator as Generator
import MultipleOutcomeModelGenerator as GeneratorMutiple
import RandomizationTest as RandomizationTest
import os
import iArt
import iArt2
import lightgbm as lgb
import xgboost as xgb
import iArt
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


# Do not change this parameter
beta_coef = None
task_id = 1

# Set the default values
max_iter = 3

def run(Nsize, filepath,  Missing_lambda,adjust = 0, model = 0, verbose=1, small_size = True, multiple = False):
    
    Missing_lambda = None

    Iter = 10000


    DataGen = GeneratorMutiple.DataGenerator(N = Nsize, strata_size=10,beta = beta_coef, MaskRate=0.25, verbose=verbose,Missing_lambda = Missing_lambda)
    X, Z, U, Y, M, S = DataGen.GenerateData()

    Framework = RandomizationTest.RandomizationTest(N = Nsize)
    reject, p_values= Framework.test(Z, X, M, Y,strata_size = 10, L=Iter, G = None,verbose=verbose)
    # Append p-values to corresponding lists

    values_oracle = [ *p_values, reject]
    #mask Y with M
    Y = np.ma.masked_array(Y, mask=M)
    Y = Y.filled(np.nan)

    #Median imputer
    print("Median")
    median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    reject, p_values = iArt2.test(Z=Z, X=X, Y=Y,S=S,G=median_imputer,L=Iter, verbose=verbose)
    values_median = [ *p_values, reject ]

    #LR imputer
    print("LR")
    BayesianRidge = IterativeImputer(estimator = linear_model.BayesianRidge(),max_iter=max_iter)
    reject, p_values = iArt.test(Z=Z, X=X, Y=Y,S=S,G=BayesianRidge,L=Iter, verbose=verbose )
    values_LR = [ *p_values, reject ]

    #XGBoost
    if small_size == True:
        print("Xgboost")
        XGBoost = IterativeImputer(estimator=xgb.XGBRegressor(n_jobs=1), max_iter=max_iter)
        reject, p_values = iArt.test(Z=Z, X=X, Y=Y,S=S,G=XGBoost,L=Iter, verbose=verbose)
        values_xgboost = [ *p_values, reject ]

    #LightGBM
    if small_size == False:
        print("LightGBM")
        LightGBM = IterativeImputer(estimator=lgb.LGBMRegressor(n_jobs=1,verbosity=-1), max_iter=max_iter)
        reject, p_values = iArt.test(Z=Z, X=X, Y=Y,S=S,G=LightGBM,L=Iter,verbose=verbose)
        values_lightgbm = [ *p_values, reject ]



    os.makedirs("%s/%f"%(filepath,beta_coef), exist_ok=True)

    # Save numpy arrays to files
    np.save('%s/%f/p_values_median_%d.npy' % (filepath, beta_coef, task_id), values_median)

    np.save('%s/%f/p_values_oracle_%d.npy' % (filepath, beta_coef, task_id), values_oracle)
    np.save('%s/%f/p_values_LR_%d.npy' % (filepath, beta_coef, task_id), values_LR)
    if small_size == True:
        np.save('%s/%f/p_values_xgboost_%d.npy' % (filepath, beta_coef, task_id), values_xgboost)
    else:
        np.save('%s/%f/p_values_lightgbm_%d.npy' % (filepath, beta_coef, task_id), values_lightgbm)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        task_id = int(sys.argv[1])
    else:
        print("Please add the job number like this\nEx.python Power.py 1")
        exit()

    lambda_values = {
        50: {
            0.0: [5.46301136050662, 1.7687104800990539, 3.6986401066938748],
            0.12: [5.507071138438006, 1.8832179319883895, 3.8250507348009557],
            0.24: [5.629938938721568, 1.9080170719416063, 3.870429428753654],
            0.36: [5.709076777442875, 1.9590050193610664, 4.018691917409632],
            0.48: [5.831068183224691, 1.9638039860442473, 4.032046646076915],
            0.6: [5.890152740793354, 2.0340630188325295, 4.188578787477003]
        },
        1000: {
            0.0: [5.445126353777186, 1.7944628138115826, 3.6890049854144222],
            0.03: [5.448889434968681, 1.799820386146107, 3.69899976186121],
            0.06: [5.481108645836731, 1.808888277601773, 3.7215141167626897],
            0.09: [5.518540969761793, 1.8313022068804186, 3.7592034824941227],
            0.12: [5.509295189307611, 1.824491093343858, 3.7653155995566836],
            0.15: [5.5323113856789075, 1.829439262086321, 3.7932522695382818]
        }
    }

    # 1000 size coef loop
    for coef in np.arange(0.2, 0.3, 0.05): 
        beta_coef = coef
        run(1000, filepath="Result/HPC_power_1000_model5",adjust =0,  Missing_lambda=lambda_values[1000].get(coef, None),model = 5, small_size=False, multiple = True)
    # 50 size coef loop
    for coef in np.arange(1.5, 2.5, 0.5): 
        beta_coef = coef
        run(50, filepath="Result/HPC_power_50_model5",adjust =0, Missing_lambda=lambda_values[50].get(coef, None),model = 5, small_size=True, multiple = True)
    