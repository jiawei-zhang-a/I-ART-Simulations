
import xgboost as xgb
import numpy as np
import multiprocessing
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
from sklearn.impute import SimpleImputer
import multiprocessing
import sys
import Simulation as Generator
import OneShot
import warnings

#from cuml import XGBRegressor
 #   XGBRegressor(tree_method='gpu_hist')

if __name__ == '__main__':
    multiprocessing.freeze_support() # This is necessary and important, not sure why 
    # Mask Rate

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="numpy.core.getlimits")

    #Argument
    task_id = 1
    save_file = False

    if len(sys.argv) == 2:
        task_id = int(sys.argv[1])
        save_file = True

    # Create an instance of the OneShot class
    Framework = OneShot.OneShotTest(N = 1000)

    # Simulate data
    DataGen = Generator.DataGenerator(N = 1000, N_T = 500, N_S = 50, beta_11 = 0, beta_12 = 0, beta_21 = 0, beta_22 = 0, beta_23 = 0, beta_31 = 0, MaskRate=0.3,Unobserved=0)

    X, Z, U, Y, M, S = DataGen.GenerateData()

    #Median imputer
    median_imputer_1 = SimpleImputer(missing_values=np.nan, strategy='median')
    median_imputer_2 = SimpleImputer(missing_values=np.nan, strategy='median')
    p11, p12, p21, p22, p31, p32, corr1, corr2, reject = Framework.one_shot_test(Z, X, M, Y, G1=median_imputer_1, G2=median_imputer_2,verbose=1)
    # Append p-values to corresponding lists
    p_values_median = [ p11, p12, p21, p22, p31, p32, corr1[2], corr2[2],reject ]

    #LR imputer
    BayesianRidge_1 = IterativeImputer(estimator = linear_model.BayesianRidge())
    BayesianRidge_2 = IterativeImputer(estimator = linear_model.BayesianRidge())
    p11, p12, p21, p22, p31, p32, corr1, corr2, reject = Framework.one_shot_test(Z, X, M, Y, G1=BayesianRidge_1, G2=BayesianRidge_2,verbose=1)
    # Append p-values to corresponding lists
    p_values_LR = [ p11, p12, p21, p22, p31, p32, corr1[2], corr2[2],reject ]

    #XGBoost
    XGBoost_1= IterativeImputer(estimator = xgb.XGBRegressor())
    XGBoost_2= IterativeImputer(estimator = xgb.XGBRegressor())
    p11, p12, p21, p22, p31, p32, corr1, corr2, reject = Framework.one_shot_test(Z, X, M, Y, G1=XGBoost_1, G2=XGBoost_2,verbose=1)
    # Append p-values to corresponding lists
    p_values_xgboost = [ p11, p12, p21, p22, p31, p32, corr1[2], corr2[2],reject ]

    print("Finished")

    #Save the file in numpy format
    if(save_file):
    # Convert lists to numpy arrays
        p_values_median = np.array(p_values_median)
        p_values_LR = np.array(p_values_LR)
        p_values_xgboost = np.array(p_values_xgboost)
        # Save numpy arrays to files
        np.save('HPC_result/p_values_median_%d.npy' % (task_id), p_values_median)
        np.save('HPC_result/p_values_LR_%d.npy' % (task_id), p_values_LR)
        np.save('HPC_result/p_values_xgboost_%d.npy' % (task_id), p_values_xgboost)      








        



