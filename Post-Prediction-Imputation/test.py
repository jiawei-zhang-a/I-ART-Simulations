
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

    #Iter
    iter = 2

    # level initialization
    level_median = 0
    level_LR = 0
    level_xgboost = 0

    
    
    # Fixed X, Z, change beta to make different Y,M
    for i in range(iter):
        
        print("Iteration: ", i)

        print("Beta: ", task_id)

        print("save_file", save_file)
        # Simulate data
        DataGen = Generator.DataGenerator(N = 1000, N_T = 500, N_S = 50, beta_11 = 0, beta_12 = 0, beta_21 = 0, beta_22 = 0, beta_23 = 0, beta_31 = 0, MaskRate=0.3,Unobserved=0)

        X, Z, U, Y, M, S = DataGen.GenerateData()

        #XGBoost
        XGBoost_1= IterativeImputer(estimator = xgb.XGBRegressor(verbose = 1) ,max_iter=10, random_state=0)
        XGBoost_2= IterativeImputer(estimator = xgb.XGBRegressor(verbose = 1) ,max_iter=10, random_state=0)
        p11, p12, p21, p22, p31, p32, corr1, corr2, reject = Framework.one_shot_test_parallel(Z, X, M, Y, G1=XGBoost_1, G2=XGBoost_2,verbose=1)
        if p31 <= 0.05 or p32 <= 0.05:
            level_xgboost += 1
    
    print("level of Median Imputer: ", level_median/iter)
    print("level of LR Imputer: ", level_LR/iter)
    print("level of XGBoost Imputer: ", level_xgboost/iter)

    #Save the file in numpy format
    if(save_file):
        # Create numpy arrays
        levels = np.array([level_median, level_LR, level_xgboost])

        # Save numpy arrays to files
        np.save('HPC_result/levels_%d.npy'%(task_id), levels)        








        


