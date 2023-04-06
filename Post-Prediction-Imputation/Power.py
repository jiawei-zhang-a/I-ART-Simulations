import sys
import xgboost as xgb
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
from sklearn.impute import SimpleImputer
import multiprocessing
import Simulation as Generator
import OneShot
import warnings

#from cuml import XGBRegressor
 #   XGBRegressor(tree_method='gpu_hist')



if __name__ == '__main__':
    multiprocessing.freeze_support() # This is necessary and important, not sure why 
    # Mask Rate

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    beta_coef = 1
    task_id = 1
    save_file = False

    if len(sys.argv) == 3:
        beta_coef = int(sys.argv[1])
        task_id = int(sys.argv[2])
        save_file = True

    # Create an instance of the OneShot class
    Framework = OneShot.OneShotTest(N = 1000)

    # power initialization
    power_median = 0
    power_LR = 0
    power_xgboost = 0

    #iteration 
    iter = 10

    # correlation initialization
    corr_median = np.zeros(iter)
    corr_LR = np.zeros(iter)
    corr_xgboost = np.zeros(iter)

    # Fixed X, Z, change beta to make different Y,M
    for i in range(iter):
        
        print("Iteration: ", i)
        # Simulate data
        DataGen = Generator.DataGenerator(N = 1000, N_T = 500, N_S = 50, beta_11 = beta_coef, beta_12 = beta_coef, beta_21 = beta_coef, beta_22 = beta_coef, beta_23 = beta_coef, beta_31 = beta_coef, MaskRate=0.3,Unobserved=0)

        X, Z, U, Y, M, S = DataGen.GenerateData()

        #test Median imputer
        median_imputer_1 = SimpleImputer(missing_values=np.nan, strategy='median')
        median_imputer_2 = SimpleImputer(missing_values=np.nan, strategy='median')
        p11, p12, p21, p22, p31, p32, corr1, corr2, reject = Framework.one_shot_test_parallel(Z, X, M, Y, G1=median_imputer_1, G2=median_imputer_2,verbose=0)
        power_median += reject
        corr_median[i] = (corr1[2] + corr2[2]) / 2

        #test LR imputer
        BayesianRidge_1 = IterativeImputer(estimator = linear_model.BayesianRidge(),max_iter=10, random_state=0)
        BayesianRidge_2 = IterativeImputer(estimator = linear_model.BayesianRidge(),max_iter=10, random_state=0)
        p11, p12, p21, p22, p31, p32, corr1, corr2, reject = Framework.one_shot_test_parallel(Z, X, M, Y, G1=BayesianRidge_1, G2=median_imputer_2,verbose=0)
        power_LR += reject
        corr_LR[i] = (corr1[2] + corr2[2]) / 2

        #XGBoost
        XGBoost_1= IterativeImputer(estimator = xgb.XGBRegressor() ,max_iter=10, random_state=0)
        XGBoost_2= IterativeImputer(estimator = xgb.XGBRegressor() ,max_iter=10, random_state=0)
        p11, p12, p21, p22, p31, p32, corr1, corr2, reject = Framework.one_shot_test_parallel(Z, X, M, Y, G1=XGBoost_1, G2=XGBoost_2,verbose=0)
        power_xgboost += reject
        corr_xgboost[i] = (corr1[2] + corr2[2]) / 2
    
    #Write result into the file
    print("Correlation of Median Imputer: " + str(np.mean(corr_median)))
    print("Correlation of LR Imputer: " + str(np.mean(corr_LR)))
    print("Correlation of XGBoost Imputer: " + str(np.mean(corr_xgboost)))
    
    print("Power of Median Imputer: " + str(power_median/iter))
    print("Power of LR Imputer: " + str(power_LR/iter))
    print("Power of XGBoost Imputer: " + str(power_xgboost/iter))

    #Save the file in numpy format
    if(save_file):
        # Create numpy arrays
        correlations = np.array([corr_median, corr_LR, corr_xgboost])
        powers = np.array([power_median, power_LR, power_xgboost])

        # Save numpy arrays to files
        np.save('HPC_result/correlations_%d_%d.npy'%(beta_coef,task_id), correlations)
        np.save('HPC_result/powers_%d_%d.npy'%(beta_coef,task_id), powers)        
    #file.close()







        


