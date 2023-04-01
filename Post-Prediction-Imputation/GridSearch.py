
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn import linear_model
from sklearn.impute import SimpleImputer
import multiprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.kernel_approximation import Nystroem
import Simulation as Generator
import OneShot
import warnings

#from cuml import XGBRegressor
 #   XGBRegressor(tree_method='gpu_hist')



if __name__ == '__main__':
    multiprocessing.freeze_support() # This is necessary and important, not sure why 
    # Mask Rate

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    #Simulate Data
    DataGen = Generator.DataGenerator(N = 1000, N_T = 500, N_S = 100, beta_11 = 1, beta_12 = 1, beta_21 = 1, beta_22 = 1, beta_23 = 1, beta_31 = 1, MaskRate=0.3,Unobserved=0)

    X, Z, U, Y, M, S = DataGen.GenerateData()

    N = len(X)

    # Create an instance of the OneShot class
    Framework = OneShot.OneShotTest(N)

    #Print the mask situation of M
    print("Mask Rate: \n", DataGen.MaskRate)

    # Fixed X, Z, change beta to make different Y,M
    for i in [0,1,5,20,50,100]:

        # Change the parameters beta
        DataGen.beta_11 = i
        DataGen.beta_12 = i
        DataGen.beta_21 = i
        DataGen.beta_22 = i
        DataGen.beta_23 = i
        DataGen.beta_31 = i
        
        # Simulate data
        Y = DataGen.GenerateY(X, U, Z)
        M = DataGen.GenerateM(X, U, Y)

        #test Median imputer
        median_imputer_1 = SimpleImputer(missing_values=np.nan, strategy='median')
        median_imputer_2 = SimpleImputer(missing_values=np.nan, strategy='median')
        p11, p12, p21, p22, p31, p32, corr1, corr2 = Framework.one_shot_test(Z, X, M, Y, G1=median_imputer_1, G2=median_imputer_2,verbose=0)
        print("One-shot test for Fisher's sharp null for Median Imputer")
        print("beta = ", i)
        print("p-values for part 1:", p11,p21,p31)
        print("p-values for part 2:", p12,p22,p32)
        print("corr1, corr2", corr1, corr2)

        #XGBoost
        XGBRegressor_1 = xgb.XGBRegressor()
        XGBRegressor_2 = xgb.XGBRegressor()

        XGBoost_1= IterativeImputer(estimator = XGBRegressor_1 ,max_iter=10, random_state=0)
        XGBoost_2= IterativeImputer(estimator = XGBRegressor_2 ,max_iter=10, random_state=0)
        p11, p12, p21, p22, p31, p32, corr1, corr2 = Framework.one_shot_test(Z, X, M, Y, G1=XGBoost_1, G2=XGBoost_2,verbose=0)
        print("One-shot test for Fisher's sharp null for XGBoost")
        print("beta = ", i)
        print("p-values for part 1:", p11,p21,p31)
        print("p-values for part 2:", p12,p22,p32)
        print("corr1, corr2", corr1, corr2)
        









        


