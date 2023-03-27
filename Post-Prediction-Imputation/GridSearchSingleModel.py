
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
import Generator
import OneShot


if __name__ == '__main__':
    multiprocessing.freeze_support() # This is necessary and important, not sure why 
    # Mask Rate


    #Simulate Data
    DataGen = Generator.DataGenerator(N = 20000, N_T = 10000, N_S = 100, beta_11 = 1, beta_12 = 1, beta_21 = 1, beta_22 = 1, beta_23 = 1, beta_31 = 1, MaskRate=0.3)

    X, Z, U, Y, M, S = DataGen.GenerateData()

    N = len(X)

    # Create an instance of the OneShot class
    Framework = OneShot.OneShotTest(N)

    #Print the mask situation of M
    print("Mask Rate:")
    M = DataGen.GenerateM(X, U, Y)
    print(M[:,0].sum() / len(M))
    print(M[:,1].sum() / len(M))
    print(M[:,2].sum() / len(M))

    # Fixed X, Z, change beta to make different Y,M
    for i in [0,1,2,5,10,20,50,100]:
        print("beta = ", i)

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
        
        #MissForest
        missForest = IterativeImputer(estimator = RandomForestRegressor(),max_iter=10, random_state=0)
        p1, p2 = Framework.one_shot_test_parallel(Z, X, M, Y, S, G1=missForest, G2=missForest, n_jobs = 12)
        print("One-shot test or Fisher's sharp null for MissForest")
        print("p-values for part 1:", p1)
        print("p-values for part 2:", p2)
    
        #BayesianRidge
        BayesianRidge = IterativeImputer(estimator = linear_model.BayesianRidge(),max_iter=10, random_state=0)
        p1, p2 = Framework.one_shot_test_parallel(Z, X, M, Y, S, G1=BayesianRidge, G2=BayesianRidge, n_jobs = 12)
        print("One-shot test for Fisher's sharp null for BayesianRidge")
        print("p-values for part 1:", p1)
        print("p-values for part 2:", p2)

        #test Median imputer
        median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        p1, p2 = Framework.one_shot_test_parallel(Z, X, M, Y, S, G1=median_imputer, G2=median_imputer, n_jobs = 12)
        print("One-shot test for Fisher's sharp null for Median imputer")
        print("p-values for part 1:", p1)
        print("p-values for part 2:", p2)




        


