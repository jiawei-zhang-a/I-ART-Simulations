
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
    for i in [1,2,5,10,20,50,100]:
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
        missForest_1 = IterativeImputer(estimator = RandomForestRegressor(),max_iter=10, random_state=0)
        missForest_2 = IterativeImputer(estimator = RandomForestRegressor(),max_iter=10, random_state=0)
        p1, p2 = Framework.one_shot_test_parallel(Z, X, M, Y, S, G1=missForest_1, G2=missForest_2, n_jobs = 12)
        print("One-shot test for Fisher's sharp null for MissForest")
        print("p-values for part 1:", p1)
        print("p-values for part 2:", p2)
        
        #KNN
        KNNimputer_1 = KNNImputer(n_neighbors=4)
        KNNimputer_2 = KNNImputer(n_neighbors=4)
        p1, p2 = Framework.one_shot_test_parallel(Z, X, M, Y, S, G1=KNNimputer_1, G2=KNNimputer_2, n_jobs = 12)
        print("One-shot test for Fisher's sharp null for KNN imputer")
        print("p-values for part 1:", p1)
        print("p-values for part 2:", p2)

        #BayesianRidge
        BayesianRidge_1 = IterativeImputer(estimator = linear_model.BayesianRidge(),max_iter=10, random_state=0)
        BayesianRidge_2 = IterativeImputer(estimator = linear_model.BayesianRidge(),max_iter=10, random_state=0)
        p1, p2 = Framework.one_shot_test_parallel(Z, X, M, Y, S, G1=BayesianRidge_1, G2=BayesianRidge_2, n_jobs = 12)
        print("One-shot test for Fisher's sharp null for BayesianRidge")
        print("p-values for part 1:", p1)
        print("p-values for part 2:", p2)

        #Nystroem Method for Kernel Approximation
        pipeline = make_pipeline(
            StandardScaler(),
            Nystroem(), 
            linear_model.Ridge()
        )
        NystroemKernel_1 = IterativeImputer(estimator = pipeline,max_iter=10, random_state=0)
        NystroemKernel_2 = IterativeImputer(estimator = pipeline,max_iter=10, random_state=0)
        p1, p2 = Framework.one_shot_test_parallel(Z, X, M, Y, S, G1=NystroemKernel_1, G2=NystroemKernel_2, n_jobs = 12)
        print("One-shot test for Fisher's sharp null for Nystroem Kernel Approximation")
        print("p-values for part 1:", p1)
        print("p-values for part 2:", p2)

        #XGBoost
        XGBoost_1 = IterativeImputer(estimator = xgb.XGBRegressor(),max_iter=10, random_state=0)
        XGBoost_2 = IterativeImputer(estimator = xgb.XGBRegressor(),max_iter=10, random_state=0)
        p1, p2 = Framework.one_shot_test_parallel(Z, X, M, Y, S, G1=XGBoost_1, G2=XGBoost_2, n_jobs = 12)
        print("One-shot test for Fisher's sharp null for XGBoost")
        print("p-values for part 1:", p1)
        print("p-values for part 2:", p2)

        #Neural Network
        print("One-shot test for Fisher's sharp null for Neural Network")
        pipeline = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(100, 100, 100,100), activation='relu', alpha=0.0001, random_state=0)
        )
        NN_imputer_1 = IterativeImputer(estimator=pipeline.named_steps['mlpregressor'], max_iter=10, random_state=0)
        NN_imputer_2 = IterativeImputer(estimator=pipeline.named_steps['mlpregressor'], max_iter=10, random_state=0)
        p1, p2 = Framework.one_shot_test_parallel(Z, X, M, Y, S, G1=NN_imputer_1, G2=NN_imputer_2, n_jobs = 12)
        print("p-values for part 1:", p1)
        print("p-values for part 2:", p2)

        #test Median imputer
        median_imputer_1 = SimpleImputer(missing_values=np.nan, strategy='median')
        median_imputer_2 = SimpleImputer(missing_values=np.nan, strategy='median')
        p1, p2 = Framework.one_shot_test_parallel(Z, X, M, Y, S, G1=median_imputer_1, G2=median_imputer_2, n_jobs = 12)
        print("One-shot test for Fisher's sharp null for Median imputer")
        print("p-values for part 1:", p1)
        print("p-values for part 2:", p2)




        


