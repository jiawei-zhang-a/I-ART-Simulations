import sys
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
import SingleOutcomeModelGenerator as Generator
import MultipleOutcomeModelGenerator as GeneratorMutiple
import RandomizationTestModelBased as RandomizationTest
import os
import lightgbm as lgb
import xgboost as xgb
import iArtModelBased as iArt
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


# Do not change this parameter
beta_coef = None
task_id = 1

# Set the default values
max_iter = 3
L = 2000

# For Compelete Analysis
class NoOpImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Initialization code can include parameters if needed
        pass

    def fit(self, X, y=None):
        # Nothing to do here, return self to allow chaining
        return self

    def transform(self, X):
        # Check if X is a numpy array, if not, convert it to avoid potential issues
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        # Return the data unchanged
        return X

    def fit_transform(self, X, y=None):
        # This method can often be optimized but here we'll just use fit and transform sequentially
        return self.fit(X, y).transform(X)


def run(Nsize, filepath,  Missing_lambda,adjust = 0, model = 0, verbose=1, small_size = True, multiple = False):
    
    Missing_lambda = None

    if beta_coef == 0:
        Iter = 10000
    else:
        Iter = 1000

    # Simulate data
    if multiple == False:
        DataGen = Generator.DataGenerator(N = Nsize, strata_size=10,beta = beta_coef,model = model, MaskRate=0.5, verbose=verbose,Missing_lambda = Missing_lambda)
        X, Z, U, Y, M, S = DataGen.GenerateData()
    else:
        DataGen = GeneratorMutiple.DataGenerator(N = Nsize, strata_size=10,beta = beta_coef, MaskRate=0.5, verbose=verbose,Missing_lambda = Missing_lambda)
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
    reject, p_values = iArt.test(Z=Z, X=X, Y=Y,S=S,G=median_imputer,L=Iter, verbose=verbose)
    values_median = [ *p_values, reject ]

    """median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    reject, p_values = iArt.test(Z=Z, X=X, Y=Y,S=S,G=median_imputer,L=Iter, verbose=verbose, covariate_adjustment=1)
    values_medianLR = [ *p_values, reject ]"""

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

    """# Combine the data into DataFrame

    # Drop the missing values only based on outcomes Y
    combined_data = combined_data.dropna(subset=['Y'])

    X = combined_data[['X1', 'X2', 'X3', 'X4', 'X5']].values
    Z = combined_data['Z'].values.reshape(-1, 1)
    Y = combined_data['Y'].values.reshape(-1, 1)
    S = combined_data['S'].values.reshape(-1, 1)

    G = NoOpImputer()

    reject, p_values = iArt.test(Z=Z, X=X, Y=Y,S =S,G=G,L=L, covariate_adjustment=adjust)
    values_complete = [ *p_values, reject ]"""


    os.makedirs("%s/%f"%(filepath,beta_coef), exist_ok=True)
    
    #os.makedirs("%s_adjusted_Median/%f"%(filepath,beta_coef), exist_ok=True)

    # Save numpy arrays to files
    np.save('%s/%f/p_values_median_%d.npy' % (filepath, beta_coef, task_id), values_median)
    #np.save('%s_adjusted_Median/%f/p_values_median_%d.npy' % (filepath, beta_coef, task_id), values_medianLR)

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
    """exit()
        # Lambda values dictionary
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
    for coef in np.arange(0.0, 0.4, 0.05): 
        beta_coef = coef
        run(1000, filepath="Result/HPC_power_1000_model5",adjust =0,  Missing_lambda=lambda_values[1000].get(coef, None),model = 5, small_size=False, multiple = True)
    # 50 size coef loop
    for coef in np.arange(0.0, 2.5, 0.5): 
        beta_coef = coef
        run(50, filepath="Result/HPC_power_50_model5",adjust =0, Missing_lambda=lambda_values[50].get(coef, None),model = 5, small_size=True, multiple = True)
    """
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