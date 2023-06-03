
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
from sklearn.base import clone
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
import xgboost as xgb


class RetrainTest:
    #load data
    def __init__(self,N):
        self.N = N

    def holm_bonferroni(self,p_values, alpha = 0.05):
        # Perform the Holm-Bonferroni correction
        reject, corrected_p_values, _, _ = multipletests(p_values, alpha=alpha, method='holm')

        # Check if any null hypothesis can be rejected
        any_rejected = any(reject)

        return any_rejected

    def getY(self, G, df_Z, df_noZ, indexY ,lenY):
        if G:
            #G2 = clone(G)
            df_imputed = G.fit_transform(df_Z)
            #df_noZ_imputed = G2.fit_transform(df_noZ)
            
        else:
            df_imputed = df_Z.to_numpy()
            #df_noZ_imputed = df_noZ.to_numpy()

        
        #df_noZ_imputed = df_noZ.to_numpy()
        G2 = IterativeImputer(estimator = linear_model.BayesianRidge(),max_iter=3)
        df_noZ_imputed = G2.fit_transform(df_noZ)

        y = df_imputed[:,indexY:indexY+lenY] - df_noZ_imputed[:,indexY-1:indexY+lenY-1]

        return y

    def get_corr(self, G, df, Y, indexY, lenY):
        # Get the imputed data Y and indicator Z
        if G is None:
            df_imputed = df.to_numpy()
        else:
            df_imputed = G.transform(df)
        y = df_imputed[:, indexY:indexY+lenY]

        # Initialize the lists to store imputed and truth values for missing positions
        y_imputed = []
        y_truth = []

        # Iterate over the rows and columns to find missing values
        for i in range(df.shape[0]):
            for j in range(lenY):
                # Check if the value in the last three columns of df is missing (you can replace 'your_missing_value' with the appropriate value or condition)
                if np.isnan(df.iloc[i, -lenY + j]):
                    y_imputed.append(y[i, j])
                    y_truth.append(Y[i, j])
        if y_imputed == y_truth:
            return 1.0

        # Calculate the correlation between the imputed data and the observed data
        corr = 0.0
        if len(y_imputed) > 0 and len(y_truth) > 0:
            val = np.corrcoef(y_imputed, y_truth)[0, 1]
            if np.isnan(val) == False:
                corr = val

        return corr

    def CombinedT(self,z,y):

        #the Wilcoxon rank sum test
        n = len(z)
        t = 0

        #O(N*Log(N)) version
        my_list = []
        for i in range(n):
            my_list.append((z[i],y[i]))
        sorted_list = sorted(my_list, key=lambda x: x[1])

        #Calculate
        for i in range(n):
            t += sorted_list[i][0] * (i + 1)
        return t

    def getCombinedT(self, y, z, lenY):
        t = []
        for i in range(lenY):
            #the Wilcoxon rank sum test
            t.append(self.T(z.reshape(-1,),y[:,i].reshape(-1,)))
        return t

    def T(self,z,y):

        #the Wilcoxon rank sum test
        n = len(z)
        t = 0
        #O(N^2) version
        """
        for n in range(N):
            rank = sum(1 for n_prime in range(N) if Y[n] >= Y[n_prime])
            T += Z[n] * rank
        """

        #O(N*Log(N)) version
        my_list = []
        for i in range(n):
            my_list.append((z[i],y[i]))
        sorted_list = sorted(my_list, key=lambda x: x[1])

        #Calculate
        for i in range(n):
            t += sorted_list[i][0] * (i + 1)
        
        return t

    def split(self, y, z, M):
        #print(y,z,M)
        missing_indices = M[:].astype(bool)
        non_missing_indices = ~missing_indices

        y_missing = y[missing_indices].reshape(-1,)
        y_non_missing = y[non_missing_indices].reshape(-1,)

        z_missing = z[missing_indices].reshape(-1,)
        z_non_missing = z[non_missing_indices].reshape(-1,)

        return y_missing, y_non_missing, z_missing, z_non_missing

    def getT(self, y, z, lenY, M, verbose = False):

        t = []
        for i in range(lenY):
            # Split the data into missing and non-missing parts using the split function
            y_missing, y_non_missing, z_missing, z_non_missing = self.split(y[:,i], z, M[:,i])
            
            # Calculate T for missing and non-missing parts
            t_missing = self.T(z_missing, y_missing.reshape(-1,))
            t_non_missing = self.T(z_non_missing, y_non_missing.reshape(-1,))

            # Sum the T values for both parts
            t_combined = t_missing + t_non_missing
            if verbose:
                print("t_non_missing:",t_non_missing)
                print("t_missing:",t_missing)
            t.append(t_combined)

        return t

    def retrain_test(self, Z, X, M, Y, Y_noZ, G,  L=10000, verbose = False):
        if G == None:
            return self.retrain_test_oracle(Z, X, M, Y, Y_noZ, G, L, verbose)   
        else:
            return self.retrain_test_imputed(Z, X, M, Y, Y_noZ, G, L, verbose)
        

    def retrain_test_oracle(self, Z, X, M, Y, Y_noZ, G,  L=10000, verbose = False):
        """
        A retrain framework for testing H_0.

        Args:
        Z: 2D array of observed treatment indicators
        X: 2D array of observed covariates
        M: 2D array of observed missing indicators
        Y: 2D array of observed values for K outcomes
        G: a function that takes (Z, X, M, Y_k) as input and returns the imputed value for outcome k
        L: number of Monte Carlo simulations (default is 10000)
        verbose: a boolean indicating whether to print training start and end (default is False)

        Returns:
        p_values: a 1D array of p-values for lenY outcomes
        reject: a boolean indicating whether the null hypothesis is rejected for each outcome
        corr: a 1D array of correlations between the imputed and observed values for lenY outcomes

        """

        df_Z = pd.DataFrame(np.concatenate((Z, X, Y), axis=1))

        Y_copy = np.ma.masked_array(Y, mask=M)
        Y_copy = Y_copy.filled(np.nan)

        df_noZ = pd.DataFrame(np.concatenate((X, Y_copy), axis=1))

        # lenY is the number of how many columns are Y
        lenY = Y.shape[1]

        # indexY is the index of the first column of Y
        indexY = Z.shape[1] + X.shape[1]

        # N is the number of rows of the data frame
        N = df_Z.shape[0]

        # re-impute the missing values and calculate the observed test statistics in part 2
        bias = self.getY(G, df_Z, df_noZ, indexY, lenY)
        t_obs = self.getT(bias, Z, lenY, M, verbose = verbose)

        #print train end
        if verbose:
            print("t_obs:"+str(t_obs))
        corr_G = self.get_corr(G, df_Z, Y, indexY, lenY)
        # simulate data and calculate test statistics
        t_sim = np.zeros((L,Y.shape[1]))

        for l in range(L):
            
            # simulate treatment indicators
            Z_sim = np.random.binomial(1, 0.5, N).reshape(-1, 1)
            
            df_Z = pd.DataFrame(np.concatenate((Z_sim, X, Y), axis=1))
            bias = self.getY(G, df_Z, df_noZ, indexY, lenY)

            # get the test statistics 
            t_sim[l] = self.getT(bias, Z_sim, lenY, M)

        if verbose:
            print("t_sims_mean:"+str(np.mean(t_sim)))

        # perform Holm-Bonferroni correction
        p_values = []
        for i in range(lenY):
            p_values.append(np.mean(t_sim[:,i] >= t_obs[i], axis=0))
        reject = self.holm_bonferroni(p_values,alpha = 0.2)
        
        return p_values, reject, corr_G

    def retrain_test_imputed(self, Z, X, M, Y, Y_noZ, G,  L=10000, verbose = False):
        """
        A retrain framework for testing H_0.

        Args:
        Z: 2D array of observed treatment indicators
        X: 2D array of observed covariates
        M: 2D array of observed missing indicators
        Y: 2D array of observed values for K outcomes
        G: a function that takes (Z, X, M, Y_k) as input and returns the imputed value for outcome k
        L: number of Monte Carlo simulations (default is 10000)
        verbose: a boolean indicating whether to print training start and end (default is False)

        Returns:
        p_values: a 1D array of p-values for lenY outcomes
        reject: a boolean indicating whether the null hypothesis is rejected for each outcome
        corr: a 1D array of correlations between the imputed and observed values for lenY outcomes

        """

        # mask Y


        df_Z = pd.DataFrame(np.concatenate((Z, X, Y), axis=1))


        df_noZ = pd.DataFrame(np.concatenate((X, Y), axis=1))
        G_clones = [clone(G) for _ in range(L)]

        # lenY is the number of how many columns are Y
        lenY = Y.shape[1]

        # indexY is the index of the first column of Y
        indexY = Z.shape[1] + X.shape[1]

        # N is the number of rows of the data frame
        N = df_Z.shape[0]

        # re-impute the missing values and calculate the observed test statistics in part 2
        bias = self.getY(G, df_Z, df_noZ, indexY, lenY)
        t_obs = self.getT(bias, Z, lenY, M, verbose = verbose)

        # get the correlation of G1 and G2
        corr_G = self.get_corr(G, df_Z, Y, indexY, lenY)

        #print train end
        if verbose:
            print("t_obs:"+str(t_obs))
            print("corr_G:"+str(corr_G))

        # simulate data and calculate test statistics
        t_sim = np.zeros((L,Y.shape[1]))

        for l in range(L):
            
            # simulate treatment indicators
            Z_sim = np.random.binomial(1, 0.5, N).reshape(-1, 1)
            
            df_Z = pd.DataFrame(np.concatenate((Z_sim, X, Y), axis=1))
            bias = self.getY(G_clones[l], df_Z, df_noZ, indexY, lenY)

            # get the test statistics 
            t_sim[l] = self.getT(bias, Z_sim, lenY, M)

        if verbose:
            print("t_sims_mean:"+str(np.mean(t_sim)))

        # perform Holm-Bonferroni correction
        p_values = []
        for i in range(lenY):
            p_values.append(np.mean(t_sim[:,i] >= t_obs[i], axis=0))
        reject = self.holm_bonferroni(p_values,alpha = 0.2)
        
        return p_values, reject, corr_G