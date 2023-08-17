
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
from sklearn.base import clone
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import linear_model
import time

class RetrainTest:
    #load data
    def __init__(self,covariance_adjustment = False):
        self.covariance_adjustment = covariance_adjustment

    def holm_bonferroni(self,p_values, alpha = 0.05):
        # Perform the Holm-Bonferroni correction
        reject, corrected_p_values, _, _ = multipletests(p_values, alpha=alpha, method='holm')

        # Check if any null hypothesis can be rejected
        any_rejected = any(reject)

        return any_rejected

    def getY(self, G, df_Z, df_noZ, indexY ,lenY):
        if self.covariance_adjustment == True:
            G_copy = clone(G)
            df_imputed = G.fit_transform(df_Z)
            df_noZ_imputed = G_copy.fit_transform(df_noZ)
            df_imputed[:,indexY:indexY+lenY] - df_noZ_imputed[:,indexY-1:indexY+lenY-1]
        else:
            df_imputed = G.fit_transform(df_Z)
            return df_imputed[:,indexY:indexY+lenY]

    def T(self,z,y):

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

    def split(self, y, z, M):
        
        missing_indices = M[:].astype(bool)
        non_missing_indices = ~missing_indices

        y_missing = y[missing_indices].reshape(-1,)
        y_non_missing = y[non_missing_indices].reshape(-1,)

        z_missing = z[missing_indices].reshape(-1,)
        z_non_missing = z[non_missing_indices].reshape(-1,)

        return y_missing, y_non_missing, z_missing, z_non_missing

    def getT(self, y, z, lenY, M, verbose):

        t = []
        for i in range(lenY):
            if np.all(M[:, i] == 0.0):
                # Column i does not contain any missing values
                pass
            else:
                # Split the data into missing and non-missing parts using the split function
                y_missing, y_non_missing, z_missing, z_non_missing = self.split(y[:,i], z, M[:,i])
                
                # Calculate T for missing and non-missing parts
                t_missing = self.T(z_missing, y_missing.reshape(-1,))
                t_non_missing = self.T(z_non_missing, y_non_missing.reshape(-1,))

                # Sum the T values for both parts
                t_combined = t_missing + t_non_missing
                t.append(t_combined)
        return t

    def getZsimTemplates(self, Z_sorted, S):
        # Create a Z_sim template for each unique value in S
        Z_sim_templates = []
        unique_strata = np.unique(S)
        for stratum in unique_strata:
            strata_indices = np.where(S == stratum)[0]
            strata_Z = Z_sorted[strata_indices]
            p = np.mean(strata_Z)
            strata_size = len(strata_indices)
            Z_sim_template = [0.0] * int(strata_size * (1 - p)) + [1.0] * int(strata_size * p)
            Z_sim_templates.append(Z_sim_template)
        return Z_sim_templates
    
    def getZsim(self, Z_sim_templates):
        Z_sim = []
        for Z_sim_template in Z_sim_templates:
            strata_Z_sim = np.array(Z_sim_template.copy())
            np.random.shuffle(strata_Z_sim)
            Z_sim.append(strata_Z_sim)
        Z_sim = np.concatenate(Z_sim).reshape(-1, 1)
        return Z_sim

    def sort(self, Z, Y, S, M):
        # Reshape Z, Y, S, M to (-1, 1) if they're not already in that shape
        if len(Z.shape) == 1 or Z.shape[1] != 1:
            Z = Z.reshape(-1, 1)
        if len(S.shape) == 1 or S.shape[1] != 1:
            S = S.reshape(-1, 1)
        if len(M.shape) == 1 or M.shape[1] != M.shape[1]:
            M = M.reshape(-1, M.shape[1])

        # Concatenate Z, Y, S, and M into a single DataFrame
        df = pd.DataFrame(np.concatenate((Z, Y, S, M), axis=1))

        # Sort the DataFrame based on S (assuming S is the column before M)
        df = df.sort_values(by=df.columns[-M.shape[1] - 1])

        # Extract Z, Y, S, and M back into separate arrays
        Z = df.iloc[:, :Z.shape[1]].values.reshape(-1, 1)
        Y = df.iloc[:, Z.shape[1]:Z.shape[1] + Y.shape[1]].values
        S = df.iloc[:, Z.shape[1] + Y.shape[1]:Z.shape[1] + Y.shape[1] + S.shape[1]].values.reshape(-1, 1)
        M = df.iloc[:, -M.shape[1]:].values

        return Z, Y, S, M


    def check_param(self,Z, Y, S, M, G, L, verbose, alpha):
        # Check Z: must be one of 1, 0, 1.0, 0.0
        if not np.all(np.isin(Z, [0, 1])):
            raise ValueError("Z must contain only 0, 1")
        
        # Check Y: must be numeric
        if not np.issubdtype(Y.dtype, np.number):
            raise ValueError("Y must contain numeric values")

        # Check M: can be None
        if M is not None and not np.all(np.isin(M, [0, 1])):
            raise ValueError("M must contain only 0 or 1")

        # Check Y: if M is None, Y must have missing values
        if M is None and not np.isnan(Y).any():
            raise ValueError("If M is None, Y must contain missing values")

        # Check S: must all be integer
        if not np.all(np.equal(S, S.astype(int))):
            raise ValueError("S must contain only integer values")

        # Check L: must be an integer greater than 0
        if not isinstance(L, int) or L <= 0:
            raise ValueError("L must be an integer greater than 0")

        # Check verbose: must be True or False
        if verbose not in [True, False, 1, 0]:
            raise ValueError("verbose must be True or False")

        # Check alpha: must be > 0 and <= 1
        if not (0 < alpha <= 1):
            raise ValueError("alpha must be greater than 0 and less than or equal to 1")
        
        # Check G: Cannot be None
        if G is None:
            raise ValueError("G cannot be None")
        
        # Check Y: must be a 2D array
        if len(Y.shape) != 2:
            raise ValueError("Y must be a 2D array")
        
    def retrain_test(self,*,Z, Y, G=IterativeImputer(estimator = linear_model.BayesianRidge()), S=None, M = None, L = 10000,verbose = False, alpha = 0.05):
        """
        RIMO:A retrain framework for testing H_0.

        Args:
        Z: 2D array of observed treatment indicators
        Y: 2D array of observed values for K outcomes
        S: 2D array of the strata indicators
        M: 2D array of observed missing indicators, if None, then get the missing indicators from Y
        G: a function that takes (Z, M, Y_k) as input and returns the imputed complete values 
        L: number of Monte Carlo simulations (default is 10000)
        verbose: a boolean indicating whether to print training start and end (default is False)

        Returns:
        p_values: a 1D array of p-values for lenY outcomes
        reject: a boolean indicating whether the null hypothesis is rejected for each outcome
        """

        start_time = time.time()

        if S is None:
            S = np.ones(Z.shape)

        #if y is dataframe, convert it to numpy array
        if isinstance(Y, pd.DataFrame):
            Y = Y.values

        # Check the validity of the input parameters
        self.check_param(Z, Y, S, M, G, L, verbose, alpha)

        # mask Y
        if M is None:
            M = np.isnan(Y).reshape(-1, Y.shape[1])

        Y = np.ma.masked_array(Y, mask=M)
        Y = Y.filled(np.nan)
        
        Z, Y, S, M = self.sort(Z, Y, S, M)

        df_Z = pd.DataFrame(np.concatenate((Z,  Y), axis=1))

        df_noZ = pd.DataFrame(Y)
        G_model = clone(G)

        # lenY is the number of how many columns are Y
        lenY = Y.shape[1]

        # indexY is the index of the first column of Y
        indexY = Z.shape[1]

        # re-impute the missing values and calculate the observed test statistics in part 2
        Y_pred = self.getY(G, df_Z, df_noZ, indexY, lenY)
        t_obs = self.getT(Y_pred, Z, lenY, M, verbose = verbose)

        #print train end
        if verbose:
            print("Observed Wilconxin rank-sum test statistics:"+str(t_obs))
            print("\nRe-impute Start\n")
            print("=========================================================")
            
        # simulate data and calculate test statistics
        t_sim = [ [] for i in range(L)]
        Z_sim_templates = self.getZsimTemplates(Z, S)

        for l in range(L):
            
            # simulate treatment indicators
            Z_sim = self.getZsim(Z_sim_templates)
            
            G_clone = clone(G_model)
            df_Z = pd.DataFrame(np.concatenate((Z_sim, Y), axis=1))
            Y_pred = self.getY(G_clone, df_Z, df_noZ, indexY, lenY)

            # get the test statistics 
            t_sim[l] = self.getT(Y_pred, Z_sim, lenY, M, verbose=False)

            if verbose:
                print(f"Iteration {l+1}/{L} completed. Test statistics[{l}]: {t_sim[l]}")

        if verbose:
            print("=========================================================")
            print("Re-impute mean t-value:"+str(np.mean(t_sim)))

        # perform Holm-Bonferroni correction
        p_values = []
        t_sim = np.array(t_sim)
        for i in range(len(t_obs)):
            p_values.append(np.mean(t_sim[:,i] >= t_obs[i], axis=0))
        reject = self.holm_bonferroni(p_values,alpha = alpha)

        if verbose:
            print("\nthe time used for the retrain framework:"+str(time.time() - start_time) + " seconds")
        
        return reject, p_values