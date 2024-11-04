
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
from sklearn.base import clone
from scipy.stats import rankdata


class RandomizationTest:
    #load data
    def __init__(self,N,covariance_adjustment = 0):
        self.N = N
        self.covariance_adjustment = covariance_adjustment # 0 = Null, 1 = LR, 2 = XG

    def holm_bonferroni(self,p_values, alpha = 0.05):
        # Perform the Holm-Bonferroni correction
        reject, corrected_p_values, _, _ = multipletests(p_values, alpha=alpha, method='holm')

        # Check if any null hypothesis can be rejected
        any_rejected = any(reject)

        return any_rejected

    def getY(self, G, df_Z, df_noZ, indeX, lenX, indexY, lenY):
        """
        Method to get the adjusted Y using either a model G or utilizing a linear regression model
        depending on the covariance adjustment strategy specified.
        
        Parameters:
        - G: Model used for initial prediction of Y
        - df_Z, df_noZ: Input DataFrames
        - indeX, lenX: Index and length to slice X data from df_noZ
        - indexY, lenY: Index and length to slice Y data
        
        Returns: Adjusted Y as per specified strategy
        """

        # Step 1: Get Y_head using model G
        if G:
            df_imputed = G.fit_transform(df_Z)
        else:
            df_imputed = df_Z.to_numpy()

        return df_imputed[:, indexY:indexY+lenY]

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

        Y_rank = rankdata(y)
        t = np.sum(Y_rank[z == 1])

        
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
    
    def T_M(self,z,m):
            
            assert len(z) == len(m)

            #t = np.sum(z[m == 1])
            z = np.array(z)
            m = np.array(m)
            t = np.sum(z[m == 1])

            
            return t


    def test(self, Z, X, M, Y, G, strata_size, L=10000, verbose = False):

        df_Z = pd.DataFrame(np.concatenate((Z, X, Y), axis=1))

        Y_copy = np.ma.masked_array(Y, mask=M)
        Y_copy = Y_copy.filled(np.nan)
        lenY = Y.shape[1]

        # N is the number of rows of the data frame
        N = df_Z.shape[0]

        # re-impute the missing values and calculate the observed test statistics in part 2
        t_obs = self.T_M(Z, M)
        
        #print train end
        if verbose:
            print("t_obs:"+str(t_obs))
            print("Permutation Start")

        # simulate data and calculate test statistics
        t_sim = np.zeros((L,lenY))

        for l in range(L):
            
            # simulate treatment indicators
            Z_sim = []
            half_strata_size = strata_size // 2  # Ensure strata_size is even

            for i in range(int(N/strata_size)):
                strata = np.array([0.0]*half_strata_size + [1.0]*half_strata_size)
                np.random.shuffle(strata)
                Z_sim.append(strata)
            Z_sim = np.concatenate(Z_sim).reshape(-1, 1)
            
            # get the test statistics 
            t_sim[l] = self.T_M(Z_sim, M)

        if verbose:
            print("t_sims_mean:"+str(np.mean(t_sim)))
            print("\n")

        # perform Holm-Bonferroni correction
        p_values = []
        for i in range(lenY):
            p_values.append(np.mean(t_sim[:,i] >= t_obs, axis=0))
        reject = self.holm_bonferroni(p_values)

        return reject,p_values

