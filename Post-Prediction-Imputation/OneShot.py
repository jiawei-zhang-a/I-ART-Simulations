
import pandas as pd
import numpy as np
import multiprocessing

class OneShotTest:
    #load data
    def __init__(self,N):
        self.N = N

    #split based on strata
    def split_df(self,df):

        # Sort the groups by the number of rows in each group
        #sorted_df = df.sort_values(by = index_S, ascending=True)
        
        # Split the sorted groups into two equal-sized sets of 100 strata each
        df_set1 = df.iloc[:int(self.N/2), :]
        df_set2 = df.iloc[int(self.N/2):self.N, :]

        #set the index of the two sets from zero to 1
        df_set1.index = range(len(df_set1))
        df_set2.index = range(len(df_set2))
        
        # Return the two sets of strata
        return df_set1, df_set2

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
        
        return np.array(t)

    def getT(self, G, df, indexY):
        
        # Get the imputed data Y and indicator Z
        df_imputed = G.transform(df)
        y = df_imputed[:, indexY:df_imputed.shape[1]]
        z = df_imputed[:, 0]
        
        t = []
        for i in range(3):
            #the Wilcoxon rank sum test
            t.append(self.T(z.reshape(-1,),y[:,i].reshape(-1,)))

        return t

    def worker(self, args):
        # unpack the arguments
        X, Y_masked, G1, G2, t1_obs, t2_obs, L, verbose = args

        # simulate data and calculate test statistics
        t1_sim = np.zeros((L,3))
        t2_sim = np.zeros((L,3))

        for l in range(L):

            # simulate treatment indicators in parts 1 and 2
            df_sim = pd.DataFrame(np.concatenate((X, Y_masked), axis=1))
            
            # split the simulated data into two parts
            df1_sim, df2_sim = self.split_df(df_sim)

            # simulate treatment indicators in parts 1 and 2
            Z_1 = np.random.binomial(1, 0.5, df1_sim.shape[0]).reshape(-1, 1)
            Z_2 = np.random.binomial(1, 0.5, df2_sim.shape[0]).reshape(-1, 1)
            df1_sim = pd.concat([pd.DataFrame(Z_1), df1_sim], axis=1)
            df2_sim = pd.concat([pd.DataFrame(Z_2), df2_sim], axis=1)

            # get the test statistics in part 1
            t1_sim[l] = self.getT(G2, df1_sim, Z_1.shape[1] + X.shape[1])

            # get the test statistics in part 2
            t2_sim[l] = self.getT(G1, df2_sim, Z_2.shape[1] + X.shape[1])

            # Calculate the completeness percentage
            if l % 100 == 0:
                print(t1_sim[l], t2_sim[l])
                completeness = l / L * 100  
                if verbose:
                    print(f"Task is {completeness:.2f}% complete.")

        p11 = np.mean(t1_sim[:,0] >= t1_obs[0], axis=0)
        p12 = np.mean(t2_sim[:,0] >= t2_obs[0], axis=0)
        p21 = np.mean(t1_sim[:,1] >= t1_obs[1], axis=0)
        p22 = np.mean(t2_sim[:,1] >= t2_obs[1], axis=0)
        p31 = np.mean(t1_sim[:,2] >= t1_obs[2], axis=0)
        p32 = np.mean(t2_sim[:,2] >= t2_obs[2], axis=0)
        
        print(p11.shape, p12.shape, p32.shape)

        return p11, p12, p21, p22, p31, p32

    def one_shot_test_parallel(self, Z, X, M, Y, G1, G2, L=10000, n_jobs=multiprocessing.cpu_count(),verbose = False):
        """
        A one-shot framework for testing H_0.

        Args:
        Z: 2D array of observed treatment indicators
        X: 2D array of observed covariates
        M: 2D array of observed missing indicators
        Y: 2D array of observed values for K outcomes
        G1: a function that takes (Z, X, M, Y_k) as input and returns the imputed value for outcome k
        G2: a function that takes (Z, X, M, Y_k) as input and returns the imputed value for outcome k
        L: number of Monte Carlo simulations (default is 10000)

        Returns:
        p1: 1D array of exact p-values for testing Fisher's sharp null in part 1
        p2: 1D array of exact p-values for testing Fisher's sharp null in part 2
        """
        #print train start
        if verbose:
            print("Training start")

        # create data a whole data frame
        Y_masked = np.ma.masked_array(Y, mask=M)
        Y_masked = Y_masked.filled(np.nan)
        df = pd.DataFrame(np.concatenate((Z, X, Y_masked), axis=1))
        
        # split the data into two parts
        df1, df2 = self.split_df(df)

        # re-impute the missing values and calculate the observed test statistics in part 2
        G2.fit(df2)
        t1_obs = self.getT(G2, df1, Z.shape[1] + X.shape[1])

        # re-impute the miassing values and calculate the observed test statistics in part 1
        G1.fit(df1)
        t2_obs = self.getT(G1, df2, Z.shape[1] + X.shape[1])


        #print train end
        if verbose:
            print("t1_obs:"+str(t1_obs), "t2_obs:"+str(t2_obs))
            print("Training end")
        
        # print the number of cores
        if verbose:
            print(f"Number of jobs: {n_jobs}")

        # simulate data and calculate test statistics in parallel
        args_list = [(X, Y_masked, G1, G2, t1_obs, t2_obs, int(L / n_jobs), verbose)] * n_jobs
        with multiprocessing.Pool(processes=n_jobs) as pool:
            p_list = pool.map(self.worker, args_list)
        p11 = np.mean([p[0] for p in p_list], axis=0)
        p12 = np.mean([p[1] for p in p_list], axis=0)
        p21 = np.mean([p[2] for p in p_list], axis=0)
        p22 = np.mean([p[3] for p in p_list], axis=0)
        p31 = np.mean([p[4] for p in p_list], axis=0)
        p32 = np.mean([p[5] for p in p_list], axis=0)
        
        return p11, p12, p21, p22, p31, p32
    
    def one_shot_test(self, Z, X, M, Y, G1, G2,  L=10000, verbose = False):
        """
        A one-shot framework for testing H_0.

        Args:
        Z: 2D array of observed treatment indicators
        X: 2D array of observed covariates
        M: 2D array of observed missing indicators
        Y: 2D array of observed values for K outcomes
        G1: a function that takes (Z, X, M, Y_k) as input and returns the imputed value for outcome k
        G2: a function that takes (Z, X, M, Y_k) as input and returns the imputed value for outcome k
        L: number of Monte Carlo simulations (default is 10000)

        Returns:
        p1: 1D array of exact p-values for testing Fisher's sharp null in part 1
        p2: 1D array of exact p-values for testing Fisher's sharp null in part 2
        """

        #print train start
        if verbose:
            print("Training start")

        # create data a whole data frame
        Y_masked = np.ma.masked_array(Y, mask=M)
        Y_masked = Y_masked.filled(np.nan)
    
        df = pd.DataFrame(np.concatenate((Z, X, Y_masked), axis=1))
        
        # split the data into two parts
        df1, df2 = self.split_df(df)

        # re-impute the missing values and calculate the observed test statistics in part 2
        t2_obs = np.zeros(3)
        G1.fit(df1)
        t2_obs = self.getT(G1, df2, Z.shape[1] + X.shape[1])

        # re-impute the missing values and calculate the observed test statistics in part 1
        t1_obs = np.zeros(3)
        G2.fit(df2)
        t1_obs = self.getT(G2, df1, Z.shape[1] + X.shape[1])

        #print train end
        if verbose:
            print("t1_obs:"+str(t1_obs), "t2_obs:"+str(t2_obs))
            print(df1.shape, df2.shape)
            print("Training end")

        # simulate data and calculate test statistics
        t1_sim = np.zeros((L,3))
        t2_sim = np.zeros((L,3))

        """
        df_imputed = G1.transform(df2)
        pd.DataFrame(df2).to_csv("samples/df_obs.csv")
        pd.DataFrame(df_imputed).to_csv("samples/df_obs_imputed.csv")
        """
        
        for l in range(L):

            # simulate treatment indicators in parts 1 and 2
            df_sim = pd.DataFrame(np.concatenate((X, Y_masked), axis=1))
            
            # split the simulated data into two parts
            df1_sim, df2_sim = self.split_df(df_sim)

            # simulate treatment indicators in parts 1 and 2
            Z_1 = np.random.binomial(1, 0.5, df1_sim.shape[0]).reshape(-1, 1)
            Z_2 = np.random.binomial(1, 0.5, df2_sim.shape[0]).reshape(-1, 1)
            df1_sim = pd.concat([pd.DataFrame(Z_1), df1_sim], axis=1)
            df2_sim = pd.concat([pd.DataFrame(Z_2), df2_sim], axis=1)
            
        
            # get the test statistics in part 1
            t1_sim[l] = self.getT(G2, df1_sim, Z.shape[1] + X.shape[1])
            
            # get the test statistics in part 2
            t2_sim[l] = self.getT(G1, df2_sim,  Z.shape[1] + X.shape[1])

            # Calculate the completeness percentage
            if l % 100 == 0:
                completeness = l / L * 100  
                
                """
                if l == 100:
                    df_imputed = G1.transform(df2_sim)
                    #pd.DataFrame(df2).to_csv("samples/df_sim.csv")
                    #pd.DataFrame(df_imputed).to_csv("samples/df_sim_imputed.csv")
                """

                if verbose:
                    print(f"Task is {completeness:.2f}% complete.")
                    print("t1_sim:"+str(t1_sim[l]), "t2_sim:"+str(t2_sim[l]))


        # calculate exact p-values for each outcome
        p11 = np.mean(t1_sim[:,0] >= t1_obs[0], axis=0)
        p12 = np.mean(t2_sim[:,0] >= t2_obs[0], axis=0)
        p21 = np.mean(t1_sim[:,1] >= t1_obs[1], axis=0)
        p22 = np.mean(t2_sim[:,1] >= t2_obs[1], axis=0)
        p31 = np.mean(t1_sim[:,2] >= t1_obs[2], axis=0)
        p32 = np.mean(t2_sim[:,2] >= t2_obs[2], axis=0)
        
        return p11, p12, p21, p22, p31, p32
    

