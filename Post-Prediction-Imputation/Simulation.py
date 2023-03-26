import numpy as np
from mv_laplace import MvLaplaceSampler
import matplotlib.pyplot as plt
import pandas as pd

class DataGenerator:
  def __init__(self, N, N_T, N_S, beta_11, beta_12, beta_21, beta_22, beta_23, beta_31, MaskRate):
    self.N = N
    self.N_T = N_T
    self.N_S = N_S
    self.beta_11 = beta_11
    self.beta_12 = beta_12
    self.beta_21 = beta_21
    self.beta_22 = beta_22
    self.beta_23 = beta_23
    self.beta_31 = beta_31
    self.MaskRate = MaskRate

  def GenerateX(self):
      # generate Xn1 and Xn2
      mean = [1/2, -1/3]
      cov = [[1, 1/2], [1/2, 1]]
      X1_2 = np.random.multivariate_normal(mean, cov, self.N)


      # generate Xn3 and Xn4
      loc = [0, 1/np.sqrt(3)]
      cov = [[1,1/np.sqrt(2)], [1/np.sqrt(2),1]]

      sampler = MvLaplaceSampler(loc, cov)
      X3_4 = sampler.sample(self.N)

      # generate Xn5
      p = 1/3
      X5 = np.random.binomial(1, p, self.N)

      # combine all generated variables into a single matrix
      X = np.hstack((X1_2, X3_4, X5.reshape(-1,1)))
      
      return X

  def GenerateU(self):
      # generate Un1
      mean = 1/2
      std = 1
      U_n1 = np.random.normal(mean, std,self.N)

      # generate Un2
      U_n2 = np.random.binomial(1, 2/3, self.N)

      # combine all generated variables into a single matrix
      U = np.concatenate((U_n1.reshape(-1, 1), U_n2.reshape(-1, 1)), axis=1)

      return U

  def GenerateZ(self):
      # generate Zn
      Z = np.zeros(self.N)
      Z[:self.N_T] = 1
      np.random.shuffle(Z)

      Z = Z.reshape(-1,1)

      return Z

  def GenerateY(self, X, U, Z):
    #def sum1():
    sum1 = np.zeros(self.N)
    for p in range(1,6):
      sum1 += np.sqrt(p) * np.exp(X[:,p-1])

    #def sum2():
    sum2 = np.zeros(self.N)
    for p in range(1,6):
      for p_2 in range(1,6):
        sum2 += X[:,p-1] * np.power(X[:,p_2-1],2)

    #def sum3():
    sum3 = np.zeros(self.N)

    for p in range(1,6):
      sum3 += np.cos(p) * X[:,p-1]

    #def sum4():
    sum4 = np.zeros(self.N)
    for p in range(1,6):
      for p_2 in range(1,6):
        sum4 += X[:,p-1] * np.exp(X[:,p_2-1])

    #def sum5():
    sum5 = np.zeros(self.N)
    for p in range(1,6):
      sum5 += np.sin(p) * X[:,p-1]

    #def sum6(): 
    sum6 = np.zeros(self.N)
    for p in range(1,6):
      for p_2 in range(1,6):
        for p_3 in range(1,6):
          sum6 += X[:,p-1] * X[:,p_2-1] * np.exp(X[:,p_3-1])


    U_n1 = U[:,0]
    U_n2 = U[:,1]
    Z = Z.reshape(-1,)

    # Calculate Y_n1
    Y_n1 = (self.beta_11 * Z + self.beta_12 * Z * sum1   + sum2 + np.sin(U_n1) + U_n2) 

    # Compute Yn2
    Y_n2 = (self.beta_21 * Z + self.beta_22 * Z * X[:,0] + self.beta_23 * Z * U_n1 * U_n2 + sum3 + sum4) 

    # Compute Yn3
    Y_n3 = (self.beta_31 * Z + sum5 + sum6 + X[:,0] * X[:,1] * np.sin(U_n1 * U_n2)) 

    Y = np.concatenate((Y_n1.reshape(-1, 1), Y_n2.reshape(-1, 1),Y_n3.reshape(-1, 1)), axis=1) 
    
    return Y

  def GenerateM(self, X, U, Y, single = True):
      n = X.shape[0]
      M = np.zeros((n, 3))
      M_lamda = np.zeros((n, 3))

      for i in range(n):
          sum1 = 0
          for p in range(1,6):
            for p_2 in range(1,6):
              sum1 += X[i,p-1] * np.power(X[i,p_2-1],2)
            
          sum2 = 0
          for p in range(1,6):
            for p_2 in range(1,6):
              sum2 += X[i,p-1] * X[i,p_2-1]

          sum3 = 0
          for p in range(1,6):
              sum3 += p * X[i,p-1] 

          sum4 = 0
          for p in range(1,6):
            for p_2 in range(1,6):
              for p_3 in range(1,6):
                sum4 += X[i,p-1] * X[i,p_2-1] * X[i,p_3-1]

          M_lamda[i][0] = np.exp(X[i, :]).sum() + sum1 + np.sin(U[i, 0])**3 + U[i, 1] + np.exp(Y[i, 0])

          M_lamda[i][1] = ((X[i, :]**3).sum() + sum2 + U[i, 0] + (Y[i, 0]**3)/2 + Y[i, 1])

          M_lamda[i][2] = (sum3 + sum4 + np.sin(U[i, 0]) * U[i, 1] + Y[i, 0] + np.exp(Y[i, 1]))

      # calculate 1 - Maskrate percentile
      lambda1 = np.percentile(M_lamda[:,0], 100 * (1-self.MaskRate))
      lambda2 = np.percentile(M_lamda[:,1], 100 * (1-self.MaskRate))
      lambda3 = np.percentile(M_lamda[:,2], 100 * (1-self.MaskRate))
          
      for i in range(self.N):
          if (np.exp(X[i, :]).sum() + sum1 + np.sin(U[i, 0])**3 + U[i, 1] + np.exp(Y[i, 0])) > lambda1:
            M[i][0] = 1 - single
          else:
            M[i][0] = 0
          
          if ((X[i, :]**3).sum() + sum2 + U[i, 0] + (Y[i, 0]**3)/2 + Y[i, 1]) > lambda2:
            M[i][1] =  1 - single
          else:
            M[i][1] =  0

          if (sum3 + sum4 + np.sin(U[i, 0]) * U[i, 1] + Y[i, 0] + np.exp(Y[i, 1])) > lambda3:
            M[i][2] =  1
          else:
            M[i][2] =  0

      return M

  def GenerateS(self, Z):
    #add strata index to the data, each strata has 100 samples, 50 Z = 1 and 50 Z = 0
    S = np.zeros((self.N,1))
    Z0_counter = 0
    Z0_index = 0
    Z1_counter = 0
    Z1_index = 0

    #once Z0_counter = 50 ---> Z0_index += 1 and Z0_counter = 0
    for i in range(self.N):
        if Z[i] == 0:
            Z0_counter += 1
            if Z0_counter == self.N_S / 2:
                Z0_index += 1
                Z0_counter = 0
            S[i] = Z0_index
        else:
            Z1_counter += 1
            if Z1_counter == self.N_S / 2:
                Z1_index += 1
                Z1_counter = 0
            S[i] = Z1_index
    
    return S

  def GenerateData(self):  
    # Generate X
    X = self.GenerateX()

    # Generate Z
    Z = self.GenerateZ()

    # Generate U
    U = self.GenerateU()

    # Generate Y
    Y = self.GenerateY(X, U, Z)

    # Generate M
    M = self.GenerateM(X, U, Y)

    # Generate S
    S = self.GenerateS(Z)

    return X, Z, U, Y, M, S

  def StoreData(self,file):
    # Generate data
    X, Z, U, Y, M, S = self.GenerateData()

    # Combine all generated variables into a single matrix
    data = np.concatenate((X, Z, U, Y, M, S), axis=1) 

    # Store data
    np.savetxt(file, data, delimiter=",")

    # Print message
    print("Data stored in SimulatedData.csv")

  def ReadData(self,file):
    # Read data
    data = np.genfromtxt(file, delimiter=',')
    # Splite into X, Z, U, Y, M, S
    X = data[:, :5]
    Z = data[:, 5]
    U = data[:, 6:8]
    Y = data[:, 8:11]
    M = data[:, 11:14]
    S = data[:, 14]

    return X, Z, U, Y, M, S