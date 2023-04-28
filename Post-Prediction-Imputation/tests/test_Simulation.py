import pytest
import numpy as np
import sys
sys.path.append('../')
from Simulation import DataGenerator

# Constants for testing
N = 100
N_T = 10
N_S = 5
beta_11 = 0.5
beta_12 = 1.0
beta_21 = 1.5
beta_22 = 2.0
beta_23 = 2.5
beta_31 = 3.0
MaskRate = 0.2
Unobserved = True

# Create a DataGenerator instance for testing
dg = DataGenerator(N, N_T, N_S, beta_11, beta_12, beta_21, beta_22, beta_23, beta_31, MaskRate, Unobserved)

# Test functions
def test_generate_x():
    X = dg.GenerateX()
    assert X.shape == (N, 5)

def test_generate_u():
    U = dg.GenerateU()
    assert U.shape == (N, 2)

def test_generate_s():
    S = dg.GenerateS()
    assert S.shape == (N, 1)

def test_generate_z():
    Z = dg.GenerateZ()
    assert Z.shape == (N, 1)
    
def test_generate_y():
    X = dg.GenerateX()
    U = dg.GenerateU()
    Z = dg.GenerateZ()
    Y = dg.GenerateY(X, U, Z)
    assert Y.shape == (N, 3)

def test_generate_m():
    X = dg.GenerateX()
    U = dg.GenerateU()
    Y = dg.GenerateY(X, U, dg.GenerateZ())
    M = dg.GenerateM(X, U, Y)
    assert M.shape == (N, 3)

def test_generate_data():
    X, Z, U, Y, M, S = dg.GenerateData()
    assert X.shape == (N, 5)
    assert Z.shape == (N, 1)
    assert U.shape == (N, 2)
    assert Y.shape == (N, 3)
    assert M.shape == (N, 3)
    assert S.shape == (N, 1)

def test_store_and_read_data(tmpdir):
    data_file = tmpdir.join("SimulatedData.csv")
    dg.StoreData(data_file)
    X, Z, U, Y, M, S = dg.ReadData(data_file)

    assert X.shape == (N, 5)
    assert Z.shape == (N,)
    assert U.shape == (N, 2)
    assert Y.shape == (N, 3)
    assert M.shape == (N, 3)
    assert S.shape == (N,)


