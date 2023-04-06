import numpy as np
import pandas as pd
import pytest
from one_shot_test import OneShotTest

@pytest.fixture
def one_shot_instance():
    return OneShotTest(N=200)

def test_split_df(one_shot_instance):
    df = pd.DataFrame(np.random.rand(200, 10))
    index_S = 5
    df_set1, df_set2 = one_shot_instance.split_df(df, index_S)
    
    assert df_set1.shape == (100, 5)
    assert df_set2.shape == (100, 5)

def test_T(one_shot_instance):
    z = np.random.randint(0, 2, 100)
    y = np.random.rand(100)
    t = one_shot_instance.T(z, y)
    
    assert isinstance(t, float)

def test_getT(one_shot_instance):
    class DummyG:
        def transform(self, df):
            return np.random.rand(df.shape[0], 2)

    G = DummyG()
    df = pd.DataFrame(np.random.rand(100, 10))
    indexY = 5
    t = one_shot_instance.getT(G, df, indexY)

    assert isinstance(t, float)

def test_worker(one_shot_instance):
    class DummyG:
        def fit(self, df):
            pass

        def transform(self, df):
            return np.random.rand(df.shape[0], 2)

    G1 = DummyG()
    G2 = DummyG()
    X = np.random.rand(100, 5)
    Y_masked = np.random.rand(100, 3)
    S = np.random.rand(100, 2)
    t1_obs = 100
    t2_obs = 200
    L = 1000
    verbose = False

    p1, p2 = one_shot_instance.worker((X, Y_masked, S, G1, G2, t1_obs, t2_obs, L, verbose))

    assert isinstance(p1, float)
    assert isinstance(p2, float)

def test_one_shot_test(one_shot_instance):
    class DummyG:
        def fit(self, df):
            pass

        def transform(self, df):
            return np.random.rand(df.shape[0], 2)

    G1 = DummyG()
    G2 = DummyG()
    Z = np.random.randint(0, 2, (100, 1))
    X = np.random.rand(100, 5)
    M = np.random.randint(0, 2, (100, 3))
    Y = np.random.rand(100, 3)
    S = np.random.rand(100, 2)
    L = 1000
    verbose = False

    p1, p2 = one_shot_instance.one_shot_test(Z, X, M, Y, S, G1, G2, L, verbose)

    assert isinstance(p1, float)
    assert isinstance(p2, float)
