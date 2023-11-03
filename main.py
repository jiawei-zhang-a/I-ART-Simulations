import numpy as np

def test_iartest():
    from i_art import iartest
    Z = [0,0,1,1]
    X = [0,1,0,1]
    Y = [0,np.nan,np.nan,1]
    result = iartest(Z,X,Y)
    print(result)

test_iartest()