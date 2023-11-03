import pytest
import sys
import numpy as np
sys.path.append('../')

def test_iartest():
    import iart
    Z = [0,0,1,1]
    X = [0,1,0,1]
    Y = [0,np.nan,np.nan,1]
    result = iart.iartest(Z,X,Y)
    print(result)
