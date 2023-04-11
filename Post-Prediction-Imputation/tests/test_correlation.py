
import numpy as np


def test_get_corr():

    y_truth = np.array([[1, 1, 1], [1, 1, 1]])
    y_imputed = np.array([[1, 1, 1], [1, 1, 1]])
    # Constants for testing
    val = np.corrcoef(y_imputed, y_truth)[0, 1]
    print(val)

test_get_corr()