import numpy as np
import os
from statsmodels.stats.multitest import multipletests

def bonferroni_correction(p_values, alpha=0.05):
    n = len(p_values)
    adjusted_p_values = [min(p * n, 1.0) for p in p_values]
    return [p <= alpha for p in adjusted_p_values]

def read_npz_files(directory,small_size=False, multiple=False, type="original"):
    summed_p_values_median = None
    summed_p_values_LR = None
    summed_p_values_lightgbm = None
    summed_p_values_xgboost = None
    summed_p_values_oracle = None

    N_p_values_median = 0
    N_p_values_LR = 0
    N_p_values_lightgbm = 0
    N_p_values_xgboost = 0
    N_p_values_oracle = 0

    summed_p_values_median = 0
    summed_p_values_LR = 0
    summed_p_values_lightgbm = 0
    summed_p_values_xgboost = 0
    summed_p_values_oracle = 0

    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            filepath = os.path.join(directory, filename)
            p_values = np.load(filepath)

            reject = p_values
            reject = any(reject)
            if "p_values_median" in filename and 'p_values_medianadjusted' not in filename:
                N_p_values_median += 1
                summed_p_values_median += reject
            elif "p_values_LR" in filename:
                N_p_values_LR += 1
                summed_p_values_LR += reject
            elif ("p_values_lightGBM"  in filename) or ("p_values_lightgbm" in filename):
                N_p_values_lightgbm += 1
                summed_p_values_lightgbm += reject
            elif "p_values_xgboost" in filename:
                N_p_values_xgboost += 1
                summed_p_values_xgboost += reject
            elif "p_values_oracle" in filename:
                N_p_values_oracle += 1
                summed_p_values_oracle += reject

    if N_p_values_median != 0:
        rejection_rate_median = summed_p_values_median / N_p_values_median
    else:
        rejection_rate_median = -1
    if N_p_values_LR != 0:
        rejection_rate_LR = summed_p_values_LR / N_p_values_LR
    else:
        rejection_rate_LR = -1
    if N_p_values_lightgbm != 0:
        rejection_rate_lightgbm = summed_p_values_lightgbm / N_p_values_lightgbm
    else:
        rejection_rate_lightgbm = -1
    if N_p_values_xgboost != 0:
        rejection_rate_xgboost = summed_p_values_xgboost / N_p_values_xgboost
    else:
        rejection_rate_xgboost = -1
    if N_p_values_oracle != 0:
        rejection_rate_oracle = summed_p_values_oracle / N_p_values_oracle
    else:
        rejection_rate_oracle = -1

    results = {
        'median_power': rejection_rate_median,
        'lr_power': rejection_rate_LR,
        'xgboost_power':    rejection_rate_xgboost,
        'oracle_power': rejection_rate_oracle,
        'lightgbm_power': rejection_rate_lightgbm,
    }

    return results     