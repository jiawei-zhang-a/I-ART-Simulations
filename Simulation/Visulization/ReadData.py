import numpy as np
import os
from statsmodels.stats.multitest import multipletests


def read_npz_files(directory,small_size=False,type="original"):
    summed_p_values_median = None
    summed_p_values_LR = None
    summed_p_values_lightGBM = None
    summed_p_values_xgboost = None
    summed_p_values_oracle = None
    if type == "original":
        N = int(len(os.listdir(directory)) / 4)
    else:
        N = int(len(os.listdir(directory)))

    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            filepath = os.path.join(directory, filename)
            p_values = np.load(filepath)

            if "p_values_median" in filename:
                if summed_p_values_median is None:
                    summed_p_values_median = (p_values<= 0.05).astype(int)
                else:
                    summed_p_values_median += (p_values<= 0.05).astype(int)
            elif "p_values_LR" in filename:
                if summed_p_values_LR is None:
                    summed_p_values_LR = (p_values<= 0.05).astype(int)
                else:
                    summed_p_values_LR += (p_values<= 0.05).astype(int)
            elif "p_values_lightGBM" in filename:
                if summed_p_values_lightGBM is None:
                    summed_p_values_lightGBM = (p_values<= 0.05).astype(int)
                else:
                    summed_p_values_lightGBM += (p_values<= 0.05).astype(int)
            elif "p_values_xgboost" in filename:
                if summed_p_values_xgboost is None:
                    summed_p_values_xgboost = (p_values<= 0.05).astype(int)
                else:
                    summed_p_values_xgboost += (p_values<= 0.05).astype(int)
            elif "p_values_oracle" in filename:
                if summed_p_values_oracle is None:
                    summed_p_values_oracle = (p_values<= 0.05).astype(int)
                else:
                    summed_p_values_oracle += (p_values<= 0.05).astype(int)

    if summed_p_values_median is None:
        summed_p_values_median = np.zeros(1)
    if summed_p_values_LR is None:
        summed_p_values_LR = np.zeros(1)
    if summed_p_values_lightGBM is None:
        summed_p_values_lightGBM = np.zeros(1)
    if summed_p_values_xgboost is None:
        summed_p_values_xgboost = np.zeros(1)
    if summed_p_values_oracle is None:
        summed_p_values_oracle = np.zeros(1)


    if small_size:
        results = {
            'median_power': summed_p_values_median[0] / N,
            'lr_power': summed_p_values_LR[0] / N,
            'xgboost_power': summed_p_values_xgboost[0] / N,
            'oracle_power': summed_p_values_oracle[0] / N,
        }
    else:
        results = {
            'median_power': summed_p_values_median[0] / N,
            'lr_power': summed_p_values_LR[0] / N,
            'lightGBM_power': summed_p_values_lightGBM[0] / N,
            'oracle_power': summed_p_values_oracle[0] / N,
            
        }

    return results


def bonferroni_correction(p_values, alpha=0.05):
    n = len(p_values)
    adjusted_p_values = [min(p * n, 1.0) for p in p_values]
    return [p <= alpha for p in adjusted_p_values]

def holm_bonferroni_correction(p_values, alpha=0.05):
    sorted_p_values = sorted((p, i) for i, p in enumerate(p_values))
    n = len(p_values)
    adjusted_p_values = [0] * n
    significant = [False] * n
    for rank, (p, original_index) in enumerate(sorted_p_values):
        adjusted_p_value = min(p * (n - rank), 1.0)
        adjusted_p_values[original_index] = adjusted_p_value
        if adjusted_p_value <= alpha:
            significant[original_index] = True
        else:
            break  # No need to continue once a test fails
    return significant

def read_npz_files_main(directory,small_size=False, multiple=False):
    summed_p_values_median = None
    summed_p_values_LR = None
    summed_p_values_lightGBM = None
    summed_p_values_xgboost = None
    summed_p_values_oracle = None

    N = int(len(os.listdir(directory)) / 4)

    if multiple:
        summed_p_values_median = 0
        summed_p_values_LR = 0
        summed_p_values_lightGBM = 0
        summed_p_values_xgboost = 0
        summed_p_values_oracle = 0
        for filename in os.listdir(directory):
            if filename.endswith(".npy"):
                filepath = os.path.join(directory, filename)
                p_values = np.load(filepath)
                reject = holm_bonferroni_correction(p_values[0:3], alpha=0.05)
                reject = any(reject)
                if "p_values_median" in filename:
                    summed_p_values_median += reject
                elif "p_values_LR" in filename:
                    summed_p_values_LR += reject
                elif "p_values_lightGBM" in filename:
                    summed_p_values_lightGBM += reject
                elif "p_values_xgboost" in filename:
                    summed_p_values_xgboost += reject
                elif "p_values_oracle" in filename:
                    summed_p_values_oracle += reject

        if small_size:
            results = {
                'median_power': summed_p_values_median / N,
                'lr_power': summed_p_values_LR / N,
                'xgboost_power': summed_p_values_xgboost / N,
                'oracle_power': summed_p_values_oracle / N,
            }
        else:
            results = {
                'median_power': summed_p_values_median / N,
                'lr_power': summed_p_values_LR / N,
                'lightGBM_power': summed_p_values_lightGBM / N,
                'oracle_power': summed_p_values_oracle / N,
            }

        return results    
    else:
        for filename in os.listdir(directory):
            if filename.endswith(".npy"):
                filepath = os.path.join(directory, filename)
                p_values = np.load(filepath)

                if "p_values_median" in filename:
                    if summed_p_values_median is None:
                        summed_p_values_median = (p_values<= 0.05).astype(int)
                    else:
                        summed_p_values_median += (p_values<= 0.05).astype(int)
                elif "p_values_LR" in filename:
                    if summed_p_values_LR is None:
                        summed_p_values_LR = (p_values<= 0.05).astype(int)
                    else:
                        summed_p_values_LR += (p_values<= 0.05).astype(int)
                elif "p_values_lightGBM" in filename:
                    if summed_p_values_lightGBM is None:
                        summed_p_values_lightGBM = (p_values<= 0.05).astype(int)
                    else:
                        summed_p_values_lightGBM += (p_values<= 0.05).astype(int)
                elif "p_values_xgboost" in filename:
                    if summed_p_values_xgboost is None:
                        summed_p_values_xgboost = (p_values<= 0.05).astype(int)
                    else:
                        summed_p_values_xgboost += (p_values<= 0.05).astype(int)
                elif "p_values_oracle" in filename:
                    if summed_p_values_oracle is None:
                        summed_p_values_oracle = (p_values<= 0.05).astype(int)
                    else:
                        summed_p_values_oracle += (p_values<= 0.05).astype(int)

        if small_size:
            results = {
                'median_power': summed_p_values_median[0] / N,
                'lr_power': summed_p_values_LR[0] / N,
                'xgboost_power': summed_p_values_xgboost[0] / N,
                'oracle_power': summed_p_values_oracle[0] / N,
            }
        else:
            results = {
                'median_power': summed_p_values_median[0] / N,
                'lr_power': summed_p_values_LR[0] / N,
                'lightGBM_power': summed_p_values_lightGBM[0] / N,
                'oracle_power': summed_p_values_oracle[0] / N,
            }

        return results