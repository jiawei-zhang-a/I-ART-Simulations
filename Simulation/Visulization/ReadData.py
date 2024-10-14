import numpy as np
import os

def read_npz_files(directory, small_size=False, multiple=False, type="original"):
    # Initialize variables to sum the results across files
    total_rejections = {
        'median': 0,
        'LR': 0,
        'lightgbm': 0,
        'xgboost': 0,
        'oracle': 0
    }

    counts = {
        'median': 0,
        'LR': 0,
        'lightgbm': 0,
        'xgboost': 0,
        'oracle': 0
    }

    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            filepath = os.path.join(directory, filename)
            # Load the dictionary from the .npy file
            data = np.load(filepath, allow_pickle=True).item()
            
            # Extract the p-values from the dictionary
            reject = data.get('reject', False)

            # Tally the rejections for each model type
            if "results_median" in filename:
                counts['median'] += 1
                total_rejections['median'] += reject
            elif "results_LR" in filename:
                counts['LR'] += 1
                total_rejections['LR'] += reject
            elif "results_lightgbm" in filename:
                counts['lightgbm'] += 1
                total_rejections['lightgbm'] += reject
            elif "results_xgboost" in filename:
                counts['xgboost'] += 1
                total_rejections['xgboost'] += reject
            elif "results_oracle" in filename:
                counts['oracle'] += 1
                total_rejections['oracle'] += reject

    # Calculate rejection rates (power) for each model
    results = {}
    for key in total_rejections:
        if counts[key] > 0:
            results[f"{key}_power"] = total_rejections[key] / counts[key]
        else:
            results[f"{key}_power"] = -1  # If no results were processed for this model

    return results
