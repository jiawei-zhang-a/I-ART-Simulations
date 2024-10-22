import numpy as np
import os
import matplotlib.pyplot as plt

def find_least_L(t_obs, t_sim, p_values, error_threshold=0.01):
    """
    Find the least L for convergence to p-values within the given error threshold.
    
    Parameters:
    - t_obs: Observed test statistic
    - t_sim: Simulated test statistics
    - p_values: Current p_values from the file
    - error_threshold: Threshold for p-value convergence
    
    Returns:
    - L: Least L where p-values converge within the error threshold
    """
    # Initialize L to the maximum number of simulations
    L = len(t_sim)
    p_values = np.zeros(L)
    final_p_value = np.mean(t_sim >= t_obs)
    
    # Compute running p-values for the model
    for i in range(1, L + 1):
        p_values[i-1] = np.mean(t_sim[:i] >= t_obs)
        
        # Check for convergence
        if np.abs(p_values[i-1] - final_p_value) < error_threshold:
            L = i
            break

    return L

def read_npz_files_L(directory, error_threshold=0.01):
    """
    Process .npz files in the directory to calculate the least L for each model type.
    
    Parameters:
    - directory: Directory containing the .npy files
    - error_threshold: Threshold for p-value convergence
    
    Returns:
    - mean_L: The mean of the least L values where p-values converge, or None if no valid L found
    """
    least_Ls = {
        'median': [],
        'LR': [],
        'lightgbm': [],
        'xgboost': [],
        'oracle': []
    }
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        data = np.load(filepath, allow_pickle=True).item()
        
        t_obs = data.get('t_obs', False)
        t_sim = data.get('t_sim', False)
        p_values = data.get('p_values', False)

        if t_obs is not False and t_sim is not False and p_values is not False:
            # Determine the model type based on the file name
            if "results_median" in filename:
                least_Ls['median'].append(find_least_L(t_obs, t_sim, p_values, error_threshold))
            elif "results_LR" in filename:
                least_Ls['LR'].append(find_least_L(t_obs, t_sim, p_values, error_threshold))
            elif "results_lightgbm" in filename:
                least_Ls['lightgbm'].append(find_least_L(t_obs, t_sim, p_values, error_threshold))
            elif "results_xgboost" in filename:
                least_Ls['xgboost'].append(find_least_L(t_obs, t_sim, p_values, error_threshold))
            elif "results_oracle" in filename:
                least_Ls['oracle'].append(find_least_L(t_obs, t_sim, p_values, error_threshold))

    # Compute the average of least Ls for each model
    average_least_Ls = {model: np.mean(Ls) for model, Ls in least_Ls.items() if Ls}

    # Return both the dictionary of least Ls and the dictionary of their averages
    return least_Ls, average_least_Ls


def plot(range, range_small, path, path_small, title, title_small, multiple=False):
    # Convergence plots for larger size
    for coef in range:
        if coef != 0.0:
            continue
        for directory in [f'{path}/{coef}']:
            results = read_npz_files_L(directory)

            print(results)

    # Convergence plots for smaller size
    for coef in range_small:
        if coef != 0.0:
            continue
        for directory in [f'{path_small}/{coef}']:
            results = read_npz_files_L(directory)

            print(results)
            

def main_pic_generator():
    plot(np.arange(0.0,0.42,0.07), np.arange(0,1.5,0.25), "../Power/timeL/HPC_power_1000_model1", "../Power/timeL/HPC_power_50_model1", "Size1000_Model1", "Size50_Model1")
    plot(np.arange(0.0,0.96,0.16), np.arange(0.0,4.8,0.8), "../Power/timeL/HPC_power_1000_model2", "../Power/timeL/HPC_power_50_model2", "Size1000_Model2", "Size50_Model2")
    plot(np.arange(0.0,0.36,0.06), np.arange(0.0,1.5,0.25), "../Power/timeL/HPC_power_1000_model3", "../Power/timeL/HPC_power_50_model3", "Size1000_Model3", "Size50_Model3")
    plot(np.arange(0.0,0.36,0.06), np.arange(0.0,1.5,0.25), "../Power/timeL/HPC_power_1000_model4", "../Power/timeL/HPC_power_50_model4", "Size1000_Model4", "Size50_Model4")

main_pic_generator()
