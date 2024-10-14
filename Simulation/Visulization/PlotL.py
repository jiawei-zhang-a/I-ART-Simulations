import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from ReadData import  read_npz_files_L

import os

def plot_convergence(t_obs, t_sim, title, output_dir="convergence_plots"):
    n_simulations = len(t_sim)
    p_values = np.zeros(n_simulations)

    # Compute running p-values
    for i in range(1, n_simulations + 1):
        p_values[i-1] = np.mean(t_sim[:i] >= t_obs)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create the plot
    plt.figure(figsize=(10,6))
    plt.plot(np.arange(1, n_simulations + 1), p_values, label="P-value")
    plt.axhline(np.mean(t_sim >= t_obs), color='r', linestyle='--', label="Final P-value")
    plt.xlabel('Number of Simulations')
    plt.ylabel('P-value')
    plt.title(title)
    plt.legend()

    # Save the plot to the specified folder
    output_filepath = os.path.join(output_dir, f"{title.replace(' ', '_').replace('/', '-')}.png")
    plt.savefig(output_filepath)

    # Close the plot to avoid showing it
    plt.close()


def plot(range, range_small, path, path_small, title, title_small, multiple=False):

    # Convergence plots for larger size
    for coef in range:
        if coef != 0.0:
            continue
        for directory in [f'{path}/{coef}']:
            results = read_npz_files_L(directory, small_size=False, multiple=multiple)

            # Convergence plot for each model
            if results.get('median_obs') is not None and results.get('median_sim') is not None:
                if len(results['median_obs']) > 0 and len(results['median_sim']) > 0:
                    plot_convergence(results['median_obs'], results['median_sim'], f"Median Model Convergence for coef {coef}")
            if results.get('LR_obs') is not None and results.get('LR_sim') is not None:
                if len(results['LR_obs']) > 0 and len(results['LR_sim']) > 0:
                    plot_convergence(results['LR_obs'], results['LR_sim'], f"LR Model Convergence for coef {coef}")
            if results.get('lightgbm_obs') is not None and results.get('lightgbm_sim') is not None:
                if len(results['lightgbm_obs']) > 0 and len(results['lightgbm_sim']) > 0:
                    plot_convergence(results['lightgbm_obs'], results['lightgbm_sim'], f"LightGBM Model Convergence for coef {coef}")
            if results.get('oracle_obs') is not None and results.get('oracle_sim') is not None:
                if len(results['oracle_obs']) > 0 and len(results['oracle_sim']) > 0:
                    plot_convergence(results['oracle_obs'], results['oracle_sim'], f"Oracle Model Convergence for coef {coef}")

    # Convergence plots for smaller size
    for coef in range_small:
        if coef != 0.0:
            continue
        for directory in [f'{path_small}/{coef}']:
            results = read_npz_files_L(directory, small_size=True, multiple=multiple)

            # Convergence plot for each model
            if results.get('median_obs') is not None and results.get('median_sim') is not None:
                if len(results['median_obs']) > 0 and len(results['median_sim']) > 0:
                    plot_convergence(results['median_obs'], results['median_sim'], f"Median Model Convergence for coef {coef} (Small)")
            if results.get('LR_obs') is not None and results.get('LR_sim') is not None:
                if len(results['LR_obs']) > 0 and len(results['LR_sim']) > 0:
                    plot_convergence(results['LR_obs'], results['LR_sim'], f"LR Model Convergence for coef {coef} (Small)")
            if results.get('xgboost_obs') is not None and results.get('xgboost_sim') is not None:
                if len(results['xgboost_obs']) > 0 and len(results['xgboost_sim']) > 0:
                    plot_convergence(results['xgboost_obs'], results['xgboost_sim'], f"XGBoost Model Convergence for coef {coef} (Small)")
            if results.get('oracle_obs') is not None and results.get('oracle_sim') is not None:
                if len(results['oracle_obs']) > 0 and len(results['oracle_sim']) > 0:
                    plot_convergence(results['oracle_obs'], results['oracle_sim'], f"Oracle Model Convergence for coef {coef} (Small)")


def main_pic_generator():

    plot(np.arange(0.0,0.42,0.07), np.arange(0,1.5,0.25), "../Power/timeL/HPC_power_1000_model1", "../Power/timeL/HPC_power_50_model1", "Size1000_Model1", "Size50_Model1")
    plot(np.arange(0.0,0.96,0.16), np.arange(0.0,4.8,0.8), "../Power/timeL/HPC_power_1000_model2", "../Power/timeL/HPC_power_50_model2", "Size1000_Model2", "Size50_Model2")
    plot(np.arange(0.0,0.36,0.06), np.arange(0.0,1.5,0.25), "../Power/timeL/HPC_power_1000_model3", "../Power/timeL/HPC_power_50_model3", "Size1000_Model3", "Size50_Model3")
    plot(np.arange(0.0,0.36,0.06), np.arange(0.0,1.5,0.25), "../Power/timeL/HPC_power_1000_model4", "../Power/timeL/HPC_power_50_model4", "Size1000_Model4", "Size50_Model4")  
    #plot(np.arange(0.2, 0.3, 0.05), np.arange(1.5, 2.5, 0.5), "../Data/Result/HPC_power_1000_Model5", "../Data/Result/HPC_power_50_Model5", "Size1000_Model5", "Size50_Model5", multiple=True)


main_pic_generator()