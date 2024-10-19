import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ReadData import  read_npz_files_L
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_convergence(t_obs_lr, t_sim_lr, t_obs_xgb, t_sim_xgb, title, output_dir="convergence_plots", log_version=True):
    n_simulations = len(t_sim_lr)  # Assume both models have the same number of simulations
    p_values_lr = np.zeros(n_simulations)
    p_values_xgb = np.zeros(n_simulations)

    # Compute running p-values for LR and XGBoost models
    for i in range(1, n_simulations + 1):
        p_values_lr[i-1] = np.mean(t_sim_lr[:i] >= t_obs_lr)
        p_values_xgb[i-1] = np.mean(t_sim_xgb[:i] >= t_obs_xgb)

    # Starting point (simulation 100)
    start_simulation = 100
    interval = 1

    # Slice the arrays to start from the 100th simulation, with interval of 5
    simulations = np.arange(start_simulation, n_simulations + 1, interval)
    p_values_lr = p_values_lr[start_simulation - 1::interval]  # Slice from index 99 with step size 5
    p_values_xgb = p_values_xgb[start_simulation - 1::interval]  # Slice from index 99 with step size 5


    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Standard plot with both LR and XGBoost lines starting from simulation 100
    plt.figure(figsize=(10, 6))
    plt.plot(simulations, p_values_lr, label="Algo 1 - Linear", color="red")
    plt.plot(simulations, p_values_xgb, label="Algo 1 - GBM", color="green")
    plt.axhline(np.mean(t_sim_lr >= t_obs_lr), color='r', linestyle='--', label="Final P-value(Algo 1 - Linear)")
    plt.axhline(np.mean(t_sim_xgb >= t_obs_xgb), color='green', linestyle='--', label="Final P-value(Algo 1 - GBM)")
    plt.xlabel('Number of Simulations')
    plt.ylabel('P-value')
    plt.title(title)
    plt.legend()

    # Save the plot
    output_filepath = os.path.join(output_dir, f"{title.replace(' ', '_').replace('/', '-')}.png")
    plt.savefig(output_filepath)
    plt.close()

    # Logarithmic version plot (if requested)
    if log_version:
        log_simulation_steps = np.logspace(np.log10(1000), np.log10(n_simulations-100), num=100, dtype=int)
        log_simulation_steps = np.unique(log_simulation_steps)  # Ensure unique steps

        p_values_log_lr = p_values_lr[log_simulation_steps - 1]  # Subset the p-values to match log steps
        p_values_log_xgb = p_values_xgb[log_simulation_steps - 1]  # Subset XGBoost p-values

        # Compute the target (final p-value)
        target_value_lr = np.mean(t_sim_lr >= t_obs_lr)
        target_value_xgb = np.mean(t_sim_xgb >= t_obs_xgb)

        # Plot the absolute difference from the target value
        plt.figure(figsize=(10, 6))
        plt.plot(log_simulation_steps, np.abs(p_values_log_lr - target_value_lr), label="|(Algo 1 - Linear) P-value - Target|", color="red")
        plt.plot(log_simulation_steps, np.abs(p_values_log_xgb - target_value_xgb), label="|(Algo 1 - Boosting) P-value - Target|", color="green")
        plt.axhline(0, color='r', linestyle='--', label="Convergence to (Algo 1 - Linear) Target")
        plt.axhline(0, color='green', linestyle='--', label="Convergence to GBM Target")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Number of Simulations (Log Scale)')
        plt.ylabel('|P-value - Target| (Log Scale)')
        plt.title(f"{title} (Log Convergence)")
        plt.legend()

        # Add a reference line for convergence rate (-1/2 slope)
        x_ref = np.logspace(np.log10(1000), np.log10(n_simulations), num=50)
        y_ref = 1 / np.sqrt(x_ref)
        plt.plot(x_ref, y_ref, 'k--', linewidth=2, label='Convergence Rate (slope = -1/2)')

        # Annotate the reference line with LaTeX-style math notation
        mid_point = len(x_ref) // 2  # Choose the midpoint for annotation
        plt.annotate(r'Convergence Rate$\sim\frac{1}{\sqrt{n}}$, Slope=-1/2', 
                    xy=(x_ref[mid_point], y_ref[mid_point]),
                    xytext=(x_ref[mid_point] * 1.5, y_ref[mid_point] * 2),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    fontsize=10, color='black')
        # Save the log version plot
        output_filepath_log = os.path.join(output_dir, f"{title.replace(' ', '_').replace('/', '-')}_log.png")
        plt.savefig(output_filepath_log)
        plt.close()

def plot(range, range_small, path, path_small, title, title_small, multiple=False):
    # Convergence plots for larger size
    for coef in range:
        if coef != 0.0:
            continue
        for directory in [f'{path}/{coef}']:
            results = read_npz_files_L(directory, small_size=False, multiple=multiple)

            if results.get('LR_obs') is not None and results.get('LR_sim') is not None and \
               results.get('lightgbm_obs') is not None and results.get('lightgbm_sim') is not None:
                plot_convergence(results['LR_obs'], results['LR_sim'], 
                                 results['lightgbm_obs'], results['lightgbm_sim'], 
                                 f"LR and lightgbm Model Convergence for coef {coef}", 
                                 os.path.join("convergence_plots", title))

    # Convergence plots for smaller size
    for coef in range_small:
        if coef != 0.0:
            continue
        for directory in [f'{path_small}/{coef}']:
            results = read_npz_files_L(directory, small_size=True, multiple=multiple)

            if results.get('LR_obs') is not None and results.get('LR_sim') is not None and \
               results.get('xgboost_obs') is not None and results.get('xgboost_sim') is not None:
                plot_convergence(results['LR_obs'], results['LR_sim'], 
                                 results['xgboost_obs'], results['xgboost_sim'], 
                                 f"LR and XGBoost Model Convergence for coef {coef} (Small)", 
                                 os.path.join("convergence_plots", title_small))


def main_pic_generator():

    plot(np.arange(0.0,0.42,0.07), np.arange(0,1.5,0.25), "../Power/timeL/HPC_power_1000_model1", "../Power/timeL/HPC_power_50_model1", "Size1000_Model1", "Size50_Model1")
    plot(np.arange(0.0,0.96,0.16), np.arange(0.0,4.8,0.8), "../Power/timeL/HPC_power_1000_model2", "../Power/timeL/HPC_power_50_model2", "Size1000_Model2", "Size50_Model2")
    plot(np.arange(0.0,0.36,0.06), np.arange(0.0,1.5,0.25), "../Power/timeL/HPC_power_1000_model3", "../Power/timeL/HPC_power_50_model3", "Size1000_Model3", "Size50_Model3")
    plot(np.arange(0.0,0.36,0.06), np.arange(0.0,1.5,0.25), "../Power/timeL/HPC_power_1000_model4", "../Power/timeL/HPC_power_50_model4", "Size1000_Model4", "Size50_Model4")  
    #plot(np.arange(0.2, 0.3, 0.05), np.arange(1.5, 2.5, 0.5), "../Data/Result/HPC_power_1000_Model5", "../Data/Result/HPC_power_50_Model5", "Size1000_Model5", "Size50_Model5", multiple=True)


main_pic_generator()