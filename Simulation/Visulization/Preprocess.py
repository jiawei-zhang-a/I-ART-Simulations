import numpy as np
import pandas as pd
import os
from ReadData import read_npz_files

def save_data(range, range_small, path, path_small, output_filename_main, output_filename_small, multiple=False):
    Power_data = []
    Power_data_small = []

    # Collect data for the main range
    for coef in range:
        row_power = [coef]
        for directory in [path + "/%f" % (coef)]:
            results = read_npz_files(directory, small_size=False, multiple=multiple)
            row_power.extend([results['median_power'], results['lr_power'], results['lightgbm_power'], results['oracle_power']])
        Power_data.append(row_power)

    # Collect data for the small range
    for coef in range_small:
        row_power_small = [coef]
        for directory in [path_small + "/%f" % (coef)]:
            results = read_npz_files(directory, small_size=True, multiple=multiple)
            row_power_small.extend([results['median_power'], results['lr_power'], results['xgboost_power'], results['oracle_power']])
        Power_data_small.append(row_power_small)

    # Ensure the tmp folder exists
    if not os.path.exists("tmp"):
        os.makedirs("tmp")
    
    # Save the main data to one file and the small data to another file in the tmp folder
    pd.to_pickle(Power_data, os.path.join("tmp", output_filename_main))
    pd.to_pickle(Power_data_small, os.path.join("tmp", output_filename_small))

def main_data_saver():
    save_data(np.arange(0.0, 0.42, 0.07), np.arange(0, 1.5, 0.25), 
              "../Data/Simulation/HPC_power_1000_model1", "../Data/Simulation/HPC_power_50_model1", 
              "Size1000_Model1.pkl", "Size50_Model1.pkl")

    save_data(np.arange(0.0, 0.96, 0.16), np.arange(0.0, 4.8, 0.8), 
              "../Data/Simulation/HPC_power_1000_model2", "../Data/Simulation/HPC_power_50_model2", 
              "Size1000_Model2.pkl", "Size50_Model2.pkl")

    save_data(np.arange(0.0, 0.36, 0.06), np.arange(0.0, 1.5, 0.25), 
              "../Data/Simulation/HPC_power_1000_model3", "../Data/Simulation/HPC_power_50_model3", 
              "Size1000_Model3.pkl", "Size50_Model3.pkl")

    save_data(np.arange(0.0, 0.36, 0.06), np.arange(0.0, 1.5, 0.25), 
              "../Data/Simulation/HPC_power_1000_model4", "../Data/Simulation/HPC_power_50_model4", 
              "Size1000_Model4.pkl", "Size50_Model4.pkl")

main_data_saver()
