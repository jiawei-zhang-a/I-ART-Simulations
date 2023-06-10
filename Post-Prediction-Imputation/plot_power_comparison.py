import numpy as np
import pandas as pd
from analysis_power import read_npz_files
from Visulization import plot_all, plot_size, plot_types, plot_unobserved, plot_only, plot_covariance


def main(type):
    power_data_with_U = [] 

    for coef in np.arange(0.0,0.5,0.05):
        row_with_U_power = [coef]
        for directory in ["Result/HPC_power_unobserved_1000_single/%f" % (coef)]:
            results = read_npz_files(directory)
            row_with_U_power.extend([results['median_power'], results['lr_power'], results['xgboost_power'], results['oracle_power']])

        power_data_with_U.append(row_with_U_power)

    plot( power_data_with_U)

def plot(data_with_U):
    plot_only.plot_results(data_with_U)
    
def main2(type):
    Power_data = []
    Power_data_with_LR_adjusted = []
    Power_data_with_XGBoost_adjusted = []

    for coef in np.arange(0.0,0.5,0.05):
        row_power = [coef]
        row_power_with_LR_adjusted = [coef]
        row_power_with_XGBoost_adjusted = [coef]
        for directory in ["Result/HPC_power_unobserved_1000_single/%f" % (coef)]:
            results = read_npz_files(directory)
            row_power.extend([results['median_power'], results['lr_power'], results['xgboost_power'], results['oracle_power']])

        for directory in [ "Result/HPC_power_unobserved_1000_LR_single/%f" % (coef)]:
            results = read_npz_files(directory)
            row_power_with_LR_adjusted.extend([results['median_power'], results['lr_power'], results['xgboost_power'], results['oracle_power']])

        for directory in [ "Result/HPC_power_unobserved_1000_xgboost_single/%f" % (coef)]:
            results = read_npz_files(directory)
            row_power_with_XGBoost_adjusted.extend([results['median_power'], results['lr_power'], results['xgboost_power'], results['oracle_power']])

        Power_data.append(row_power)
        Power_data_with_LR_adjusted.append(row_power_with_LR_adjusted)
        Power_data_with_XGBoost_adjusted.append(row_power_with_XGBoost_adjusted)
    print(Power_data)
    print(Power_data_with_LR_adjusted)
    print(Power_data_with_XGBoost_adjusted)
    
    plot_covariance.plot_results(Power_data, Power_data_with_LR_adjusted, Power_data_with_XGBoost_adjusted)

main2("single")


