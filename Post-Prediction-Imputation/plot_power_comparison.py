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
            row_with_U_power.extend([results['median_power'], results['lr_power'], results['xgboost_power']])

        power_data_with_U.append(row_with_U_power)

    plot( power_data_with_U)

def plot(data_with_U):
    plot_only.plot_results(data_with_U)
    
def main2(type):
    Power_data = []

    for coef in np.arange(0.0,0.2,0.02):
        row_power = [coef]
        for directory in ["Result/HPC_power_50_single/%f" % (coef)]:
            results = read_npz_files(directory)
            row_power.extend([results['median_power'], results['lr_power'], results['xgboost_power']])

        for directory in [ "Result/HPC_power_2000_single/%f" % (coef)]:
            results = read_npz_files(directory)
            row_power.extend([results['median_power'], results['lr_power'], results['xgboost_power']])

        for directory in [ "Result/HPC_power_20000_single/%f" % (coef)]:
            results = read_npz_files(directory)
            row_power.extend([results['median_power'], results['lr_power'], results['xgboost_power']])

        Power_data.append(row_power)
    print(Power_data)

    plot_size.plot_results(Power_data)

main2("single")


