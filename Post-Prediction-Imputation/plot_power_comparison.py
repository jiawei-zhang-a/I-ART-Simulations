import numpy as np
import pandas as pd
from analysis_power import read_npz_files
from Visulization import plot_all, plot_size, plot_types, plot_unobserved


def main(type):
    power_data = []
    power_data_with_U = [] 

    for coef in np.arange(0.00,0.2,0.05):
        row_power = [coef]
        row_with_U_power = [coef]
        for directory in ["Result/HPC_power_1000_%s/%f" % (type,coef),
                          "Result/HPC_power_2000_%s/%f" % (type,coef)]:
            results = read_npz_files(directory)
            row_power.extend([results['median_power'], results['lr_power'], results['xgboost_power'], results['oracle_power']])

            results_with_U = read_npz_files(directory.replace("HPC_power_", "HPC_power_unobserved_"))
            row_with_U_power.extend([results_with_U['median_power'], results_with_U['lr_power'], results_with_U['xgboost_power'], results_with_U['oracle_power']])

        power_data.append(row_power)
        power_data_with_U.append(row_with_U_power)

    plot(power_data, power_data_with_U)




def plot(data, data_with_U):
    plot_size.plot_results(data)
    plot_types.plot_results(data, data_with_U)

main("single")


