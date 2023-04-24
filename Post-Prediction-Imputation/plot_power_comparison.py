import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis_power import read_and_print_npz_files
from Visulization import plot_all, plot_size, plot_types, plot_unobserved


def main(type):
    data = []
    data_with_U = []

    for coef in np.arange(0.02, 0.2, 0.02):
        row = [coef]
        row_with_U = [coef]
        for directory in ["Result/HPC_power_1000_%s/%f" % (type,coef),
                          "Result/HPC_power_2000_%s/%f" % (type,coef)]:
            results = read_and_print_npz_files(directory)
            row.extend([results['median'], results['lr'], results['xgboost']])
            
            results_with_U = read_and_print_npz_files(directory.replace("HPC_power_", "HPC_power_unobserved_"))
            row_with_U.extend([results_with_U['median'], results_with_U['lr'], results_with_U['xgboost']])

        data.append(row)
        data_with_U.append(row_with_U)

    print(data)
    print(data_with_U)

    plot_all.plot_results(data, data_with_U)
    plot_size.plot_results(data, data_with_U)
    plot_types.plot_results(data, data_with_U)
    plot_unobserved.plot_results(data, data_with_U)



main("single")
main("multiple")