import numpy as np
import pandas as pd
from analysis_power import read_npz_files
from Visulization import plot_all, plot_size, plot_types, plot_unobserved


def main(type):
    data = []
    data_with_U = []

    for coef in np.arange(0.02, 0.2, 0.02):
        row = [coef]
        row_with_U = [coef]
        for directory in ["Result/HPC_power_1000_%s/%f" % (type,coef),
                          "Result/HPC_power_2000_%s/%f" % (type,coef)]:
            results = read_npz_files(directory)
            row.extend([results['median'], results['lr'], results['xgboost']])
            
            results_with_U = read_npz_files(directory.replace("HPC_power_", "HPC_power_unobserved_"))
            row_with_U.extend([results_with_U['median'], results_with_U['lr'], results_with_U['xgboost']])

        data.append(row)
        data_with_U.append(row_with_U)

    print(data)
    print(data_with_U)
    plot(data, data_with_U)

"""
data = [[0.02, 0.1075, 0.0865, 0.083, 0.1305, 0.1205, 0.108], [0.04, 0.182, 0.164, 0.158, 0.2565, 0.2165, 0.192], [0.06, 0.257, 0.2305, 0.209, 0.3975, 0.3215, 0.2875], [0.08, 0.3535, 0.2905, 0.266, 0.591, 0.476, 0.422], [0.1, 0.4875, 0.397, 0.361, 0.7485, 0.6185, 0.5455], [0.12000000000000001, 0.598, 0.4975, 0.4365, 0.8655, 0.75, 0.6705], [0.13999999999999999, 0.694, 0.5675, 0.506, 0.931, 0.8335, 0.7545], [0.16, 0.783, 0.6605, 0.5855, 0.9585, 0.886, 0.8045], [0.18, 0.8415, 0.7295, 0.643, 0.985, 0.956, 0.894]]
data_with_U = [[0.02, 0.0735, 0.077, 0.0765, 0.097, 0.0895, 0.088], [0.04, 0.1005, 0.0925, 0.0845, 0.1235, 0.117, 0.101], [0.06, 0.137, 0.1365, 0.121, 0.2015, 0.1815, 0.149], [0.08, 0.178, 0.167, 0.148, 0.2855, 0.267, 0.218], [0.1, 0.238, 0.222, 0.192, 0.402, 0.368, 0.3], [0.12000000000000001, 0.3275, 0.3065, 0.2605, 0.5275, 0.4795, 0.3855], [0.13999999999999999, 0.406, 0.376, 0.322, 0.6115, 0.568, 0.458], [0.16, 0.4535, 0.4035, 0.3395, 0.7135, 0.6575, 0.5365], [0.18, 0.552, 0.493, 0.408, 0.7845, 0.7265, 0.606]]
"""
def plot(data, data_with_U):
    plot_all.plot_results(data, data_with_U)
    plot_size.plot_results(data)
    plot_types.plot_results(data, data_with_U)
    plot_unobserved.plot_results(data, data_with_U)

#main("single")
main("multi")

