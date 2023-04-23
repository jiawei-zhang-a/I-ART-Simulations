import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis_power import read_and_print_npz_files

def main():
    columns = ['beta', 'I50_Median', 'I50_Linear', 'I50_XGBoost', 'I100_Median', 'I100_Linear', 'I100_XGBoost']
    data = []
    data_with_U = []

    for coef in np.arange(0.02, 0.2, 0.02):
        row = [coef]
        row_with_U = [coef]
        for directory in ["Result/HPC_power_1000_single/%f" % coef,
                          "Result/HPC_power_2000_single/%f" % coef]:
            results = read_and_print_npz_files(directory)
            row.extend([results['median'], results['lr'], results['xgboost']])
            
            results_with_U = read_and_print_npz_files(directory.replace("HPC_power_", "HPC_power_unobserved_"))
            row_with_U.extend([results_with_U['median'], results_with_U['lr'], results_with_U['xgboost']])

        data.append(row)
        data_with_U.append(row_with_U)

    plot_results(data, data_with_U, columns)

def plot_results(data, data_with_U, columns):
    df = pd.DataFrame(data, columns=columns)
    df_with_U = pd.DataFrame(data_with_U, columns=columns)

    plt.figure(figsize=(12, 8))

    colors = {'Median': 'blue', 'Linear': 'red', 'XGBoost': 'green'}
    linestyles = {'I50': '-', 'I100': '--'}

    for col in columns[1:]:
        method = col.split('_')[1]
        dataset = col.split('_')[0]
        linestyle = linestyles[dataset]
        plt.plot(df['beta'], df[col], marker='o', label=col, color=colors[method], linestyle=linestyle)
        plt.plot(df_with_U['beta'], df_with_U[col], marker='o', label=col + " with U", color=colors[method], linestyle=linestyle, alpha=0.5)

    plt.xlabel('Beta')
    plt.ylabel('Absolute Error')
    plt.title('Performance of Imputation Methods for Varying Beta')
    plt.legend()
    plt.grid()

    plt.show()

main()
