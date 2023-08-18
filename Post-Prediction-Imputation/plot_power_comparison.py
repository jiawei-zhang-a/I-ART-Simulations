import numpy as np
import pandas as pd
from analysis_power import read_npz_files
import matplotlib.pyplot as plt
import os

def plot_results(data, title, xsticks):
    plt.clf()

    columns = ['beta', 'Imputer_GradientBoosting', 'Imputer_Oracle',"Imputer_Median","Imputer_LR", "Imputer_GradientBoosting-adjusted", "Imputer_Oracle-adjusted", "Imputer_Median-adjusted", "Imputer_LR-adjusted"]

    df = pd.DataFrame(data, columns=columns)

    plt.figure(figsize=(10, 6))

    colors = {'Imputer_GradientBoosting': 'red', 'Imputer_Oracle': 'blue',"Imputer_Median":"green","Imputer_LR":"orange", "Imputer_GradientBoosting-adjusted": 'red', "Imputer_Oracle-adjusted": 'blue', "Imputer_Median-adjusted":'green', "Imputer_LR-adjusted":'orange'}
    linestyles = {'Imputer_GradientBoosting': '--', 'Imputer_Oracle': '--', "Imputer_Median":'--',"Imputer_LR":'--',"Imputer_GradientBoosting-adjusted": '-', "Imputer_Oracle-adjusted": '-' ,"Imputer_Median-adjusted":'-',"Imputer_LR-adjusted":'-'}

    for col in columns[1:]:
        linestyle = linestyles[col]
        method = col.split('_')[1]

        plt.plot(df['beta'], df[col], marker='o', label=method, color=colors[col], linestyle=linestyle)

    plt.xlabel('Beta')
    plt.ylabel('Power')
    plt.title(title)
    plt.legend()
    plt.grid()
    # plt.show()
    if not os.path.exists("pic"):
        os.makedirs("pic")
    plt.xticks(xsticks)
    # Setting y-axis ticks with custom intervals
    y_ticks = [i/100.0 for i in range(0, 105, 20)]  # Starts from 0, ends at 1.05, with an interval of 0.05
    y_ticks.append(0.05)
    plt.yticks(y_ticks)

    plt.savefig("pic/" + title + ".png", format='png', dpi=600)


def main():
    Power_data = []
    Power_data_small = []

    for coef in np.arange(0.0,0.3 ,0.05):
        row_power = [coef]
        for directory in [ "Result/HPC_power_1000_unobserved_interference_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=False)
            row_power.extend([ results['lightGBM_power'], results['oracle_power'], results['median_power'], results['lr_power']])
        for directory in ["Result/HPC_power_1000_unobserved_interference_adjusted_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=False)
            row_power.extend([ results['lightGBM_power'],results['oracle_power'], results['median_power'], results['lr_power'] ])
        Power_data.append(row_power)

    plot_results(Power_data,  "Size-1000, Single: Covariance Adjusted, ", np.arange(0.0,0.3 ,0.05)) 

    for coef in np.arange(0.0,1.2,0.2):
        row_power_small = [coef]
        for directory in ["Result/HPC_power_50_unobserved_interference_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=True)
            row_power_small.extend([results['xgboost_power'], results['oracle_power'], results['median_power'], results['lr_power']])
        for directory in [ "Result/HPC_power_50_unobserved_interference_adjusted_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=True)
            row_power_small.extend([results['xgboost_power'], results['oracle_power'], results['median_power'], results['lr_power']])
        Power_data_small.append(row_power_small)

    plot_results(Power_data_small, "Size-50, Single: Covariance Adjusted, ", np.arange(0.0,1.2,0.2))   
     

main()


