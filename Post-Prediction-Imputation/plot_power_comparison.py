import numpy as np
import pandas as pd
from analysis_power import read_npz_files
import matplotlib.pyplot as plt
import os

def plot_results(data, title):
    plt.clf()

    columns = ['beta', 'Imputer_GradientBoosting', 'Imputer_Oracle', "Imputer_GradientBoosting-adjustment", "Imputer_Oracle-adjustment"]

    df = pd.DataFrame(data, columns=columns)

    plt.figure(figsize=(10, 6))

    colors = {'Imputer_GradientBoosting': 'red', 'Imputer_Oracle': 'blue', "Imputer_GradientBoosting-adjustment": 'red', "Imputer_Oracle-adjustment": 'blue'}
    linestyles = {'Imputer_GradientBoosting': '--', 'Imputer_Oracle': '--', "Imputer_GradientBoosting-adjustment": '-', "Imputer_Oracle-adjustment": '-'}

    for col in columns[1:]:
        linestyle = linestyles[col]
        plt.plot(df['beta'], df[col], marker='o', label=col, color=colors[col], linestyle=linestyle)

    plt.xlabel('Beta')
    plt.ylabel('Power')
    plt.title(title)
    plt.legend()
    plt.grid()
    # plt.show()
    if not os.path.exists("pic"):
        os.makedirs("pic")

    plt.savefig("pic/" + title + ".png", format='png', dpi=600)


def main():
    Power_data = []
    Power_data_small = []

    """for coef in np.arange(0.0,0.42,0.07):
        row_power = [coef]
        for directory in [ "Result/HPC_power_1000_unobserved_linearZ_linearX_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=False,adjustment=False)
            row_power.extend([ results['lightGBM_power'], results['oracle_power'],])
        for directory in ["Result/HPC_power_1000_unobserved_linearZ_linearX_adjustment_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=False,adjustment=True)
            row_power.extend([ results['lightGBM_power'],results['oracle_power'] ])
        Power_data.append(row_power)

    plot_results(Power_data,  "Covariance adjustment, 1model, Size 1000") """

    for coef in np.arange(0,1.5,0.25):
        row_power_small = [coef]
        for directory in ["Result/HPC_power_50_unobserved_linearZ_linearX_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=True,adjustment=False)
            row_power_small.extend([results['xgboost_power'], results['oracle_power']])
        for directory in [ "Result/HPC_power_50_unobserved_linearZ_linearX_adjustment_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=True,adjustment=True)
            row_power_small.extend([results['xgboost_power'], results['oracle_power']])
        Power_data_small.append(row_power_small)

    plot_results(Power_data_small, "Covariance adjustment, 1 model, Size 50") 

    Power_data = []
    Power_data_small = []
    
    """for coef in np.arange(0.0,1.26,0.18):
        row_power = [coef]
        for directory in [ "Result/HPC_power_1000_unobserved_linearZ_nonlinearX_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=False,adjustment=False)
            row_power.extend([ results['lightGBM_power'], results['oracle_power'],])
        for directory in ["Result/HPC_power_1000_unobserved_linearZ_nonlinearX_adjustment_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=False,adjustment=True)
            row_power.extend([ results['lightGBM_power'],results['oracle_power'] ])
        Power_data.append(row_power)

    plot_results(Power_data,  "Covariance adjustment,2 model, Size 1000") """

    for coef in np.arange(0.0,4.8,0.8):
        row_power_small = [coef]
        for directory in ["Result/HPC_power_50_unobserved_linearZ_nonlinearX_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=True,adjustment=False)
            row_power_small.extend([results['xgboost_power'], results['oracle_power']])
        for directory in [ "Result/HPC_power_50_unobserved_linearZ_nonlinearX_adjustment_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=True,adjustment=True)
            row_power_small.extend([results['xgboost_power'], results['oracle_power']])
        Power_data_small.append(row_power_small)
    plot_results(Power_data_small, "Covariance adjustment,2 model, Size 50") 
    Power_data = []
    Power_data_small = []

    
    """for coef in np.arange(0.0,0.36,0.06):
        row_power = [coef]
        for directory in [ "Result/HPC_power_1000_unobserved_nonlinearZ_nonlinearX_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=False,adjustment=False)
            row_power.extend([ results['lightGBM_power'], results['oracle_power'],])
        for directory in ["Result/HPC_power_1000_unobserved_nonlinearZ_nonlinearX_adjustment_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=False,adjustment=True)
            row_power.extend([ results['lightGBM_power'],results['oracle_power'] ])
        Power_data.append(row_power)

    plot_results(Power_data,  "Covariance adjustment, 3 model, Size 1000") """

    for coef in np.arange(0.0,1.5,0.25):
        row_power_small = [coef]
        for directory in ["Result/HPC_power_50_unobserved_nonlinearZ_nonlinearX_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=True,adjustment=False)
            row_power_small.extend([results['xgboost_power'], results['oracle_power']])
        for directory in [ "Result/HPC_power_50_unobserved_nonlinearZ_nonlinearX_adjustment_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=True,adjustment=True)
            row_power_small.extend([results['xgboost_power'], results['oracle_power']])
        Power_data_small.append(row_power_small)

    plot_results(Power_data_small, "Covariance adjustment,3 model, Size 50") 
    exit()
    for coef in np.arange(0.0,0.3 ,0.05):
        row_power = [coef]
        for directory in [ "Result/HPC_power_1000_unobserved_interference_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=False,adjustment=False)
            row_power.extend([ results['lightGBM_power'], results['oracle_power'],])
        for directory in ["Result/HPC_power_1000_unobserved_interference_adjusted_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=False,adjustment=True)
            row_power.extend([ results['lightGBM_power'],results['oracle_power'] ])
        Power_data.append(row_power)

    plot_results(Power_data,  "Covariance adjustment, Size 1000") 

    for coef in np.arange(0.0,1.2,0.2):
        row_power_small = [coef]
        for directory in ["Result/HPC_power_50_unobserved_interference_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=True,adjustment=False)
            row_power_small.extend([results['xgboost_power'], results['oracle_power']])
        for directory in [ "Result/HPC_power_50_unobserved_interference_adjusted_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=True,adjustment=True)
            row_power_small.extend([results['xgboost_power'], results['oracle_power']])
        Power_data_small.append(row_power_small)

    plot_results(Power_data_small, "Covariance adjustment, Size 50") 
     

main()


