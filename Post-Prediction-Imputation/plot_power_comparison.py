import numpy as np
import pandas as pd
from analysis_power import read_npz_files
import matplotlib.pyplot as plt

def plot_results_50(data):
    columns = ['beta', 'Size50_Median', 'Size50_Linear', 'Size50_XGBoost']

    df = pd.DataFrame(data, columns=columns)

    plt.figure(figsize=(10, 6))

    colors = {'Median': 'blue', 'Linear': 'red', 'XGBoost': 'green'}
    linestyles = {'Size50': '-'}

    for col in columns[1:]:
        method = col.split('_')[1]
        dataset = col.split('_')[0]
        linestyle = linestyles[dataset]
        plt.plot(df['beta'], df[col], marker='o', label=col, color=colors[method], linestyle=linestyle)
        
    plt.xlabel('Beta')
    plt.ylabel('Power')
    plt.title('Small Size(50)')
    plt.legend()
    plt.grid()
    plt.show()

def plot_results_2000(data):
    columns = ['beta', 'Size2000_Median', 'Size2000_Linear', 'Size2000_XGBoost']

    df = pd.DataFrame(data, columns=columns)

    plt.figure(figsize=(10, 6))

    colors = {'Median': 'blue', 'Linear': 'red', 'XGBoost': 'green'}
    linestyles = {'Size2000': '-'}

    for col in columns[1:]:
        method = col.split('_')[1]
        dataset = col.split('_')[0]
        linestyle = linestyles[dataset]
        plt.plot(df['beta'], df[col], marker='o', label=col, color=colors[method], linestyle=linestyle)

    plt.xlabel('Beta')
    plt.ylabel('Power')
    plt.title('Medium Size(2000)')
    plt.legend()
    plt.grid()
    plt.show()


def plot_results_20000(data):
    columns = ['beta', 'Size20000_Median', 'Size20000_Linear', 'Size20000_XGBoost']

    df = pd.DataFrame(data, columns=columns)

    plt.figure(figsize=(10, 6))

    colors = {'Median': 'blue', 'Linear': 'red', 'XGBoost': 'green'}
    linestyles = {'Size20000': '-'}

    for col in columns[1:]:
        method = col.split('_')[1]
        dataset = col.split('_')[0]
        linestyle = linestyles[dataset]
        plt.plot(df['beta'], df[col], marker='o', label=col, color=colors[method], linestyle=linestyle)
        
    plt.xlabel('Beta')
    plt.ylabel('Power')
    plt.title('Large Size(20000)')
    plt.legend()
    plt.grid()
    plt.show()
        
def main(type):
    Power_data_50 = []
    Power_data_2000 = []
    Power_data_20000= []

    for coef in np.arange(0.0,2,0.05):
        row_power = [coef]
        for directory in ["Result/HPC_power_50_single/%f" % (coef)]:
            results = read_npz_files(directory)
            row_power.extend([results['median_power'], results['lr_power'], results['xgboost_power']])
        Power_data_50.append(row_power)
    plot_results_50(Power_data_50)

    for coef in np.arange(0.0,0.4,0.02):
        row_power = [coef]
        for directory in [ "Result/HPC_power_2000_single/%f" % (coef)]:
            results = read_npz_files(directory)
            row_power.extend([results['median_power'], results['lr_power'], results['xgboost_power']])

        Power_data_2000.append(row_power) 
    plot_results_2000(Power_data_2000)

main("single")
