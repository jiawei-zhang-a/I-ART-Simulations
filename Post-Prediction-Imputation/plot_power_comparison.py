import numpy as np
import pandas as pd
from analysis_power import read_npz_files
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_results(data, title, xsticks):
    plt.clf()

    columns = ['beta', 'Imputer_PREP-GBM',
               "Imputer_PREP-RidgeReg", "Imputer_GBM-adjusted",  "Imputer_LR-adjusted"]

    df = pd.DataFrame(data, columns=columns)

    plt.figure(figsize=(10, 6))

    colors = {
        'Imputer_PREP-GBM': 'green', 
        "Imputer_PREP-RidgeReg": "red",
        "Imputer_GBM-adjusted": 'green', 
        "Imputer_LR-adjusted": 'red'
    }

    linestyles = {
        'Imputer_PREP-GBM': '-', 
        "Imputer_PREP-RidgeReg": '-', 
        "Imputer_GBM-adjusted": '--', 
        "Imputer_LR-adjusted": '--'
    }

    for col in columns[1:]:
        plt.plot(df['beta'], df[col], marker='o', color=colors[col], linestyle=linestyles[col], linewidth=2)

    plt.xlabel(r'$\beta$',fontsize=30)
    plt.ylabel('Power',fontsize=30)
    plt.grid()

    if not os.path.exists("pic"):
        os.makedirs("pic")
    plt.xticks(xsticks)
    y_ticks = [i / 100.0 for i in range(25, 105, 25)]
    y_ticks.append(0.05)
    plt.yticks(y_ticks)
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.savefig("pic/" + title + ".pdf", format='pdf', bbox_inches='tight')


def plot(range,range_small, path,path_small, title, title_small):
    Power_data = []
    Power_data_small = []

    for coef in range:
        row_power = [coef]
        for directory in [path + "/%f" % (coef)]:
            results = read_npz_files(directory,small_size=False)
            row_power.extend([ results['lightGBM_power'], results['lr_power']])
        for directory in [path + "_adjusted_LightGBM/%f" % (coef)]:
            results = read_npz_files(directory,small_size=False, type='adjusted')
            row_power.extend([ results['lightGBM_power'] ])
        for directory in [path + "_adjusted_LR/%f" % (coef)]:
            results = read_npz_files(directory,small_size=False, type='adjusted')
            row_power.extend([ results['lr_power'] ])
        Power_data.append(row_power)
    print(Power_data)
    plot_results(Power_data, title, range)

    for coef in range_small:
        row_power_small = [coef]
        for directory in [path_small + "/%f" % (coef)]:
            results = read_npz_files(directory,small_size=True)
            row_power_small.extend([results['xgboost_power'], results['lr_power']])
        for directory in [path_small + "_adjusted_Xgboost/%f" % (coef)]:
            results = read_npz_files(directory,small_size=True, type='adjusted')
            row_power_small.extend([results['xgboost_power']])
        for directory in [path_small + "_adjusted_LR/%f" % (coef)]:
            results = read_npz_files(directory,small_size=True, type='adjusted')
            row_power_small.extend([results['lr_power']])
        Power_data_small.append(row_power_small)
    print(Power_data_small)
    plot_results(Power_data_small, title_small, range_small)



def main():
    #plot(np.arange(0.0,0.42,0.07), np.arange(0,1.5,0.25), "Result/HPC_power_1000_unobserved_linearZ_linearX", "Result/HPC_power_50_unobserved_linearZ_linearX", "Size-1000, Single: Covariance Adjusted- linearX-linearZ, ", "Size-50, Single: Covariance Adjusted-linearX-linearZ ")
    #plot(np.arange(0.0,0.96,0.16), np.arange(0,4.8,0.8), "Result/HPC_power_1000_unobserved_linearZ_nonlinearX", "Result/HPC_power_50_unobserved_linearZ_nonlinearX", "Size-1000, Single: Covariance Adjusted- linearZ-nonlinearX, ", "Size-50, Single: Covariance Adjusted-linearZ-nonlinearX ")
    #plot(np.arange(0.0,0.42,0.07), np.arange(0,1.5,0.25), "Result/HPC_power_1000_model1", "Result/HPC_power_50_model1", "Size1000_Model1_CovariateAdjusted", "Size50_Model1_CovariateAdjusted")
    #plot(np.arange(0.0,0.96,0.16), np.arange(0.0,4.8,0.8), "Result/HPC_power_1000_model2", "Result/HPC_power_50_model2", "Size1000_Model2_CovariateAdjusted", "Size50_Model2_CovariateAdjusted")
    plot(np.arange(0.0,0.36,0.06), np.arange(0.0,1.5,0.25), "Result/HPC_power_1000_model3", "Result/HPC_power_50_model3", "Size1000_Model3_CovariateAdjusted", "Size50_Model3_CovariateAdjusted")
    plot(np.arange(0.0,0.36,0.06), np.arange(0.0,1.5,0.25), "Result/HPC_power_1000_model4", "Result/HPC_power_50_model4", "Size1000_Model4_CovariateAdjusted", "Size50_Model4_CovariateAdjusted")
    plot(np.arange(0.0,0.6 ,0.1), np.arange(0.0,18,3), "Result/HPC_power_1000_model6", "Result/HPC_power_50_model6", "Size1000_Model6_CovariateAdjusted", "Size50_Model6_CovariateAdjusted")
    """
    Power_data = []
    Power_data_small = []

    for coef in np.arange(0.0,0.42,0.07):
        row_power = [coef]
        for directory in [ "Result/HPC_power_1000_unobserved_interference_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=False)
            row_power.extend([ results['lightGBM_power'], results['lr_power']])
        for directory in ["Result/HPC_power_1000_unobserved_interference_adjusted_3_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=False, type='adjusted')
            row_power.extend([ results['lightGBM_power'] ])
        for directory in ["Result/HPC_power_1000_unobserved_interference_adjusted_1_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=False, type='adjusted')
            row_power.extend([ results['lr_power'] ])
        Power_data.append(row_power)
    print(Power_data)
    plot_results(Power_data,  "Size-1000, Single: Covariance Adjusted, ", np.arange(0.0,0.42,0.07)) 

    for coef in np.arange(0,1.5,0.25):
        row_power_small = [coef]
        for directory in ["Result/HPC_power_50_unobserved_interference_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=True)
            row_power_small.extend([results['xgboost_power'], results['lr_power']])
        for directory in ["Result/HPC_power_50_unobserved_interference_adjusted_2_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=True, type='adjusted')
            row_power_small.extend([results['xgboost_power']])
        for directory in [ "Result/HPC_power_50_unobserved_interference_adjusted_1_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=True, type='adjusted')
            row_power_small.extend([results['lr_power']])
        Power_data_small.append(row_power_small)
    print(Power_data)
    plot_results(Power_data_small, "Size-50, Single: Covariance Adjusted, ", np.arange(0,1.5,0.25)) 
    """
    #plot(np.arange(0.0,0.42,0.07), np.arange(0,1.5,0.25), "Result/HPC_power_1000_unobserved_interference", "Result/HPC_power_50_unobserved_interference", "Size-1000, Single: Covariance Adjusted, ", "Size-50, Single: Covariance Adjusted, ")
    """
    for coef in np.arange(0.0,0.6 ,0.1):
        row_power = [coef]
        for directory in [ "Result/HPC_power_1000_unobserved_interference_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=False)
            row_power.extend([ results['lightGBM_power'], results['lr_power']])
        for directory in ["Result/HPC_power_1000_unobserved_interference_adjusted_3_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=False, type='adjusted')
            row_power.extend([ results['lightGBM_power'] ])
        for directory in ["Result/HPC_power_1000_unobserved_interference_adjusted_1_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=False, type='adjusted')
            row_power.extend([ results['lr_power'] ])
        Power_data.append(row_power)
    print(Power_data)
    plot_results(Power_data,  "Size-1000, Single: Covariance Adjusted, ", np.arange(0.0,0.6 ,0.1)) 

    for coef in np.arange(0.0,18,3):
        row_power_small = [coef]
        for directory in ["Result/HPC_power_50_unobserved_interference_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=True)
            row_power_small.extend([results['xgboost_power'], results['lr_power']])
        for directory in ["Result/HPC_power_50_unobserved_interference_adjusted_2_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=True, type='adjusted')
            row_power_small.extend([results['xgboost_power']])
        for directory in [ "Result/HPC_power_50_unobserved_interference_adjusted_1_single/%f" % (coef)]:
            results = read_npz_files(directory,small_size=True, type='adjusted')
            row_power_small.extend([results['lr_power']])
        Power_data_small.append(row_power_small)
    print(Power_data)
    plot_results(Power_data_small, "Size-50, Single: Covariance Adjusted, ", np.arange(0.0,18,3))   
    """
main()
