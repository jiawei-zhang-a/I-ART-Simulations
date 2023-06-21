import pandas as pd
import matplotlib.pyplot as plt

def plot_results(data):
    columns = ['beta', 'I50_Median', 'I50_Linear', 'I50_XGBoost', 'I50_Oracle', 
               'I2000_Median', 'I2000_Linear', 'I2000_XGBoost', 'I2000_Oracle', 
               'I20000_Median', 'I20000_Linear', 'I20000_XGBoost', 'I20000_Oracle']

    df = pd.DataFrame(data, columns=columns)

    plt.figure(figsize=(10, 6))

    colors = {'Median': 'blue', 'Linear': 'red', 'XGBoost': 'green', 'Oracle': 'purple'}
    linestyles = {'I50': '-', 'I2000': '--', 'I20000': ':'}

    for col in columns[1:]:
        method = col.split('_')[1]
        dataset = col.split('_')[0]
        linestyle = linestyles[dataset]
        plt.plot(df['beta'], df[col], marker='o', label=col, color=colors[method], linestyle=linestyle)
        if col == "I50_Oracle" or col == "I2000_Oracle" or col == "I20000_Oracle":
            plt.xlabel('Beta')
            plt.ylabel('Power(covariate adjusted)')
            plt.title('Performance of Imputation Methods for Varying Beta')
            plt.legend()
            plt.grid()

            plt.show()

