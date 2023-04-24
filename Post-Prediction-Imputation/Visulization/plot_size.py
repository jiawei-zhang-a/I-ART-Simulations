import pandas as pd
import matplotlib.pyplot as plt
"""
data = [
    [4, 0.1385, 0.141, 0.2405, 0.197, 0.1915, 0.3455],
    [6, 0.199, 0.287, 0.4295, 0.2775, 0.4185, 0.6195],
    [8, 0.2385, 0.441, 0.602, 0.3795, 0.705, 0.836],
    [10, 0.316, 0.632, 0.7115, 0.5165, 0.8855, 0.926],
    [12, 0.3925, 0.7735, 0.761, 0.6345, 0.96, 0.8905],
    [14, 0.5125, 0.8875, 0.8, 0.752, 0.989, 0.8775],
    [16, 0.582, 0.9455, 0.84, 0.8345, 0.998,  0.911]
]
"""

def plot_results(data):
    columns = ['beta', 'I50_Median', 'I50_Linear', 'I50_XGBoost', 'I100_Median', 'I100_Linear', 'I100_XGBoost']

    df = pd.DataFrame(data, columns=columns)

    plt.figure(figsize=(10, 6))

    colors = {'Median': 'blue', 'Linear': 'red', 'XGBoost': 'green'}
    linestyles = {'I50': '-', 'I100': '--'}

    for col in columns[1:]:
        method = col.split('_')[1]
        dataset = col.split('_')[0]
        linestyle = linestyles[dataset]
        plt.plot(df['beta'], df[col], marker='o', label=col, color=colors[method], linestyle=linestyle)

    plt.xlabel('Beta')
    plt.ylabel('Absolute Error')
    plt.title('Performance of Imputation Methods for Varying Beta')
    plt.legend()
    plt.grid()

    plt.show()

