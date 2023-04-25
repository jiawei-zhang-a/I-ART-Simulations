import pandas as pd
import matplotlib.pyplot as plt

def plot_results(data, data_with_U):
    columns = ['beta', 'I50_Median', 'I50_Linear', 'I50_XGBoost', 'I50_Oracle', 'I100_Median', 'I100_Linear', 'I100_XGBoost', 'I100_Oracle']
    df = pd.DataFrame(data, columns=columns)
    df_with_U = pd.DataFrame(data_with_U, columns=columns)

    plt.figure(figsize=(12, 8))

    colors = {'Median': 'blue', 'Linear': 'red', 'XGBoost': 'green', 'Oracle': 'purple'}
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
