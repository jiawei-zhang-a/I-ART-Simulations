import pandas as pd
import matplotlib.pyplot as plt

def plot_results(data, data_LR_adjusted, data_xgboost_adjusted):
    data =  [[row[0], row[1], row[2], row[3], row[4]] for row in data]
    data_LR_adjusted =  [[row[0], row[1], row[2], row[3], row[4]] for row in data_LR_adjusted]
    data_xgboost_adjusted =  [[row[0], row[1], row[2], row[3], row[4]] for row in data_xgboost_adjusted]
    
    columns = ['beta', 'Median', 'Linear', 'XGBoost', 'Oracle', 'Median_LR', 'Linear_LR', 'XGBoost_LR', 'I100_Oracle_LR', 'Median_XGBoost', 'Linear_XGBoost', 'XGBoost_XGBoost', 'I100_Oracle_XGBoost']

    df = pd.DataFrame(data, columns=columns)
    df_LR_adjusted = pd.DataFrame(data_LR_adjusted, columns=columns)
    df_xgboost_adjusted = pd.DataFrame(data_xgboost_adjusted, columns=columns)

    plt.figure(figsize=(12, 8))

    colors = {'Median': 'blue', 'Linear': 'red', 'XGBoost': 'green', 'Oracle': 'purple'}
    linestyles = {'Original': '-', 'LR_adjusted': '--', 'XGBoost_adjusted': '-.'}

    for col in columns[1:]:
        method = col.split('_')[1]
        dataset = col.split('_')[0]
        linestyle = linestyles[dataset]
        plt.plot(df['beta'], df[col], marker='o', label=col, color=colors[method], linestyle=linestyle)
        plt.plot(df_LR_adjusted['beta'], df_LR_adjusted[col], marker='o', label=col + " with LR", color=colors[method], linestyle=linestyle, alpha=0.5)
        plt.plot(df_xgboost_adjusted['beta'], df_xgboost_adjusted[col], marker='o', label=col + " with XGBoost", color=colors[method], linestyle=linestyle, alpha=0.5)

    plt.xlabel('Beta')
    plt.ylabel('Absolute Error')
    plt.title('Performance of Imputation Methods for Varying Beta')
    plt.legend()
    plt.grid()

    plt.show()
