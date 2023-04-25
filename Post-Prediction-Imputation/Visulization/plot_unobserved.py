import pandas as pd
import matplotlib.pyplot as plt

def plot_results(data, data_with_U):
    columns = ['beta', 'Median', 'Linear', 'XGBoost', 'Oracle']
    data =  [[row[0], row[1], row[2], row[3], row[4]] for row in data]
    data_with_U =  [[row[0], row[1], row[2], row[3], row[4]] for row in data_with_U]

    df = pd.DataFrame(data, columns=columns)
    df_with_U = pd.DataFrame(data_with_U, columns=columns)

    plt.figure(figsize=(10, 6))

    colors = {'Median': 'blue', 'Linear': 'red', 'XGBoost': 'green', 'Oracle': 'purple'}
    linestyles = {'No_U': '-', 'With_U': '--'}

    for i, col in enumerate(columns[1:]):
        plt.plot(df['beta'], df[col], marker='o', label=f'{col} (No U)', color=colors[col], linestyle=linestyles['No_U'])
        plt.plot(df_with_U['beta'], df_with_U[col], marker='o', label=f'{col} (With U)', color=colors[col], linestyle=linestyles['With_U'])

    plt.xlabel('Beta')
    plt.ylabel('Absolute Error')
    plt.title('Performance of Imputation Methods for Varying Beta (I=50)')
    plt.legend()
    plt.grid()

    plt.show()
