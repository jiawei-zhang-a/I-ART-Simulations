import pandas as pd
import matplotlib.pyplot as plt

def plot_results(data, data_with_U):
    data =  [[row[0], row[1], row[2], row[3], row[4]] for row in data]
    data_with_U =  [[row[0], row[1], row[2], row[3], row[4]] for row in data_with_U]
    columns = ['beta', 'Median', 'Linear', 'XGBoost', 'Oracle']

    df = pd.DataFrame(data, columns=columns)
    df_with_U = pd.DataFrame(data_with_U, columns=columns)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    colors = {'Median': 'blue', 'Linear': 'red', 'XGBoost': 'green', 'Oracle': 'purple'}

    # Plot data without U
    for col in columns[1:]:
        ax1.plot(df['beta'], df[col], marker='o', label=col, color=colors[col])

    ax1.set_xlabel('Beta')
    ax1.set_ylabel('Power(covariate adjusted)')
    ax1.set_title('Performance of Imputation Methods for Varying Beta (I=50, No U)')
    ax1.legend()
    ax1.grid()

    # Plot data with U
    for col in columns[1:]:
        ax2.plot(df_with_U['beta'], df_with_U[col], marker='o', label=col, color=colors[col])

    ax2.set_xlabel('Beta')
    ax2.set_ylabel('Power(covariate adjusted)')
    ax2.set_title('Performance of Imputation Methods for Varying Beta (I=50, With U)')
    ax2.legend()
    ax2.grid()

    plt.show()
