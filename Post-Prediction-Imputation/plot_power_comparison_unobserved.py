import pandas as pd
import matplotlib.pyplot as plt

data = [
    [4, 0.1385, 0.141, 0.2405],
    [6, 0.199, 0.287, 0.4295],
    [8, 0.2385, 0.441, 0.602],
    [10, 0.316, 0.632, 0.7115],
    [12, 0.3925, 0.7735, 0.761],
    [14, 0.5125, 0.8875, 0.8]
]

data_with_U = [
    [4, 0.19, 0.08, 0.3415],
    [6, 0.286, 0.14, 0.5915],
    [8, 0.3975, 0.2865, 0.749],
    [10, 0.4965, 0.5015, 0.832],
    [12, 0.6135, 0.7045, 0.89],
    [14, 0.6975, 0.8535, 0.9105]
]

columns = ['beta', 'Median', 'Linear', 'XGBoost']

df = pd.DataFrame(data, columns=columns)
df_with_U = pd.DataFrame(data_with_U, columns=columns)

plt.figure(figsize=(10, 6))

colors = {'Median': 'blue', 'Linear': 'red', 'XGBoost': 'green'}
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
