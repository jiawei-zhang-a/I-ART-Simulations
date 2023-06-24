import pandas as pd
import matplotlib.pyplot as plt

def plot_results(data):
    columns = ['beta', 'Size50_Median', 'Size50_Linear', 'Size50_XGBoost', 
               'Size2000_Median', 'Size2000_Linear', 'Size2000_XGBoost', 
               'Size20000_Median', 'Size20000_Linear', 'Size20000_XGBoost']

    df = pd.DataFrame(data, columns=columns)

    plt.figure(figsize=(10, 6))

    colors = {'Median': 'blue', 'Linear': 'red', 'XGBoost': 'green', 'Oracle': 'purple'}
    linestyles = {'Size50': '-', 'Size2000': '--', 'Size20000': ':'}

    for col in columns[1:]:
        method = col.split('_')[1]
        dataset = col.split('_')[0]
        linestyle = linestyles[dataset]
        plt.plot(df['beta'], df[col], marker='o', label=col, color=colors[method], linestyle=linestyle)
        
        if col == "Size50_XGBoost":
            plt.xlabel('Beta')
            plt.ylabel('Power')
            plt.title('Small Size(50)')
            plt.legend()
            plt.grid()
            plt.show()

        if col == "Size2000_XGBoost":
            plt.xlabel('Beta')
            plt.ylabel('Power')
            plt.title('Medium Size(2000)')
            plt.legend()
            plt.grid()
            plt.show()

        if col == "Size20000_XGBoost":
            plt.xlabel('Beta')
            plt.ylabel('Power')
            plt.title('Large Size(20000)')
            plt.legend()
            plt.grid()
            plt.show()
        

