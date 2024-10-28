import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_results(data, title, xsticks, lr, gbm, multipleimputation):
    columns = ['beta', 'Imputer_Median', 'Imputer_PREP-RidgeReg',  'Imputer_PREP-GBM', 'Imputer_Oracle']

    df = pd.DataFrame(data, columns=columns)

    plt.figure(figsize=(10, 6))

    colors = {'Median': 'blue', 'PREP-RidgeReg': 'red', 'PREP-GBM': 'green', 'Oracle':'purple'}
    linestyles = {'Imputer': '-'}

    for col in columns[1:]:
        method = col.split('_')[1]
        dataset = col.split('_')[0]
        linestyle = linestyles[dataset]
        plt.plot(df['beta'], df[col], marker='o', label=method, color=colors[method], linestyle=linestyle, linewidth=2.0)
        
    plt.xlabel(r'$\beta$', fontsize=30)
    plt.ylabel('Rejection Rate', fontsize=30)
    plt.grid()

    plt.scatter(0, lr, color='red', marker='x', s=50, label="Linear Regression")
    plt.scatter(0, gbm, color='green', marker='x', s=50,  label="Gradient Boosting Machine")
    #plt.scatter(0, multipleimputation, color='purple', marker='x', s=50, label="Multiple Imputation")

    
    y_ticks = [i / 100.0 for i in range(25, 105, 25)]
    y_ticks.append(0.05)
    plt.yticks(y_ticks)
    plt.xticks(xsticks)
    plt.tick_params(axis='both', which='major', labelsize=25)

    if not os.path.exists("pic"):
        os.makedirs("pic")

    plt.savefig("pic/" + title + ".pdf", bbox_inches='tight')

def load_and_plot(main_filename, small_filename, title_main, title_small, xsticks_main, xsticks_small, lr50, gbm50, lr1000, gbm1000, multipleimputation1000, multipleimputation50):
    # Load the main and small data from the .pkl files in the tmp folder
    main_data = pd.read_pickle(os.path.join("tmp", main_filename))
    small_data = pd.read_pickle(os.path.join("tmp", small_filename))
    
    # Plot the main and small data
    plot_results(main_data, title_main, xsticks_main, lr1000, gbm1000, multipleimputation1000)
    plot_results(small_data, title_small, xsticks_small, lr50, gbm50, multipleimputation50)

def main_pic_plotter():
    load_and_plot("Size1000_Model1.pkl", "Size50_Model1.pkl", "Size1000_Model1", "Size50_Model1", 
                  np.arange(0.0, 0.42, 0.07), np.arange(0, 1.5, 0.25), 0.1115,0.0985, 0.181,0.189, 0.122,0.091 )
    load_and_plot("Size1000_Model2.pkl", "Size50_Model2.pkl", "Size1000_Model2", "Size50_Model2", 
                  np.arange(0.0, 0.96, 0.16), np.arange(0.0, 4.8, 0.8), 0.084, 0.0905, 0.0755, 0.14, 0.075, 0.0675)
    load_and_plot("Size1000_Model3.pkl", "Size50_Model3.pkl", "Size1000_Model3", "Size50_Model3", 
                  np.arange(0.0, 0.36, 0.06), np.arange(0.0, 1.5, 0.25), 0.0865, 0.09, 0.0895, 0.14, 0.0715, 0.069)
    load_and_plot("Size1000_Model4.pkl", "Size50_Model4.pkl", "Size1000_Model4", "Size50_Model4", 
                  np.arange(0.0, 0.36, 0.06), np.arange(0.0, 1.5, 0.25), 0.0765, 0.0905, 0.0845, 0.127, 0.0605, 0.0625)

main_pic_plotter()
