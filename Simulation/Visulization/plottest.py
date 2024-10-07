import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results( ):

    plt.figure(figsize=(10, 6))


    plt.xlabel(r'$\beta$',fontsize=30)
    plt.ylabel('Rejection Rate',fontsize=30)
    plt.grid()
    # Setting y-axis ticks with custom intervals
    y_ticks = [i/100.0 for i in range(25, 105, 25)]  # Starts from 0, ends at 1.05, with an interval of 0.05
    y_ticks.append(0.05)
    plt.yticks(y_ticks)
    plt.tick_params(axis='both', which='major', labelsize=25)

    #plt.show()
    if not os.path.exists("pic"):
        os.makedirs("pic")


    # Example: adding a triangle at (0, 0.5)
    plt.scatter(0, 0.5, color='black', marker='^', s=100, label="Custom Triangle")  # 's' is the size of the marker
    
    # Example: adding a square at (0, 0.3)
    plt.scatter(0, 0.3, color='black', marker='s', s=100, label="Custom Square")

    plt.show()

plot_results( )