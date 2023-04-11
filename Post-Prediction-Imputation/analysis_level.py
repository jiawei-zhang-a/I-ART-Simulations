import os
import numpy as np
import matplotlib.pyplot as plt

def read_and_print_npz_files(directory):
    
    summed_p_values_median = None
    summed_p_values_LR = None
    summed_p_values_xgboost = None

    p_values_median = []
    p_values_LR = []
    p_values_xgboost = []    

    N = int(len(os.listdir(directory)) / 3)
    # Loop through all the files in the directory
    for filename in os.listdir(directory):

        # Check if the file is a numpy file
        if filename.endswith(".npy"):
            # Load the numpy array from the file
            filepath = os.path.join(directory, filename)
            p_values = np.load(filepath)

            # Add the array to the running sum based on the filename
            if "p_values_median" in filename:
                if summed_p_values_median is None:
                    summed_p_values_median = p_values
                else:
                    summed_p_values_median += p_values
                p_values_median.append(list(p_values))
            elif "p_values_LR" in filename:
                if summed_p_values_LR is None:
                    summed_p_values_LR = p_values
                else:
                    summed_p_values_LR += p_values
                p_values_LR.append(list(p_values))
            elif "p_values_xgboost" in filename:
                if summed_p_values_xgboost is None:
                    summed_p_values_xgboost = p_values
                else:
                    summed_p_values_xgboost += p_values
                p_values_xgboost.append(list(p_values))

    # Print the summed arrays
    print("Summed p-values for Median Imputer:")
    print(summed_p_values_median/N)
    print("Summed p-values for LR Imputer:")
    print(summed_p_values_LR/N)
    print("Summed p-values for XGBoost Imputer:")
    print(summed_p_values_xgboost/N)
    
    # Plot the distribution of the first 6 p-values for each imputer
    print("Plotting the distribution of the first 6 p-values for each imputer")
    plot_p_values_distribution(p_values_median, "Median Imputer")
    plot_p_values_distribution(p_values_LR, "LR Imputer")
    plot_p_values_distribution(p_values_xgboost, "XGBoost Imputer")

def plot_p_values_distribution(p_values, imputer_name):
    p_values = np.array(p_values)
    fig, axs = plt.subplots(1, 6, figsize=(12, 4), tight_layout=True)
    for i in range(6):
        axs[i].hist(p_values[:,i])
        axs[i].set_title(f"p-value {i+1}")

    fig.suptitle(f"Distribution of first 6 p-values for {imputer_name}")
    plt.show()

read_and_print_npz_files('HPC_result')
