import numpy as np
import os

def read_and_print_npz_files(directory, file):
    
    file.write("Analysis of : " + directory + "\n")

    summed_p_values_median = None
    summed_p_values_LR = None
    summed_p_values_xgboost = None

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
            elif "p_values_LR" in filename:
                if summed_p_values_LR is None:
                    summed_p_values_LR = p_values
                else:
                    summed_p_values_LR += p_values
            elif "p_values_xgboost" in filename:
                if summed_p_values_xgboost is None:
                    summed_p_values_xgboost = p_values
                else:
                    summed_p_values_xgboost += p_values

    file.write("Mean p-values for Median Imputer:\n")
    file.write(str(summed_p_values_median/N) + "\n")
    file.write("Mean p-values for LR Imputer:\n")
    file.write(str(summed_p_values_LR/N) + "\n")
    file.write("Mean p-values for XGBoost Imputer:\n")
    file.write(str(summed_p_values_xgboost/N) + "\n")

def main():
    for i in [4, 6, 8, 10, 12, 14]:
        with open("power.result", "a") as file:
            file.write("beta: " + str(i) + "\n")
            read_and_print_npz_files('HPC_Power/' + str(i), file)
            read_and_print_npz_files('HPC_Power_unobserved/' + str(i), file)
            read_and_print_npz_files('HPC_Power_2000/' + str(i), file)
            read_and_print_npz_files('HPC_Power_unobserved_2000/' + str(i), file)

main()