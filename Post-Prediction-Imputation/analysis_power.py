import numpy as np
import os

def read_and_print_npz_files(directory):
    
    print("Analysis of : " + directory)

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

    # Print the summed arrays
    print("Mean p-values for Median Imputer:")
    print(summed_p_values_median/N)
    print("Mean p-values for LR Imputer:")
    print(summed_p_values_LR/N)
    print("Mean p-values for XGBoost Imputer:")
    print(summed_p_values_xgboost/N)


def main():
    for i in [4, 6, 8, 10, 12, 14]:
        print("beta: ", i)
        # Sum up correlations arrays

        read_and_print_npz_files('HPC_beta/' + str(i))
        read_and_print_npz_files('HPC_beta_unobserved/' + str(i))
        read_and_print_npz_files('HPC_beta_2000/' + str(i))
        read_and_print_npz_files('HPC_beta_unobserved_2000/' + str(i))

main()