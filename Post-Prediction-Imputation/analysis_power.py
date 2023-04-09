import numpy as np
import os
import glob

folder_path = 'HPC_Power'

def sum_npy_files(directory, prefix):
    # Initialize a variable to store the sum of the arrays
    summed_arrays = None

    # Loop through all the files in the directory
    for filename in os.listdir(directory):
        # Check if the file matches the prefix and is a npy file
        if filename.startswith(prefix) and filename.endswith(".npy"):
            # Load the numpy array from the file
            filepath = os.path.join(directory, filename)
            array = np.load(filepath)

            # Add the array to the running sum
            if summed_arrays is None:
                summed_arrays = array
            else:
                summed_arrays += array

    # Return the summed arrays
    return summed_arrays

def main():
    for i in [1, 2, 4, 5, 8, 10, 12, 14, 16, 18, 20, 32]:
        print("beta: ", i)
        # Sum up correlations arrays
        summed_correlations = sum_npy_files(folder_path + '/%d'%(i), "correlations_")

        # Sum up powers arrays
        summed_powers = sum_npy_files(folder_path + '/%d'%(i), "powers_")

        print("correlations: ", summed_correlations)
        print("powers: ", summed_powers)

main()