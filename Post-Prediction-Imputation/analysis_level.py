import os
import numpy as np

def read_and_print_npz_files(directory):
    summed_levels = None

    # Loop through all the files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a numpy file
        if filename.endswith(".npy"):
            # Load the numpy array from the file
            filepath = os.path.join(directory, filename)
            levels = np.load(filepath)
            # Add the array to the running sum
            if summed_levels is None:
                summed_levels = levels
            else:
                summed_levels += levels

    # Print the summed array
    print(summed_levels)

read_and_print_npz_files('HPC_result')
