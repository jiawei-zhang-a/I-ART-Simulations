import os
import numpy as np

def read_and_print_npz_files(directory):
    # Iterate over all files in the directory
    for file_name in os.listdir(directory):
        # Check if the file is a numpy npz file
        if file_name.endswith('.npy'):
            # Load the numpy npz file
            file_path = os.path.join(directory, file_name)
            data = np.load(file_path)

            # Print the file name and its content
            print(f"File: {file_name}")
            print("Content:")
            for key in data:
                print(f"{key}: {data[key]}")
            print("")

read_and_print_npz_files('HPC_result')
