import numpy as np
import os
import glob

folder_path = 'HPC_result'

# Find all correlation and power files
correlation_files = glob.glob(os.path.join(folder_path, 'correlations_*.npy'))
power_files = glob.glob(os.path.join(folder_path, 'powers_*.npy'))

# Initialize arrays for accumulated correlations and powers
accumulated_correlations = None
accumulated_powers = None

# Iterate over correlation files, load the contents, and add them to accumulated_correlations
for file_path in correlation_files:
    loaded_array = np.load(file_path)
    
    if accumulated_correlations is None:
        accumulated_correlations = loaded_array
    else:
        accumulated_correlations += loaded_array

# Iterate over power files, load the contents, and add them to accumulated_powers
for file_path in power_files:
    loaded_array = np.load(file_path)
    
    if accumulated_powers is None:
        accumulated_powers = loaded_array
    else:
        accumulated_powers += loaded_array

print("Accumulated Correlations:\n", accumulated_correlations)
print("Accumulated Powers:\n", accumulated_powers)
