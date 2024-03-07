import os
import re

def parse_median_value(line):
    match = re.search(r'\(True, \[(\d+\.\d+)\]\)', line)
    if match:
        return float(match.group(1))
    else:
        raise ValueError("Could not parse median value")

# Set the directory where your files are located
directory = 'Result'  # replace with your actual directory path

largest_difference = 0
file_with_largest_difference = None

# Iterate over the files in the directory
for filename in os.listdir(directory):
    if filename.startswith("p_values") and filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as file:
            contents = file.read()
            # Assuming the format of the content is consistent with the example given
            ridge_median = parse_median_value(re.search(r'RidgeRegression with covariate adjustment: (.*)', contents).group(1))
            lightgbm_median = parse_median_value(re.search(r'LightGBM with covariate adjustment: (.*)', contents).group(1))
            
            # Calculate the absolute difference
            difference = abs(ridge_median - lightgbm_median)
            
            # Check if this is the largest difference so far
            if difference > largest_difference:
                largest_difference = difference
                file_with_largest_difference = filename

# Print the results
print(f"The file with the largest difference is: {file_with_largest_difference}")
print(f"The largest difference is: {largest_difference}")
