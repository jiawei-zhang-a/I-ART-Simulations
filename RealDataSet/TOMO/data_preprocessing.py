import pandas as pd
import iArt
import numpy as np


# Define the path to the TSV file
file_path = 'DS0001/36158-0001-Data.tsv'

# Load the TSV file into a DataFrame
df = pd.read_csv(file_path, sep='\t')

# Filter the DataFrame for rows where 'WAVE' equals 2
filtered_df = df[df['WAVE'] == 2]

Y = filtered_df['SCWM_CWH']

S = filtered_df['STUDYGROUP']

# Condition = 1, intervention group; Condition = 2, control group, make 2 to be 0
filtered_df['CONDITION'] = filtered_df['CONDITION'].replace(2, 0)

Z = filtered_df['CONDITION']

result = iArt.test(Z=Z, X=X, Y=Y,S=S, L=1000, verbose=True, alternative = 'two-sided')
print(result)