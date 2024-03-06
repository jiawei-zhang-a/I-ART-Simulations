import iArt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer


# Define the path to the TSV file
file_path = 'DS0001/36158-0001-Data.tsv'

# Load the TSV file into a DataFrame
df = pd.read_csv(file_path, sep='\t')

# Assuming 'outcome_df' is your DataFrame and 'SCWM_CWH' is the column of interest
# Compute the correlation matrix
correlation_matrix = df.corr()

# Extract correlation with 'SCWM_CWH' and remove its self-correlation and 'WAVE' if it exists
correlation_with_SCWM_CWH = correlation_matrix['SCWM_CWH'].drop(['SCWM_CWH', 'WAVE', 'WM_CWH8R','WM_CWH7R','WM_CWH2R','WM_HPQ2','WM_CWH6R','WM_CWH4R'] if 'WAVE' in correlation_matrix.columns else 'SCWM_CWH')

# Find the most highly correlated term
most_highly_correlated_term = correlation_with_SCWM_CWH.idxmax()

# Print the highest term and its correlation number
print(f"The most highly correlated term with 'SCWM_CWH' is: ", most_highly_correlated_term)


# Visualization
plt.figure(figsize=(10, 6))
correlation_with_SCWM_CWH.plot(kind='bar')
plt.title(f"Correlation of 'SCWM_CWH' with Other Terms")
plt.ylabel('Correlation Coefficient')
plt.xlabel('Terms')
plt.axhline(0, color='grey', lw=0.5)
plt.xticks(rotation=45)
plt.tight_layout()

# Highlight the most highly correlated term
plt.bar(most_highly_correlated_term, correlation_with_SCWM_CWH[most_highly_correlated_term], color='red')

plt.show()