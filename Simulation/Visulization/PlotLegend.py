import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Create a new figure for the legend
fig_leg, ax_leg = plt.subplots(figsize=(6, 1))  # Adjust the size as needed
ax_leg.axis('off')

# Create the legend
custom_lines_types = [
    Line2D([0], [0], color='red', lw=2, linestyle='-'),
    Line2D([0], [0], color='red', lw=2, linestyle='--'),
    Line2D([0], [0], color='green', lw=2, linestyle='-'),
    Line2D([0], [0], color='green', lw=2, linestyle='--'),
    Line2D([0], [0], color='blue', lw=2, linestyle='-'),
    Line2D([0], [0], color='blue', lw=2, linestyle='--')
]

# Create a new figure for the custom_lines_types legend
fig_leg1, ax_leg1 = plt.subplots(figsize=(12, 1))  # Adjust the size as needed
ax_leg1.axis('off')

# Create the legend for custom_lines_types
legend1 = ax_leg1.legend(custom_lines_types, 
                          ['Method 1 (Algo 1 - Linear)', 'Method 2 (Algo 2 - Linear)', 'Method 3 (Algo 1 - Boosting)', 'Method 4 (Algo 2 - Boosting)', 'Method 5 (Median Imputation)', 'Method 6 (Median Imputation with Covariate Adjustment)'], 
                          loc='center', 
                          ncol=3,  # Set to 6 to make all items appear in a single line
                          fontsize='large')  # Adjust as needed

# Save the legend
fig_leg1.savefig('pic/legendcustomlinestypesCovariateAdjustment.pdf', format='pdf', bbox_inches='tight')# Create a figure just for the legend
fig, ax = plt.subplots(figsize=(4, 0.5))  # Smaller figure size for more compact legend

# Adding fake scatter points just to get the legend, and keeping them in a single row
scatter1 = plt.scatter([], [], color='red', marker='x', s=50, label="Model Based Imputation - Linear")
scatter2 = plt.scatter([], [], color='green', marker='x', s=50, label="Model Based Imputation - Boosting")

# Adding legend with smaller font size
legend = ax.legend(loc='center', frameon=True, fontsize=8, ncol=2)  # ncol=2 makes it a single row with two columns

# Remove the axes
ax.set_axis_off()

# Adjust layout to make the legend compact and save to the specified path
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Save the legend to a PDF
pdf_filename = "pic/legend_typeone_error.pdf"
fig.savefig(pdf_filename, format='pdf', bbox_inches='tight')

# Close the plot
plt.close()