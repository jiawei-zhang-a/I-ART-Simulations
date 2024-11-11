import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Define the line styles for each method
legend_lines = [
    Line2D([0], [0], color='red', lw=2, linestyle='-'),
    Line2D([0], [0], color='green', lw=2, linestyle='-'),
    Line2D([0], [0], color='black', lw=2, linestyle='-')
]

# Define labels for each method
legend_labels = [
    'Method 1 (Algo 1 - Linear)',
    'Method 2 (Algo 1 - Boosting)',
    r'Method 3 ($T_{M}(\mathbf{Z}, \mathbf{M})$)'
]

# Create a figure specifically for the legend
fig, ax = plt.subplots(figsize=(8, 1))  # Adjust size for spacing as needed
ax.axis('off')  # Hide axes for a cleaner look

# Create the legend
ax.legend(legend_lines, legend_labels, loc='center', fontsize='large', ncol=3)

# Save the legend to a PDF file
output_filename = "pic/legend_T_M.pdf"
fig.savefig(output_filename, format='pdf', bbox_inches='tight')

# Close the figure after saving
plt.close(fig)
