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
    Line2D([0], [0], color='green', lw=2, linestyle='--')
]

# Create a new figure for the custom_lines_types legend
fig_leg1, ax_leg1 = plt.subplots(figsize=(12, 1))  # Adjust the size as needed
ax_leg1.axis('off')

# Create the legend for custom_lines_types
legend1 = ax_leg1.legend(custom_lines_types, 
                          ['Method 1 (Algo 1 - Linear)', 'Method 2 (Algo 2 - Linear)', 'Method 3 (Algo 1 - Boosting)', 'Method 4 (Algo 2 - Boosting)'], 
                          loc='center', 
                          ncol=4,  # Set to 4 to make all items appear in a single line
                          fontsize='large')  # Adjust as needed

# Save the legend
fig_leg1.savefig('pic/legend_custom_lines_types.pdf', format='pdf', bbox_inches='tight')

