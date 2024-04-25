import numpy as np
from scipy import stats
import pandas as pd

# Load the arrays from the .npz file
arrays = np.load('Data/arrays.npz')

X = arrays['X']
Y = arrays['Y']

def summarize_data(X, Y):
    summary = {}
    for i, col in enumerate(X.T, 1):
        # Ignoring NaN values for calculations
        valid_col = col[~np.isnan(col)]
        
        summary[f'X_{i}'] = {
            'Frequency': dict(zip(*np.unique(col, return_counts=True))),
            'Portion': {k: v / len(col) for k, v in zip(*np.unique(col, return_counts=True))},
            'Missing': np.count_nonzero(np.isnan(col)),
            'Missing Portion': np.count_nonzero(np.isnan(col)) / len(col),
            'Mean': np.mean(valid_col),
            'Median': np.median(valid_col),
            'Mode': stats.mode(valid_col)[0][0] if len(valid_col) > 0 else np.NaN,
            'Minimum': np.min(valid_col),
            'Maximum': np.max(valid_col),
            'Standard Deviation': np.std(valid_col)
        }
    
    valid_Y = Y[~np.isnan(Y)]
    summary['Y'] = {
        'Frequency': dict(zip(*np.unique(Y, return_counts=True))),
        'Portion': {k: v / len(Y) for k, v in zip(*np.unique(Y, return_counts=True))},
        'Missing': np.count_nonzero(np.isnan(Y)),
        'Missing Portion': np.count_nonzero(np.isnan(Y)) / len(Y),
        'Mean': np.mean(valid_Y),
        'Median': np.median(valid_Y),
        'Mode': stats.mode(valid_Y)[0][0] if len(valid_Y) > 0 else np.NaN,
        'Minimum': np.min(valid_Y),
        'Maximum': np.max(valid_Y),
        'Standard Deviation': np.std(valid_Y)
    }
    
    return summary

summary_stats = summarize_data(X, Y)

def generate_latex_tables(summary_stats):
    latex_code = ""
    
    for var, stats in summary_stats.items():
        main_stats_latex = f"""
\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{|l|l|}}
\\hline
\\textbf{{Statistic}} & \\textbf{{Value}} \\\\ \\hline
Mean               & {stats['Mean']:.5f} \\\\ \\hline
Median             & {stats['Median']:.5f} \\\\ \\hline
Mode               & {stats['Mode']} \\\\ \\hline
Minimum            & {stats['Minimum']:.5f} \\\\ \\hline
Maximum            & {stats['Maximum']:.5f} \\\\ \\hline
Standard Deviation & {stats['Standard Deviation']:.5f} \\\\ \\hline
Missing            & {stats['Missing']} \\\\ \\hline
Missing Portion    & {stats['Missing Portion']*100:.5f}\\% \\\\ \\hline
\\end{{tabular}}
\\caption{{Summary Statistics for {var}}}
\\end{{table}}
"""
        latex_code += main_stats_latex

        values_stats_latex = f"""
\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{|l|l|l|}}
\\hline
\\textbf{{Value}} & \\textbf{{Frequency}} & \\textbf{{Portion (\\%)}} \\\\ \\hline
"""
        for value, freq in stats['Frequency'].items():
            if pd.isna(value):  # Handle NaN values specifically
                value_label = "Missing"
                portion = stats['Missing Portion']
            else:
                value_label = value
                portion = stats['Portion'].get(value, 0)  # Use .get to avoid KeyError, default to 0
            values_stats_latex += f"{value_label} & {freq} & {portion*100:.2f}\\% \\\\ \\hline\n"
        
        values_stats_latex += "\\end{tabular}\n\\caption{Frequency and Portion for " + var + "}\n\\end{table}\n"
        latex_code += values_stats_latex

    return latex_code

latex_code = generate_latex_tables(summary_stats)

# Save the latex_code to a .tex file
with open('Data/summary_stats.tex', 'w') as f:
    f.write(latex_code)
