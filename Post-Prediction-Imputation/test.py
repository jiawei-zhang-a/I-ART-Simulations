from statsmodels.stats.multitest import multipletests
p_values = [0.1,0.2,0.3]
reject, corrected_p_values, _, _ = multipletests(p_values,  method='holm')
print(corrected_p_values)