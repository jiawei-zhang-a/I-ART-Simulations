import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
   
# Add a reference line for convergence rate (-1/2 slope)
x_ref = np.logspace(np.log10(1000), np.log10(10000), num=50)
y_ref = np.log(1 / np.sqrt(x_ref))
x_ref = np.log(x_ref)
print(x_ref)
print(y_ref)
plt.plot(x_ref, y_ref, 'k--', linewidth=2, label='Convergence Rate (slope = -1/2)')
plt.savefig("log" +  ".pdf", format='pdf', bbox_inches='tight')
exit()