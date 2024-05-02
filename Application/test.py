from scipy.stats import rankdata
import numpy as np
    #the Wilcoxon rank sum test

y = np.array([1,2,2,3,3,3])
z = np.array([1,1,0,0,0,1])

def test(y,z):
    Y_rank = rankdata(y)
    print(Y_rank)
    t = np.sum(Y_rank[z == 1])
    return t
    
    
    


def test2(y,z):
    n = len(z)
    t = 0
    my_list = []
    for i in range(n):
        my_list.append((z[i],y[i], i+1))
    sorted_list = sorted(my_list, key=lambda x: x[1])
    for i in range(n):
        t += sorted_list[i][0] * (i + 1)
    
    print(sorted_list)

test(y,z)
test2(y,z)