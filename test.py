
from multiprocessing import Pool,Process, Queue, current_process, freeze_support
import numpy as np

def worker(args):
    a, b = args
    return a * a, b * b
def test():
    n_jobs = 4
    args_list = [(1,2)] * n_jobs

    with Pool(processes=n_jobs) as pool:
        print("sadasdasd")
        p_list = pool.map(worker, args_list)
    p1 = np.mean([p[0] for p in p_list], axis=0)
    p2 = np.mean([p[1] for p in p_list], axis=0)
    print(p1, p2)

freeze_support()
test()