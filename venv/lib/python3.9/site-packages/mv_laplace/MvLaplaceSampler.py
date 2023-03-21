import numpy as np
from scipy import stats

class MvLaplaceSampler:
    def __init__(self, loc: np.array, cov: np.array):
        self.loc, self.var, self.cov = loc, np.diag(cov), cov
        
        self.mv_normal = stats.multivariate_normal(mean=self.loc, cov=self.cov)
        self.normal = stats.norm(loc=self.loc, scale=np.sqrt(self.var))
        self.laplace = stats.laplace(loc=self.loc, scale=np.sqrt(self.var/2))
        
    def sample(self, sample_size: int=None):
        mv_samples = self.mv_normal.rvs(sample_size)
        cdf_samples = self.normal.cdf(mv_samples)
        laplace_samples = self.laplace.ppf(cdf_samples)
        return laplace_samples
