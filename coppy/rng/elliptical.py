"""Multivariate elliptical copula module contain class for sample from a multivariate
elliptical copula.
"""

import numpy as np
import math
from .base import Multivariate, CopulaTypes
from scipy.stats import norm, t

class Gaussian(Multivariate):

    copula_type = CopulaTypes.GAUSSIAN

    def is_pos_def(self):
        """Validate the Sigma inserted.
        
        Raises
        ------
            ValueError : If Sigma is not positive semi definite.
        """
        if (not np.all(np.linalg.eigvals(self.Sigma) > 0)):
            message = "The inserted covariance matrix {} is not positive semi definite for the given {} copula."
            raise ValueError(message.format(self.Sigma, self.copula_type.name))

    def sample_unimargin(self):
        self.is_pos_def()
        mu = np.zeros(self.d)
        sample = np.random.multivariate_normal(mean = mu, cov = self.Sigma, size = self.n_sample)
        return norm.cdf(sample)

class Student(Multivariate):

    copula_type = CopulaTypes.STUDENT
    theta_interval = [0, float('inf')]
    invalid_thetas = [0]

    def is_pos_def(self):
        """Validate the Sigma inserted.
        
        Raises
        ------
            ValueError : If Sigma is not positive semi definite.
        """
        if (not np.all(np.linalg.eigvals(self.Sigma) > 0)):
            message = "The inserted covariance matrix {} is not positive semi definite for the given {} copula."
            raise ValueError(message.format(self.Sigma, self.copula_type.name))

    def multivariatet(self):
        '''
        Output:
        Produce n_sample samples of d-dimensional multivariate t distribution
        '''
        g = np.tile(np.random.gamma(self.theta/2.,2./self.theta,self.n_sample),(self.d,1)).T
        Z = np.random.multivariate_normal(np.zeros(self.d),self.Sigma,self.n_sample)
        return Z/np.sqrt(g)

    def sample_unimargin(self):
        sample = self.multivariatet()
        return t.cdf(sample, self.theta)