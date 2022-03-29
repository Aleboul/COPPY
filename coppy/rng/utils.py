import itertools
import numpy as np
import math
import sys
import scipy.special as sc

DBL_EPSILON = sys.float_info.epsilon

def maximum_n(n, x):
    """
        Step 2 of Algorithm 2.2.
    """
    for i in range(1,n):
        if(x[0] < x[i]) : x[0] = x[i]
    return x[0]

def subsets(d):
    """All subsets of \{1,\dots,d} (empty set is not taken into account).
    
    Output
    ------
        pset (list[list]) : a list containing (2^d)-1 list([int]).
    """
    x = range(0,d)
    pset = [list(subset) for i in range(0, len(x)+1) for subset in itertools.combinations(x,i)]
    del pset[0]
    return pset

def rpstable(cexp):
    """Sample from a Positive Stable distribution.
    """
    if cexp==1: return 0
    tcexp = 1-cexp
    u = np.random.uniform(size = 1) * math.pi
    w = math.log(np.random.exponential(size = 1))
    a = math.log(math.sin(tcexp*u)) + (cexp / tcexp) * math.log(math.sin(cexp*u)) - (1/tcexp) * math.log(math.sin(u))
    return (tcexp / cexp) * (a-w)

def mvrnorm_chol_arma(n, mu, chol_cov):
    """ Set all the elements to random values using 
    a normal/Gaussian distribution with mu mean and chol_cov * chol_cov.T variance.
    Input
    -----
        n        : sample's length
        mu       : mean
        chol_cov : cholesky decomposition of the desired covariance matrix
    Output
    ------
        extremal distribution of the Husler-Reiss
    """

    Y = np.random.normal(size = chol_cov.shape[0]).reshape((n,chol_cov.shape[0]))
    samp = (Y @ chol_cov).reshape((n,chol_cov.shape[0]))
    samp = samp - mu
    return np.squeeze(np.asarray(samp))

def rdir(n, alpha, normalize = True):
    """ Random variate generation for Dirichlet distribution on \eqn{S_{d}}{Sd}
    A function to sample Dirichlet random variables, based on the representation
    as ratios of Gamma. 

    Input
    -----
        n         : sample size
        alpha     : vector of parameter
        normalize : If \code{False}, the function returns Gamma variates with parameter
                    \code{alpha}.
    """
    sample = np.zeros((n,len(alpha)))
    for j in range(0, len(alpha)):
        sample[:,j] = np.random.gamma(alpha[j], 1.0, size =n)
    if normalize:
        for i in range(0,n):
            sample[i,:] = sample[i,:] / np.sum(sample[i,:])

    return sample

"""
    Define a multivariate Unit-Simplex
"""

def simplex(d, n = 1, a = 0, b = 1):
    """http://www.eirene.de/Devroye.pdf
    Algorithm page 207.
    """
    output = np.zeros([n,d])
    for k in range(0,n):
        x_ = np.zeros(d+1)
        y_ = np.zeros(d)
        for i in range(1, d):
            x_[i] = np.random.uniform(a,b)
        x_[d] = 1.0
        x_ = np.sort(x_)
        for i in range(1,d+1):
            y_[i-1] = x_[i] - x_[i-1]
        output[k,:] = y_
    return output

"""
    sample from Sibuya distribution
    see rSibuya.c from R copula package
    and Proposition 3.2 of Efficiently sampling nested Archimedean copulas
"""

def rSibuya(alpha, gamma_1_a):
    U = np.random.uniform(0.0,1.0,1)
    if (U <= alpha):
        return 1.0
    else:
        xMax = 1.0 / DBL_EPSILON
        Ginv = np.power((1-U)*gamma_1_a, -1.0/alpha)
        fGinv = math.floor(Ginv)
        if (Ginv > xMax) : 
            return fGinv
        if (1-U < 1.0 /(fGinv * sc.beta(fGinv, 1.0 - alpha))) :
            return math.ceil(Ginv)

        return fGinv

def rSibuya_vec(V, n, alpha):
    if (n >=1):
        gamma_1_a = sc.gamma(1.0 - alpha)

        for i in range(0,n):
            V[i] = rSibuya(alpha, gamma_1_a)

def rSibuya_vec_c(n, alpha):
    res = np.zeros(n)
    rSibuya_vec(res, n, alpha)
    return res

"""
    Sample a Log(p) distribution
"""

def rLogarithmic(p):
    """
        See logarithmic.c from R package actuar
    """
    if (p < 0 or p >1):
        return np.nan
    if p == 0:
        return 1.0

    if (p < 0.95):
        s = -p/np.log1p(-p)
        x = 1.0
        u = np.random.uniform(0.0,1.0,1)
        
        while (u > s):
            u -= s
            x += 1.0
            s *= p * (x - 1.0) / x

        return x

    else :
        r = np.log1p(-p)
        v = np.random.uniform(0.0,1.0,1)

        if (v >= p):
            return 1.0
        
        u = np.random.uniform(0.0, 1.0, 1)
        q = -np.expm1(r * u)

        if (v <= (q*q)):
            return math.floor(1.0 + math.log(v) / math.log(q))
        if (v <= q):
            return 2.0
        return 1.0
