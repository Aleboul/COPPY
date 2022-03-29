"""Base module contains method for sampling from a multivariate extreme value copula and
to compute the asymptotic variance of the w-madogram with missing or complete data.

A multivariate copula $C : [0,1]^d \rightarrow [0,1]$ of a d-dimensional random vector $X$ allows
us to separate the effect of dependence from the effect of the marginal distributions. The
copula function completely chracterizes the stochastic dependence between the margins of $X$.
Extreme value copulas are characterized by the stable tail dependence function which the restriction
to the unit simplex is called Pickands dependence function.

Structure :

- Multivariate copula (:py:class:`Multivariate`)
    - Extreme value copula (:py:class:`Extreme`)
"""
Epsilon = 1e-12
import numpy as np
import numpy.matlib
import math

from scipy.integrate import quad
from enum import Enum
from scipy.optimize import brentq

class CopulaTypes(Enum):
    """
        Available multivariate copula
    """

    CLAYTON = 1
    AMH = 3
    GUMBEL = 4
    FRANK = 5
    JOE = 6
    NELSEN_9 = 9
    NELSEN_10 = 10
    NELSEN_11 = 11
    NELSEN_12 = 12
    NELSEN_13 = 13
    NELSEN_14 = 14
    NELSEN_15 = 15
    NELSEN_22 = 22
    HUSLER_REISS = 23
    ASYMMETRIC_LOGISTIC = 24
    ASYMMETRIC_NEGATIVE_LOGISTIC = 25
    ASYMMETRIC_MIXED_MODEL = 26
    tEV = 27
    GAUSSIAN = 28
    STUDENT = 29
    DIRICHLET = 30
    BILOG = 31

class Multivariate(object):
    """Base class for multivariate copulas.
    This class allows to instantiate all its subclasses and serves
    as a unique entry point for the multivariate copulas classes.
    It permit also to compute the variance of the w-madogram for a given point.

    Attributes
    ----------
        copula_type(CopulaTypes)    : see CopulaTypes class.
        d(int)                      : dimension.
        theta_interval(list[float]) : interval of valid theta for the given copula family
        invalid_thetas(list[float]) : values, that even though they belong to
                                      :attr: `theta_interval`, shouldn't be considered as valid.
        theta(list[float])          : parameter for the parametric copula
        var_mado(list[float])       : value of the theoretical variance for a given point in the simplex
        Sigma                       : covariance matrix (only for elliptical, Husler-Reiss).
        asy(list[float])            : asymmetry for the Asy. Log. model.
        psi1, psi2(float)           : asymmetry for the Asy. Log. model when d = 2.

    Methods
    -------
        sample (np.array([float])) : array of shape n_sample x d of the desired multivariate copula model
                                     where the margins are inverted by the specified generalized inverse 
                                     of a cdf.
    """
    copula_type = None
    theta_interval = []
    invalid_thetas = []
    n_sample = []
    psi1 = 0.0
    psi2 = []
    asy = []
    theta = []
    d = None
    Sigma = None

    def __init__(self, theta = None, n_sample = None, asy = None, psi1 = 0.0, psi2 = None, d = 2, Sigma = None):
        """Initialize Multivariate object.
            
        Inputs
        ------
            copula_type (copula_type or st) : subtype of the copula
            theta (list[float] or None)     : list of parameters for the parametric copula
            asy (list[float] or None)       : list of asymetric values for the copula
            n_sample (int or None)          : number of sampled observation
            d (int or None)                 : dimension
        """
        self.theta = theta
        self.n_sample = n_sample
        self.asy = asy
        self.psi1 = psi1
        self.psi2 = psi2
        self.d = d
        self.Sigma = Sigma

    def check_theta(self):
        """Validate the theta inserted.
        
        Raises
        ------
            ValueError : If there is not in :attr:`theta_interval` or is in :attr:`invalid_thetas`.
        """
        lower, upper = self.theta_interval
        if (not lower <= self.theta <= upper) or (self.theta in self.invalid_thetas):
            message = "The inserted theta value {} is out of limits for the given {} copula."
            raise ValueError(message.format(self.theta, self.copula_type.name))

    def _generate_randomness(self):
        """Generate a bivariate sample draw identically and
        independently from a uniform over the segment [0,1].

        Output
        ------
            output (np.array([float]) with shape n_sample x 2) : a n_sample x 2 array with each component
                                                                 sampled from the desired copula under 
                                                                 the unit interval.
        """
        v_1 = np.random.uniform(low = 0.0, high = 1.0, size = self.n_sample)
        v_2 = np.random.uniform(low = 0.0, high = 1.0, size = self.n_sample)
        output = np.vstack([v_1, v_2]).T
        return output
    
    def sample(self, inv_cdf):
        """Draws a bivariate sample the desired copula and invert it by
        a given generalized inverse of cumulative distribution function

        Inputs
        ------
            inv_cdf : generalized inverse of cumulative distribution function

        Output
        ------
            output (np.array([float]) with sape n_sample x d) : sample where the margins
                                                                are specified inv_cdf.
        """
        if isinstance(inv_cdf, list) == False:
            message = "inv_cdf should be a list"
            raise ValueError(message)
        if len(inv_cdf) == 1:
            inv_cdf = np.repeat(inv_cdf, self.d)
        elif len(inv_cdf) == self.d :
            pass
        else :
            message = "inv_cdf should be a list of length 1 or {}"
            raise ValueError(message.format(self.d))
        sample_ = self.sample_unimargin()
        output = np.array([inv_cdf[j](sample_[:,j]) for j in range(0, self.d)])
        output = np.ravel(output).reshape(self.n_sample, self.d, order = 'F')
        return output

class Archimedean(Multivariate):
    """Base class for multivariate archimedean copulas.
    This class allowd to use methods which use the generator function.

    Methods
    -------
        sample_uni (np.array([float])) : sample form the multivariate archimedean copula.
                                         Margins are uniform on [0,1].
    """

    def _C(self,u):
        """Return the value of the copula taken on (u,v)
        .. math:: C(u,v) = \phi^\leftarrow (\phi(u) + \phi(v)), 0<u,v<1
        """
        value_ = self._generator_inv(np.sum(self._generator(u)))
        return value_
    
    def _cond_sim(self):
        """Perform conditional simulation. Only useful for Archimedean copulas
        where the frailty distribution is still unknown.

        CopulaTypes
        -----------
            NELSEN_9
            NELSEN_10
            NELSEN_11
            NELSEN_12
            NELSEN_13
            NELSEN_14
            NELSEN_15
            NELSEN_22
        """
        self.check_theta()
        if self.d == 2:
            output = np.zeros((self.n_sample, self.d))
        else :
            message = "This generator can't generate an Archimedean copula for d greater than 2"
            raise ValueError(message)
        X = self._generate_randomness()
        Epsilon = 1e-12
        for i in range(0,self.n_sample):
            v = X[i]
            def func(x):
                value_ = ( x - self._generator(x) / self._generator_dot(x)) - v[1]
                return(value_)
            if func(Epsilon) > 0.0 :
                sol = 0.0
            else :
                sol = brentq(func, Epsilon,1-Epsilon)
            u = [self._generator_inv(v[0] * self._generator(sol)) , self._generator_inv((1-v[0])*self._generator(sol))]
            output[i,:] = u
        return output

    def _frailty_sim(self):
        """Sample from Archimedean copula using algorithm where the frailty
        distribution is known.

        CopulaTypes
        -----------
            CLAYTON
            AMH
            JOE
            FRANK
        """
        if self.d > 2:
            if self.theta < 0:
                message = "The inserted theta value {} is out of limits for the given {} copula. In dimension greater than 2, positive association are only allowed."
                raise ValueError(message.format(self.theta, self.copula_type.name))
        self.check_theta()
        output = np.zeros((self.n_sample, self.d))
        for i in range(0, self.n_sample):
            e = np.random.gamma(1, 1, self.d)
            v = self.rFrailty()
            u = self._generator_inv(e/v)
            output[i,:] = u
        return output

    def sample_unimargin(self):
        """Sample from Archimedean copula with uniform margins.
        Performs different algorithm if the frailty distribution of the
        chosen Archimedean copula is known or not.

        Output
        ------
            output (np.array([float]) with shape n_sample x d) : sample from the desired
                                                                 copula with uniform margins.

        RaiseValueError
        ---------------
            generator function can't generate Archimedean copula for d > 2
        
        """
        output = []
        condsim_numbers = [CopulaTypes.NELSEN_9,CopulaTypes.NELSEN_10,CopulaTypes.NELSEN_11,CopulaTypes.NELSEN_12,CopulaTypes.NELSEN_13,
                           CopulaTypes.NELSEN_14,CopulaTypes.NELSEN_15,CopulaTypes.NELSEN_22]
        frailty_numbers = [CopulaTypes.FRANK, CopulaTypes.AMH, CopulaTypes.JOE, CopulaTypes.CLAYTON]
        if (self.copula_type in condsim_numbers):
            output = self._cond_sim()
        if self.copula_type in frailty_numbers:
            if (self.d == 2) and (self.theta < 0):
                output = self._cond_sim()
            else :
                output = self._frailty_sim()
        return output
        

class Extreme(Multivariate):
    """Base class for multivariate extreme value copulas.
    This class allows to use methods which use the Pickands dependence function.

    Methods
    -------
        sample_uni (np.array([float])) : sample from the desired multivariate copula model. 
                                         Margins are uniform on [0,1].
        var_FMado (float)              : gives the asymptotic variance of w-madogram for a
                                         multivariate extreme value copula.
    
    Examples
    --------
        >>> import base
        >>> import mv_evd
        >>> import matplotlib.pyplot as plt

        >>> copula = mv_evd.Logistic(theta = 0.5, d = 3, n_sample = 1024)
        >>> sample = copula.sample_uni()

        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, projection = '3d')
        >>> ax.scatter3D(sample[:,0],sample[:,1],sample[:,2], c = 'lightblue', s = 1.0, alpha = 0.5)
        >>> plt.show()
    """

    def _frechet(self, x):
        """
            Probability distribution function for Frechet's law
        """
        return np.exp(-1/x)

    def _l(self, u):
        """Return the value of the stable tail dependence function on u.
        Pickands is parametrize as A(w_0, \dots, w_{d-1}) with w_0 = 1-\sum_{j=1}^{d-1} w_j

        Inputs
        ------
            u (list[float]) : d-list with each components between 0 and 1.
        """
        s  = np.sum(u)
        w_ = u / s
        value_ = s*self._A(w_)
        return value_

    def _C(self, u):
        """Return the value of the copula taken on u
        .. math:: C(u) = exp(-\ell(-log(u_1), \dots, -log(u_d))), \quad u \in [0,1]^d.

        Inputs
        ------
            u (list[float]) : d-list of float between 0 and 1.
        """
        log_u_ = np.log(u)
        value_ = math.exp(-self._l(-log_u_))
        return value_

    def _mu(self, u, j):
        """Return the value of the jth partial derivative of l.
        ..math:: \dot{\ell}_j(u), \quad u \in ]0,1[^d, \quad, j \in \{0,\dots,d-1\}.

        Inputs
        ------
            u (list[float]) : list of float between 0 and 1.
            j (int)         : jth derivative of the stable tail dependence function.
        """
        s = np.sum(u)
        w_ = u / s
        if j == 0 :
            deriv_ = []
            for j in range(1,self.d):
                value_deriv = self._Adot(w_, j) * w_[j]
                deriv_.append(value_deriv)
            value_ = self._A(w_) - np.sum(deriv_)
            return value_
        else :
            deriv_ = []
            for i in range(1, self.d):
                if i == j:
                    value_deriv = -(1-w_[i]) * self._Adot(w_, i)
                    deriv_.append(value_deriv)
                else:
                    value_deriv = self._Adot(w_, i) * w_[i]
                    deriv_.append(value_deriv)
            value_ = self._A(w_) - np.sum(deriv_)
            return value_

    def _dotC(self, u, j):
        """Return the value of \dot{C}_j taken on u.
        .. math:: \dot{C}_j = C(u)/u_j * _mu_j(u), \quad u \in [0,1]^d, \quad j \in \{0 , \dots, d-1\}.
        """
        value_ = (self._C(u) / u[j]) * self._mu(-np.log(u),j)
        return value_

    def _cond_sim(self):
        """Draw a bivariate sample from an extreme value copula using conditional simulation.
        Margins are uniform.

        CopulaTypes
        -----------
            ASYMMETRIC_MIXED
            ASYMMETRIC_NEGATIVE_LOGISTIC
        """
        self.check_theta()
        if self.d > 2:
            message = "The dimension {} inserted is not compatible with the {} extreme value model chosen."
            raise ValueError(message.format(self.d, self.copula_type.name))
        output = np.zeros((self.n_sample,self.d))
        X = self._generate_randomness()
        for i in range(0,self.n_sample):
            v = X[i]
            def func(x):
                u = np.array([v[0],x])
                value_ = self._dotC(u, 0) - v[1]
                return(value_)
            sol = brentq(func, Epsilon,1-Epsilon)
            u = [v[0], sol]
            output[i,:] = u
        return output

    def _ext_sim(self):
        """ Multivariate extreme value distribution sampling algorithm via extremal functions.
        See Dombry et al [2016], exact simulation of max-stable process for more details.

        CopulaTypes
        -----------
            HUSLER_REISS
            tEV
            DIRICHLET
        """
        if self.copula_type == CopulaTypes.tEV:
            stdev = np.exp(0.5 * np.log(np.diag(self.Sigma)))
            stdevmat = np.linalg.inv(np.diag(stdev))
            self.Sigma = stdevmat @ self.Sigma @ stdevmat

        output = np.zeros((self.n_sample, self.d))
        matsim = [CopulaTypes.HUSLER_REISS, CopulaTypes.tEV]
        dir = [CopulaTypes.DIRICHLET, CopulaTypes.BILOG]
        for i in range(0, self.n_sample):
            zeta_I = np.random.exponential(1)
            if self.copula_type in matsim:
                Covar = self._Sigma2Covar(0)
                cholesky = np.linalg.cholesky(Covar).T
                Y = self.rExtFunc(0, cholesky)
            if self.copula_type in dir:
                Y = self.rExtFunc(0)
            
            output[i,:] = Y / zeta_I

            for j in range(1, self.d):
                zeta_I = np.random.exponential(1)
                if self.copula_type in matsim:
                    Covar = self._Sigma2Covar(j)
                    cholesky = np.linalg.cholesky(Covar).T

                while (1.0 / zeta_I > output[i,j]):
                    if self.copula_type in matsim:
                        Y = self.rExtFunc(j, cholesky)
                    if self.copula_type in dir:
                        Y = self.rExtFunc(j)
                    res = True
                    for k in range(0, j):
                        if (Y[k] / zeta_I >= output[i,k]):
                            res = False
                    
                    if res:
                        output[i,:] = np.maximum(output[i,:], Y / zeta_I)
                    zeta_I += np.random.exponential(1)

        return output

    def sample_unimargin(self):
        """Sample from EV copula with uniform margins.
        Performs different algorithms if a fast random number generator
        is known.

        Output
        ------
            output (np.array([float]) with shape n_sample x d) : sample from the desired
                                                                 copula with uniform margins.

        RaiseValueError
        ---------------
            generator function can't generate Archimedean copula for d > 2
        """
        output = []
        condsim_numbers = [CopulaTypes.ASYMMETRIC_NEGATIVE_LOGISTIC, CopulaTypes.ASYMMETRIC_MIXED_MODEL,CopulaTypes.ASYMMETRIC_NEGATIVE_LOGISTIC]
        extsim_numbers = [CopulaTypes.HUSLER_REISS, CopulaTypes.tEV, CopulaTypes.DIRICHLET, CopulaTypes.BILOG]
        if self.copula_type in condsim_numbers:
            output = self._cond_sim()
        if self.copula_type == CopulaTypes.GUMBEL:
            output = self._frechet(self.rmvlog_tawn())
            output.reshape(self.n_sample, self.d)
        if self.copula_type == CopulaTypes.ASYMMETRIC_LOGISTIC :
            nb = int(2**self.d -1)
            dep = np.repeat(self.theta, nb - self.d)
            if (self.d == 2) and (self.asy is None):
                self.asy = [self.psi1, self.psi2, [1-self.psi1, 1-self.psi2]]
            asy = self.mvalog_check(dep).reshape(-1)
            dep = np.concatenate([np.repeat(1,self.d), dep], axis = None)
            output = self._frechet(self.rmvalog_tawn(nb,dep,asy))
            output = output.reshape(self.n_sample,self.d)
        if self.copula_type in extsim_numbers:
            output = np.exp(-1/self._ext_sim())
        return output

    def true_wmado(self, w):
        """Return the value of the w_madogram taken on w.

        Inputs
        ------
            w (list of [float]) : element of the simplex.
        """
        value = self._A(w) / (1+self._A(w)) - (1/self.d)*np.sum(w / (1+w))
        return value

    # Compute asymptotic variance of the multivariate madogram

    def _integrand_ev1(self, s, w, j):
        """First integrand.

        Inputs
        ------
            s(float)       : float between 0 and 1.
            w(list[float]) : d-array of the simplex.
            j(int)         : int \geq 0.
        """
        z = s*w / (1-w[j])
        z[j] = (1-s) # start at 0 if j = 1
        A_j = self._A(w) / w[j]
        value_ = self._A(z) + (1-s)*(A_j + (1-w[j])/w[j] - 1) + s*w[j] / (1-w[j])+1
        return math.pow(value_,-2)

    def _integrand_ev2(self, s, w, j, k):
        """Second integrand.

        Inputs
        ------
            s (float)       : float between 0 and 1.
            w (list[float]) : d-array of the simplex.
            j (int)         : int \geq 0.
            k (int)         : int \neq j.
        """
        z = 0 * w
        z[j] = (1-s)
        z[k] = s
        A_j = self._A(w) / w[j]
        A_k = self._A(w) / w[k]
        value_ = self._A(z) + (1-s) * (A_j + (1-w[j])/w[j] - 1) + s * (A_k + (1-w[k])/w[k] -1) + 1
        return math.pow(value_, -2)
    
    def _integrand_ev3(self,s, w, j, k):
        """Third integrand.

        Inputs
        ------
            s(float)       : float between 0 and 1.
            w(list[float]) : d-array of the simplex.
            j(int)         : int \geq 0.
            k(int)         : int \neq j.
        """
        z = 0 * w
        z[j] = (1-s)
        z[k] = s
        value_ = self._A(z) + (1-s) * (1-w[j])/w[j] + s * (1-w[k])/w[k]+1
        return math.pow(value_,-2)

    def _integrand_ev4(self, s, w, j):
        """Fourth integrand.

        Inputs
        ------
            s (float)       : float between 0 and 1.
            w (list[float]) : d-array of the simplex.
            j (int)         : int \geq 0.
        """
        z = s*w / (1-w[j])
        z[j] = (1-s)
        value_ = self._A(z) + (1-s) * (1-w[j])/w[j] + s*w[j]/(1-w[j])+1
        return math.pow(value_,-2)

    def _integrand_ev5(self, s, w, j,k):
        """Fifth integrand.

        Inputs
        ------
            s(float)       : float between 0 and 1.
            w(list[float]) : d-array of the simplex.
            j(int)         : int \geq 1.
            k(int)         : int \geq j.
        """
        z = 0 * w
        z[j] = (1-s)
        z[k] = s
        A_k = self._A(w) / w[k]
        value_ = self._A(z) + (1-s) * (1-w[j])/w[j] + s * (A_k + (1-w[k])/w[k]-1)+1
        return math.pow(value_,-2)

    def _integrand_ev6(self, s, w, j,k):
        """Sixth integrand.

        Inputs
        ------
            s(float)       : float between 0 and 1.
            w(list[float]) : d-array of the simplex.
            j(int)         : int \geq k.
            k(int)         : int \geq 0.
        """
        z = 0 * w
        z[k] = (1-s)
        z[j] = s
        A_k = self._A(w) / w[k]
        value_ = self._A(z) + (1-s) * (A_k + (1-w[k])/w[k]-1) + s * (1-w[j])/w[j]+1
        return math.pow(value_,-2)

    def var_mado(self, w, P, p, corr = {False, True}):
        """Return the variance of the Madogram for a given point on the simplex

        Inputs
        ------
            w (list[float])  : array in the simplex .. math:: (w_0, \dots, w_{d-1}).
            P (array[float]) : d \times d array of probabilities, margins are in the diagonal
                               while probabilities of two entries may be missing are in the antidiagonal.
            p ([float])      : joint probability of missing.
        """

        if corr :
            lambda_ = w
        else :
            lambda_ = np.zeros(self.d)

        ## Calcul de .. math:: \sigma_{d+1}^2
        squared_gamma_1 = math.pow(p,-1)*(math.pow(1+self._A(w),-2) * self._A(w) / (2+self._A(w)))
        squared_gamma_ = []
        for j in range(0,self.d):
            v_ = math.pow(P[j][j],-1)*(math.pow(self._mu(w,j) / (1+self._A(w)),2) * w[j] / (2*self._A(w) + 1 + 1 - w[j]))
            squared_gamma_.append(v_)
        gamma_1_ = []
        for j in range(0, self.d):
            v_1 = self._mu(w,j) / (2 * math.pow(1+self._A(w),2)) * (w[j] / (2*self._A(w) + 1 + 1 - w[j]))
            v_2 = self._mu(w,j) / (2 * math.pow(1+self._A(w),2))
            v_3 = self._mu(w,j) / (w[j]*(1-w[j])) * quad(lambda s : self._integrand_ev1(s, w, j), 0.0, 1-w[j])[0]
            v_  = math.pow(P[j][j],-1)*(v_1 - v_2 + v_3)
            gamma_1_.append(v_)
        tau_ = []
        for k in range(0, self.d):
            for j in range(0, k):
                v_1 = self._mu(w,j) * self._mu(w,k) * math.pow(1+self._A(w),-2)
                v_2 = self._mu(w,j) * self._mu(w,k) / (w[j] * w[k]) * quad(lambda s : self._integrand_ev2(s, w, j, k), 0.0, 1.0)[0]
                v_  = (P[j][k] / (P[j][j] * P[k][k]))*(v_2 - v_1)
                tau_.append(v_)

        squared_sigma_d_1 = squared_gamma_1 + np.sum(squared_gamma_) - 2 * np.sum(gamma_1_) + 2 * np.sum(tau_)
        if p < 1:
            ## Calcul de .. math:: \sigma_{j}^2
            squared_sigma_ = []
            for j in range(0, self.d):
                v_ = (math.pow(p,-1) - math.pow(P[j][j],-1))*math.pow(1+w[j],-2) * w[j]/(2+w[j])
                v_ = math.pow(1+lambda_[j]*(self.d-1),2) * v_
                squared_sigma_.append(v_)

            ## Calcul de .. math:: \sigma_{jk} with j < k
            sigma_ = []
            for k in range(0, self.d):
                for j in range(0,k):
                    v_1 = 1/(w[j] * w[k]) * quad(lambda s : self._integrand_ev3(s, w, j, k), 0.0,1.0)[0]
                    v_2 = 1/(1+w[j]) * 1/(1+w[k])
                    v_  = (math.pow(p,-1) - math.pow(P[j][j],-1) - math.pow(P[k][k],-1) + P[j][k]/(P[j][j]*P[k][k]))*(v_1 - v_2)
                    v_  = (1+lambda_[j]*(self.d-1)) * (1+lambda_[k]*(self.d-1)) * v_
                    sigma_.append(v_)

            ## Calcul de .. math:: \sigma_{j}^{(1)}, j \in \{1,dots,d\}
            sigma_1_ = []
            for j in range(0, self.d):
                v_1 = 1/(w[j] *(1-w[j])) * quad(lambda s : self._integrand_ev4(s, w, j), 0.0,1 - w[j])[0]
                v_2 = 1/(1+self._A(w)) * (1/(2+self._A(w)) - 1 / (1+w[j]))
                v_  = (math.pow(p,-1) - math.pow(P[j][j],-1))*(v_1 + v_2)
                v_  = (1+lambda_[j]*(self.d-1))*v_
                sigma_1_.append(v_)

            sigma_2_ = []
            for k in range(0, self.d):
                for j in range(0, self.d):
                    if j == k:
                        v_ = 0
                        sigma_2_.append(v_)
                    elif j < k:
                        v_1 = self._mu(w,k) / (w[j] * w[k]) * quad(lambda s : self._integrand_ev5(s, w, j, k), 0.0,1.0)[0]
                        v_2 = self._mu(w,k) / (1+self._A(w)) * 1 /(1+w[j])
                        v_  = (math.pow(P[k][k],-1) - P[j][k]/(P[j][j]*P[k][k]))*(v_1 - v_2)
                        v_  = (1 + lambda_[j]*(self.d-1))*v_
                        sigma_2_.append(v_)
                    else :
                        v_1 = self._mu(w,k) / (w[j] * w[k]) * quad(lambda s : self._integrand_ev6(s, w, j, k), 0.0,1.0)[0]
                        v_2 = self._mu(w,k) / (1+self._A(w)) * 1 /(1+w[j])
                        v_  = (math.pow(P[k][k],-1) - P[k][j]/(P[j][j]*P[k][k]))*(v_1 - v_2)
                        v_  = (1 + lambda_[j]*(self.d-1))*v_
                        sigma_2_.append(v_)

            if corr :
                return (1/self.d**2) * np.sum(squared_sigma_) + squared_sigma_d_1 + (2/self.d**2) * np.sum(sigma_) - (2/self.d) * np.sum(sigma_1_) + (2/self.d) * np.sum(sigma_2_)
            else :
                return (1/self.d**2) * np.sum(squared_sigma_) + squared_sigma_d_1 + (2/self.d**2) * np.sum(sigma_) - (2/self.d) * np.sum(sigma_1_) + (2/self.d) * np.sum(sigma_2_)
        
        else:
            return squared_sigma_d_1