"""Multivariate extreme value copula module contains methods for sampling from a multivariate
extreme value copula and to compute the asymptotic variance of the w-madogram under missing or
complete data.

Multivariate extreme value copulas are characterized by their stable tail dependence function 
which the restriction to the unit simplex gives the Pickands dependence function. The copula 
function

..math:: C(u) = exp\{-\ell(-log(u_1), \dots, -log(u_d))\}, \quad 0 < u_j \leq 1,

is a multivariate extreme value copula. To sample from a multivariate extreme value copula, we 
implement the Algoritm 2.1 and 2.2 from Stephenson (2002).

Structure :

- Extreme value copula (:py:class:`Extreme`) from copy.multivariate.base.py
    - Logistic model (:py:class:`Logistic`)
    - Asymmetric logistic model (:py:class:`Asymmetric_logistic`)
"""
import numpy as np
import numpy.matlib
import math

from .utils import rpstable, maximum_n, subsets, mvrnorm_chol_arma, rdir
from .base import Multivariate, CopulaTypes, Extreme
from scipy.stats import norm
from scipy.stats import t

"""
    Commentaire : 
        Gérer les indices de dérivations
"""

class Logistic(Extreme):
    """
        Class for multivariate Logistic copula model.
    """

    copula_type = CopulaTypes.GUMBEL
    theta_interval = [0, 1]
    invalid_thetas = [0]

    def _A(self, t):
        """Return the value of the Pickands dependence function taken on t.
        ..math:: A(t) = (\sum_{j=1}^d t_i^{1/\theta})^\theta, \quad t \in \Delta^{d-1}.

        Inputs
        ------
            t (list[float]) : list of elements of the simplex in R^{d}
        """
        s = np.sum(t)
        value_ = math.pow(np.sum(np.power(t, 1/self.theta)),self.theta)

        return value_

    def _Adot(self, t, j):
        """Return the value of jth partial derivative of the Pickands dependence function taken on t

        Inputs
        ------
            t(list[float]) : list of elements of the simplex in R^{d}
                         j : index of the partial derivative \geq 1
        """
        s = np.sum(t[1:]) # \sum_{j=1}^{d-1} t_j
        value_1 = (1/self.theta * math.pow(t[j],(1-self.theta)/self.theta) - 1/self.theta * math.pow(1-s,(1-self.theta)/self.theta))
        value_2 = math.pow(self._A(t), (self.theta - 1)/self.theta)
        value_  = self.theta * value_1 * value_2
        return value_

    def rmvlog_tawn(self):
        """ Algorithm 2.1 of Stephenson (2002).
        """
        sim = np.zeros(self.n_sample * self.d)
        for i in range(0 , self.n_sample):
            s = rpstable(self.theta)
            for j in range(0,self.d):
                sim[i*self.d + j] = math.exp(self.theta * (s - math.log(np.random.exponential(size = 1))))
        return sim

    def sample_unimargin(self):
        """Draws a sample from a multivariate Logistic model.

        Output
        ------
        sim (np.array([float])) : dataset of shape n_sample x d
        """
        sim = self._frechet(self.rmvlog_tawn())
        return sim.reshape(self.n_sample, self.d)

class Asymmetric_logistic(Extreme):
    """
        Class for multivariate asymmetric logistic copula model
    """

    copula_type = CopulaTypes.ASYMMETRIC_LOGISTIC
    
    def _A(self, t):
        """Return the value of the Pickands dependence function taken on t
        ..math:: A(t) = \sum_{b \in B} (\sum_{j \in b} (\psi_{j,b} t_j)^{1/\theta_b}))^{\theta_b}, \quad t \in \Delta^{d-1}

        Inputs
        ------
            t (list[float]) : list of elements of the simplex in R^{d}
        """
        nb = int(2**self.d - 1)
        dep = np.repeat(self.theta, nb - self.d)
        dep = np.concatenate([np.repeat(1,self.d), dep], axis = None)
        asy = self.mvalog_check(dep)
        A_ = []
        for b in range(0, nb):
            x = np.power(asy[b,:], 1/dep[b])
            y = np.power(t, 1/dep[b])
            value = np.dot(x, y)
            A_.append(np.power(value, dep[b]))

        return np.sum(A_)

    def _Adot(self, t,j):
        """Return the value of jth partial derivative of the Pickands dependence function taken on t

        Inputs
        ------
            t(list[float]) : list of elements of the simplex in R^{d-1}
                         j : index of the partial derivative >= 1
        """
        nb = int(2**self.d - 1)
        dep = np.repeat(self.theta, nb - self.d)
        dep = np.concatenate([np.repeat(1,self.d), dep], axis = None)
        asy = self.mvalog_check(dep)
        Adot_ = []
        for b in range(0, nb):
            z = np.zeros(self.d) ; z[0] = -np.power(t[0], (1-dep[b]) / dep[b]) ; z[j] = np.power(t[j], (1-dep[b]) / dep[b])
            x = np.power(asy[b,:], 1/dep[b])
            y = np.power(t, 1/dep[b])
            value_1 = np.dot(x, z)
            value_2 = np.power(np.dot(x,y), (dep[b] - 1))
            Adot_.append(value_1 * value_2)

        return np.sum(Adot_)


    def rmvalog_tawn(self,nb, alpha, asy):
        """ Algorithm 2.2 of Stephenson (2008). """
        sim = np.zeros(self.n_sample*self.d)
        gevsim = np.zeros(nb*self.d)
        maxsim = np.zeros(nb)
        for i in range(0,self.n_sample):
            for j in range(0, nb):
                if alpha[j] != 1:
                    s = rpstable(alpha[j])
                else: s = 0
                for k in range(0, self.d):
                    if asy[j*self.d+k] != 0:
                        gevsim[j*self.d+k] = asy[j*self.d+k] * math.exp(alpha[j] * (s -math.log(np.random.exponential(size = 1))))

            for j in range(0,self.d):
                for k in range(0,nb):
                    maxsim[k] = gevsim[k*self.d+j]

                sim[i*self.d+j] = maximum_n(nb, maxsim)

        return sim
    
    def mvalog_check(self, dep):
        if(dep.any() <= 0 or dep.any() > 1.0):
            raise TypeError('invalid argument for theta')
        nb = 2 ** self.d - 1
        if(not isinstance(self.asy, list) or len(self.asy) != nb) :
            raise TypeError('asy should be a list of length', nb)

        def tasy(theta, b):
            trans = np.zeros([nb,self.d])
            for i in range(0, nb):
                j = b[i]
                trans[i,j] = theta[i]
            return trans
        
        b = subsets(self.d)
        asy = tasy(self.asy, b)
        y = np.sum(asy, axis = 0)
        indices = [index for index in range(len(dep)) if dep[index] == 1.0]
        if y.any() != 1.0:
            raise TypeError("asy does not satisfy the appropriate constraints, sum")
        for index in indices:
            if np.sum(dep[index]) > 0 and (index >= self.d):
                raise TypeError("asy does not satisfy the appropriate constrains")
        return asy

class Husler_Reiss(Extreme):
    """Class for Hussler_Reiss copula model"""

    copula_type = CopulaTypes.HUSLER_REISS
    theta_interval = [0,float('inf')] 
    invalid_thetas = []

    def _A(self, u):
        """Return the generator function.
        .. math:: A(t) = (1-t)\Phi(\theta + \frac{1}{2\theta}log\frac{1-t}{t}) + t\Phi(\theta + \frac{1}{2\theta}log\frac{t}{1-t}), \quad 0 < t < 1

        Inputs
        ------
            u (list[1-float,float]) : points of the simplexe we evaluate the pickands.
        """
        value_1 = (1-u[1]) * norm.cdf(self.theta + 1/(2*self.theta) * math.log((1-u[1])/u[1])) # u[0] = (1-t), u[1] = t
        value_2 = u[1] * norm.cdf(self.theta + 1/(2*self.theta)*math.log(u[1]/(1-u[1])))
        return value_1 + value_2

    def _Adot(self, u, j):
        """Return the derivative

        Inputs
        ------
            u (list[1-float,float]) : points of the simplexe we evaluate the pickands.
        """
        value_1 = norm.cdf(self.theta + 1 / (2*self.theta) * math.log((1-u[1])/u[1]))
        value_2 = (1/u[1]) * norm.pdf(self.theta + 1 / (2*self.theta) * math.log((1-u[1])/u[1]))
        value_3 = norm.cdf(self.theta + 1/(2*self.theta) * math.log(u[1]/(1-u[1])))
        value_4 = (1/(1-u[1])) * norm.pdf(self.theta + 1/(2*self.theta) * math.log(u[1]/(1-u[1])))
        return - value_1 - value_2 + value_3 + value_4

    def _Sigma2Covar(self, index):
        """ Operation on the covariance matrix to sample from the extremal function.

        Input
        -----
            index : index of the location. An integer in {0, ..., \eqn{d-1}}
        
        """
        Covar = 0.5 * (numpy.matlib.repmat(self.Sigma[:,index],1, self.Sigma.shape[0]) + numpy.matlib.repmat(self.Sigma[index,:], self.Sigma.shape[1], 1) - self.Sigma)
        Covar = np.delete(Covar, index, axis = 0) ; Covar = np.delete(Covar, index, axis = 1)
        #print("covar",Covar)
        return Covar

    def rExtFunc(self,index, cholesky):
        """ Generate from extremal Husler-Reiss distribution \eqn{Y \sim {P_x}}, where
        \eqn{P_{x}} is probability of extremal function
    
        Input
        -----
            index    : index of the location. An integer in {0, ..., \eqn{d-1}}
            Sigma    : a covariance matrix formed from the symmetric square matrix of coefficients \eqn{\lambda^2}
            cholesky : the Cholesky root of \code{Sigma}
    
        Output
        ------
            \code{d}-vector from \eqn{P_x}
    
        https://github.com/lbelzile/mev/blob/main/src/sampling.cpp
        """
        mu = self.Sigma[:,index] / 2
        #print("mu",mu)
        mu = np.delete(mu, index)
        #mu = np.diag(Sigma_tilde) /2
        normalsamp = mvrnorm_chol_arma(1, mu, cholesky)
    
        indexentry = 0
        normalsamp = np.insert(normalsamp, index, indexentry)
        mu = np.insert(mu, index, indexentry)
        samp = np.exp(normalsamp)
        samp[index] = 1.0
        return samp

class Asy_neg_log(Extreme):
    """Class for asymmetric negative logistic copula model."""

    copula_type = CopulaTypes.ASYMMETRIC_NEGATIVE_LOGISTIC
    theta_interval = [1,float('inf')] 
    invalid_thetas = []

    def _A(self, u):
        """Return the Pickands dependence function.
        .. math:: \A(t) = 1-[(\psi_1(1-t))^{-\theta} + (\psi_2t)^{-\theta}]^\frac{1}{\theta}, \quad 0 < t < 1.

         Inputs
        ------
            u (list[1-float,float]) : points of the simplexe we evaluate the pickands.
        """
        value_ = 1-math.pow(math.pow(self.psi1*u[0],-self.theta) + math.pow(self.psi2*u[1],-self.theta), -1/self.theta)
        return value_

    def _Adot(self, u, j = 0):
        """Return the derivative of the Pickands dependence function.
        """
        value_1 = 1/(u[0]*math.pow(self.psi1*u[0], self.theta)) - 1/(u[1]*math.pow(self.psi2*u[1],self.theta))
        value_2 = math.pow(self.psi2*u[1], -self.theta) + math.pow(self.psi1*u[0],-self.theta)
        return value_1*math.pow(value_2,-1/self.theta-1)

class Asy_mix(Extreme):
    """Class for asymmetric mixed model"""

    copula_type = CopulaTypes.ASYMMETRIC_MIXED_MODEL
    theta_interval = [0,float('inf')] 
    invalid_thetas = []

    def _A(self, u):
        """Return the Pickands dependence function.
        .. math:: A(t) = 1-(\theta+\psi_1)*t + \theta*t^2 + psi_1 * t^3, \quad 0 < t < 1.

         Inputs
        ------
            u (list[1-float,float]) : points of the simplexe we evaluate the pickands.
        """
        value_ = 1-(self.theta + self.psi1)*u[1] + self.theta*math.pow(u[1],2) + self.psi1*math.pow(u[1],3)
        return value_

    def _Adot(self, u, j = 0):
        """Return the derivative of the Pickands dependence function.
        """
        value_ =-(self.theta+self.psi1) + 2*self.theta*u[1]+3*self.psi1*math.pow(u[1],2)
        return value_

    def check_parameters(self):
        """
            Validate the parameters inserted.

            This method is used to assert if the parameters are in the valid range for the model.

            Raises :
                ValueError : If theta or psi_1 does not satisfy the constraints.
        """

        if (not self.theta >= 0) or (not self.theta + 3*self.psi1 >=0) or (not self.theta + self.psi1 <= 1) or (self.theta + 2*self.psi1 <= 1):
            message = 'Parameters inserted {}, {} does not satisfy the inequalities for the given {} copula'
            raise ValueError(message.format(self.theta, self.psi1, self.copulaTypes.name))

class tEV(Extreme):
    """Class for t extreme value model"""

    copula_type = CopulaTypes.tEV
    theta_interval = [-1,1] 
    invalid_thetas = []

    def z(self,w):
        value_ = math.pow((1+self.psi1),1/2)*(math.pow(w/(1-w),1/self.psi1) - self.theta)*math.pow(1-math.pow(self.theta,2),-1/2)
        return value_

    def _A(self, u):
        """Return the Pickands dependence function.
        .. math:: A(w) = wt_{\chi+1}(z_w)+(1-w)t_{\chi+1}(z_{1-w}) \quad 0 < w < 1.
        .. math:: z_w  = (1+\chi)^\frac{1}{2}[(w/(1-w))^\frac{1}{\chi} - \rho](1-\rho^2)^\frac{-1}{2}.

         Inputs
        ------
            u (list[1-float,float]) : points of the simplexe we evaluate the pickands.
        """
        value_ = u[1]*t.cdf(self.z(u[1]), df = self.psi1 + 1)+(1-u[1])*t.cdf(self.z(u[0]), df = self.psi1 + 1)
        return value_

    def _Adot(self, u, j=0):
        """Return the derivative of the Pickands dependence function.
        """
        value_1 = t.cdf(self.z(u[1]), df = self.psi1 +1)
        value_2 = (1/u[0]) * t.pdf(self.z(u[1]), df = self.psi1+1)  * math.pow((1+self.psi1),1/2) * math.pow(1-math.pow(self.theta,2),-1/2) * math.pow(u[1]/u[0], 1/self.psi1)
        value_3 = t.cdf(self.z(u[0]), df = self.psi1 + 1)
        value_4 = (1/u[1]) * t.pdf(self.z(u[0]), df = self.psi1 + 1) * math.pow((1+self.psi1),1/2) * math.pow(1-math.pow(self.theta,2),-1/2) * math.pow(u[0]/u[1], 1/self.psi1)
        return  value_1 + value_2 - value_3 - value_4

    def _Sigma2Covar(self, index):
        """ Operation on the covariance matrix to sample from the extremal function.
        Input
        -----
            index : index of the location. An integer in {0, ..., \eqn{d-1}}
        
        """
        Covar = (self.Sigma - np.matrix(self.Sigma[:,index]) @ np.matrix(self.Sigma[index,:])) / (self.psi1 + 1.0)
        Covar = np.delete(Covar, index, axis = 0) ; Covar = np.delete(Covar, index, axis = 1)
        return Covar

    def rExtFunc(self, index, cholesky):
        """ Generate from extremal Student-t probability of extremal function

        Input
        -----
            index    : index of the location. An integer in {0, ..., \eqn{d-1}}
            Sigma    : a positive semi-definite correlation matrix
            cholesky : Cholesky root of transformed correlation matrix
            alpha    : the alpha parameter. Corresponds to degrees of freedom - 1
        
        https://github.com/lbelzile/mev/blob/main/src/sampling.cpp
        """

        zeromean = np.zeros(self.Sigma.shape[1]-1)
        normalsamp = mvrnorm_chol_arma(1, zeromean, cholesky)
        indexentry = 0
        normalsamp = np.insert(normalsamp, index, indexentry)
        nu = np.random.chisquare(self.psi1 + 1.0, size = 1)
        studsamp = np.exp(0.5 * (np.log(self.psi1 + 1.0) - np.log(nu))) * normalsamp + np.squeeze(np.asarray(self.Sigma[:,index]))
        samp = np.power(np.maximum(studsamp,0), self.psi1)
        samp[index] = 1.0
        return samp
    
class Dirichlet(Extreme):
    """ Class for Dirichlet mixmture model introduced by Boldi & Davison (2007) """

    copula_type = CopulaTypes.DIRICHLET

    def _A(self, u):
        raise NotImplementedError

    def _Adot(self, u, j=0):
        raise NotImplementedError

    def rExtFunc(self, index):
        """ Generate from extremal Dirichlet \eqn{Y \sim {P_x}}, where
        \eqn{P_{x}} is the probability of extremal functions from a Dirichlet mixture

        Input
        -----
            d     : dimension of the 1-sample.
            index : index of the location. An integer in {0, ..., \eqn{d-1}}.
            Sigma : a \eqn{d \times n} dimensional vector of positive parameter 
                    values for the Dirichlet vector.
            theta : a \code{m} vector of mixture weights, which sum to 1.

        Output
        ------
            a \code{d}-vector from \eqn{P_x}
        """
        int_seq = np.arange(self.d)
        # Probability weights
        w = np.zeros(len(self.theta))
        for k in range(0, len(self.theta)):
            w[k] = len(self.theta) * self.theta[k] * self.Sigma[index, k] / sum(self.Sigma[:,k])

        m = np.random.choice(int_seq, 1, False, w)[0]

        G = np.zeros(self.d)
        G0 = np.random.gamma(self.Sigma[index,m] + 1.0, 1.0, size = 1)
        for j in range(0,self.d):
            G[j] = np.random.gamma(self.Sigma[j, m], 1.0, size = 1) / G0
        G[index] = 1.0
        return G

class Bilog(Extreme):
    """ The bilogistic distribution model Smith (1990) """
    
    copula_type = CopulaTypes.BILOG

    def _A(self, u):
        raise NotImplementedError

    def _Adot(self, u, j=0):
        raise NotImplementedError

    def rExtFunc(self, index, normalize = True):
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
        alpha_star = np.ones(self.d)
        sample = np.zeros(self.d)
        alpha_star[index] = 1.0 - self.theta[index]
        sample = rdir(1, alpha_star, True)[0,:]
        for i in range(0,self.d):
            sample[i] = np.exp(-self.theta[i] * np.log(sample[i]) + math.lgamma(self.d-self.theta[i]) - math.lgamma(1-self.theta[i]))
        sample = sample / sample[index]
        return sample
