"""Classes of random variables."""

import numpy as np
import numpy.random as rnd
from numpy import sqrt
from struct_tools import NicePrint

from dapper.tools.matrices import CovMat


class RV(NicePrint):
    """Class to represent random variables."""

    printopts = NicePrint.printopts.copy()
    printopts["ordering"] = "linenumber"
    printopts["reverse"] = True

    def __init__(self, M, **kwargs):
        """Initalization arguments:

        - M    <int>     : ndim
        - is0  <bool>    : if True, the random variable is identically 0
        - func <func(N)> : use this sampling function. Example:
                           `RV(M=4,func=lambda N: rand(N,4)`
        - file <str>     : draw from file. Example:
                           `RV(M=4,file=dpr.rc.dirs.data/'tmp.npz')`

        The following kwords (versions) are available,
        but should not be used for anything serious
        (use instead subclasses, like `GaussRV`).

        - icdf <func(x)> : marginal/independent  "inverse transform" sampling.
                           Example: `RV(M=4,icdf = scipy.stats.norm.ppf)`
        - cdf <func(x)>  : as icdf, but with approximate icdf, from interpolation.
                           Example: `RV(M=4,cdf = scipy.stats.norm.cdf)`
        - pdf  <func(x)> : "acceptance-rejection" sampling. Not implemented.
        """
        self.M = M
        for key, value in kwargs.items():
            setattr(self, key, value)

    def sample(self, N):
        if getattr(self, 'is0', False):
            # Identically 0
            E = np.zeros((N, self.M))
        elif hasattr(self, 'func'):
            # Provided by function
            E = self.func(N)
        elif hasattr(self, 'file'):
            # Provided by numpy file with sample
            data   = np.load(self.file)
            sample = data['sample']
            N0     = len(sample)
            if 'w' in data:
                w = data['w']
            else:
                w = np.ones(N0)/N0
            idx = rnd.choice(N0, N, replace=True, p=w)
            E   = sample[idx]
        elif hasattr(self, 'icdf'):
            # Independent "inverse transform" sampling
            icdf = np.vectorize(self.icdf)
            uu   = rnd.rand(N, self.M)
            E    = icdf(uu)
        elif hasattr(self, 'cdf'):
            # Like above, but with inv-cdf approximate, from interpolation
            if not hasattr(self, 'icdf_interp'):
                # Define inverse-cdf
                from scipy.interpolate import interp1d
                from scipy.optimize import fsolve
                cdf    = self.cdf
                Left,  = fsolve(lambda x: cdf(x) - 1e-9, 0.1)  # noqa
                Right, = fsolve(lambda x: cdf(x) - (1-1e-9), 0.1)  # noqa
                xx     = np.linspace(Left, Right, 1001)
                uu     = np.vectorize(cdf)(xx)
                icdf   = interp1d(uu, xx)
                self.icdf_interp = np.vectorize(icdf)
            uu = rnd.rand(N, self.M)
            E  = self.icdf_interp(uu)
        elif hasattr(self, 'pdf'):
            # "acceptance-rejection" sampling
            raise NotImplementedError
        else:
            raise KeyError
        assert self.M == E.shape[1]
        return E


# TODO 4: improve constructor (treatment of arg cases is too fragile).
class RV_with_mean_and_cov(RV):
    """Generic multivariate random variable characterized by mean and cov.

    This class must be subclassed to provide sample(),
    i.e. its main purpose is provide a common convenience constructor.
    """

    def __init__(self, mu=0, C=0, M=None):
        """Init allowing for shortcut notation."""
        if isinstance(mu, CovMat):
            raise TypeError("Got a covariance paramter as mu. "
                            + "Use kword syntax (C=...) ?")

        # Set mu
        mu = np.atleast_1d(mu)
        assert mu.ndim == 1
        if len(mu) > 1:
            if M is None:
                M = len(mu)
            else:
                assert len(mu) == M
        else:
            if M is not None:
                mu = np.ones(M)*mu

        # Set C
        if isinstance(C, CovMat):
            if M is None:
                M = C.M
        else:
            if np.isscalar(C) and C == 0:
                pass  # Assign as pure 0!
            else:
                if np.isscalar(C):
                    M = len(mu)
                    C = CovMat(C*np.ones(M), 'diag')
                else:
                    C = CovMat(C)
                    if M is None:
                        M = C.M

        # Validation
        if len(mu) not in (1, M):
            raise TypeError("Inconsistent shapes of (M,mu,C)")
        if M is None:
            raise TypeError("Could not deduce the value of M")
        try:
            if M != C.M:
                raise TypeError("Inconsistent shapes of (M,mu,C)")
        except AttributeError:
            pass

        # Assign
        self.M  = M
        self.mu = mu
        self.C  = C

    def sample(self, N):
        """Sample N realizations. Returns N-by-M (ndim) sample matrix.

        Example
        -------
        >>> plt.scatter(*(UniRV(C=randcov(2)).sample(10**4).T))  # doctest: +SKIP
        """
        if self.C == 0:
            D = np.zeros((N, self.M))
        else:
            D = self._sample(N)
        return self.mu + D

    def _sample(self, N):
        raise NotImplementedError("Must be implemented in subclass")


class GaussRV(RV_with_mean_and_cov):
    """Gaussian (Normal) multivariate random variable."""

    def _sample(self, N):
        R = self.C.Right
        D = rnd.randn(N, len(R)) @ R
        return D


class LaplaceRV(RV_with_mean_and_cov):
    """Laplace (double exponential) multivariate random variable.

    This is an elliptical generalization. Ref:
    Eltoft (2006) "On the Multivariate Laplace Distribution".
    """

    def _sample(self, N):
        R = self.C.Right
        z = rnd.exponential(1, N)
        D = rnd.randn(N, len(R))
        D = z[:, None]*D
        return D @ R / sqrt(2)


class LaplaceParallelRV(RV_with_mean_and_cov):
    """A NON-elliptical multivariate version of Laplace (double exponential) RV."""

    def _sample(self, N):
        # R = self.C.Right   # contour: sheared rectangle
        R = self.C.sym_sqrt  # contour: rotated rectangle
        D = rnd.laplace(0, 1, (N, len(R)))
        return D @ R / sqrt(2)


class StudRV(RV_with_mean_and_cov):
    """Student-t multivariate random variable.

    Assumes the covariance exists,
    which requires degreee-of-freedom (dof) > 1+ndim.
    Also requires that dof be integer,
    since chi2 is sampled via Gaussians.
    """

    def __init__(self, dof, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dof = dof

    def _sample(self, N):
        R = self.C.Right
        nu = self.dof
        r = nu/np.sum(rnd.randn(N, nu)**2, axis=1)  # InvChi2
        D = sqrt(r)[:, None]*rnd.randn(N, len(R))
        return D @ R * sqrt((nu-2)/nu)


class UniRV(RV_with_mean_and_cov):
    """Uniform multivariate random variable.

    Has an elliptic-shape support.
    Ref: Voelker et al. (2017) "Efficiently sampling
    vectors and coordinates from the n-sphere and n-ball"
    """

    def _sample(self, N):
        R = self.C.Right
        D = rnd.randn(N, len(R))
        r = rnd.rand(N)**(1/len(R)) / np.sqrt(np.sum(D**2, axis=1))
        D = r[:, None]*D
        return D @ R * 2


class UniParallelRV(RV_with_mean_and_cov):
    """Uniform multivariate random variable.

    Has a parallelogram-shaped support, as determined by the cholesky factor
    applied to the (corners of) the hypercube.
    """

    def _sample(self, N):
        R = self.C.Right
        D = rnd.rand(N, len(R))-0.5
        return D @ R * sqrt(12)
