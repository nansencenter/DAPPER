"""Linear advection (i.e. translation) in 1D.

Optimal solution provided by Kalman filter (ExtKF).
System is typically used with a relatively large size (Nx=1000),
but initialized with a moderate wavenumber (k),
which a DA method should hopefully be able to exploit.

The system (and the impact on the DA) can also be adjusted by
selecting non-optimal time steps, and/or various noise configs.

A summary for the purpose of DA is provided in section 3.3 of thesis found at
ora.ox.ac.uk/objects/uuid:9f9961f0-6906-4147-a8a9-ca9f2d0e4a12
"""

import numpy as np
import scipy.linalg as sla
from scipy import sparse


def Fmat(Nx, c, dx, dt):
    """Generate transition matrix.

    - `Nx` - System size
    - `c`  - Velocity of wave. Wave travels to the rigth for `c>0`.
    - `dx` - Grid spacing
    - `dt` - Time step

    Note that the 1st-order upwind scheme used here is exact
    (vis-a-vis the analytic solution) only for `dt = abs(dx/c)`,
    in which case it corresponds to
    `np.roll(x,1,axis=x.ndim-1)`, i.e. circshift in Matlab.
    """
    assert np.abs(c*dt/dx) <= 1, "Must satisfy CFL condition"
    # 1st order explicit upwind scheme
    row1     = np.zeros(Nx)
    row1[-1] = +(np.sign(c)+1)/2
    row1[+1] = -(np.sign(c)-1)/2
    row1[0]  = -1
    L        = sla.circulant(row1)
    F        = np.eye(Nx) + (dt/dx*np.abs(c))*L
    F        = sparse.dia_matrix(F)
    return F


def basis_vector(Nx, k):
    """Generate basis vectors.

    - Nx - state vector length
    - k  - max wavenumber (wavelengths to fit into interval 1:Nx)
    """
    mm = np.arange(1, Nx+1) / Nx
    kk = np.arange(k+1)[:, None]       # Wavenumbers
    aa = np.random.rand(k+1)           # Amplitudes
    pp = np.random.rand(k+1)[:, None]  # Phases

    s  = aa @ np.sin(2*np.pi*(kk * mm + pp))

    # % Normalise
    sd = np.std(s, ddof=1)
    # if Nx >= (2*k + 1)
    # % See analytic_normzt.m
    # sd = np.sqrt(sum(aa(2:end).^2)*(Nx/2)/(Nx-1));
    s  = s/sd

    return s


# Initialization as suggested by sakov'2008 "implications of...",
# (but with some minor differences).
def sinusoidal_sample(Nx, k, N):
    """Generate N basis vectors, and center them.

    The centring is not naturally a part of the basis generation,
    but serves to avoid the initial transitory regime
    if the model is dissipative(, and more ?).

    Example:
    >>> E = sinusoidal_sample(100, 4, 5)
    >>> plt.plot(E.T)  # doctest: +SKIP
    """
    sample = np.zeros((N, Nx))
    for n in range(N):
        sample[n] = basis_vector(Nx, k)

    # Note: Each sample member is centered
    # -- Not the sample as a whole.
    sample = sample - np.mean(sample, axis=1, keepdims=True)

    return sample


def periodic_distance_range(M):
    return np.minimum(np.arange(M), np.arange(M, 0, -1))
    # return np.roll(np.abs(np.arange(M) - M//2), (M+1)//2)
    # return np.concatenate((range((M+1)//2), range(M//2,0,-1)))


# Initialization as suggested by evensen2009
def homogeneous_1D_cov(M, d, kind='Expo'):
    """Generate a covariance matrix for a 1D homogenous random field.

    Generate initial correlations for Linear Advection experiment.

    `d` - decorr length, where the `unit distance = M(i)-M(i-1)` for all `i`.
    """
    row1 = periodic_distance_range(M)

    # If the correlation function is strictly non-negative,
    # the correlation length is often defined as the area under
    # the normailsed correlation function (ie. corr(0) = 1).
    #
    # This can be motivated by looking at the exponential correlation function,
    #         corr(h) = exp(-h/d).
    # The area under the curve (from 0 to infty) equals d, which is also
    # the point where the initial tangent hits the x-axis.
    #
    # For the gaussian correlation function,
    #         corr(h) = exp(-h^2/a^2)
    # the area under the curve equals sqrt(pi*a^2)/2.
    # Thus we should set a^2 = 4/pi*d^2 ~= d^2.

    if kind == 'Gauss':
        # Gaussian covariance
        nugget = 1e-5
        a = 2/np.sqrt(np.pi)*d
        C = nugget*np.eye(M) + (1-nugget)*np.exp(-sla.toeplitz(row1/a)**2)
    elif kind == 'Expo':
        # Exponential covariance
        nugget = 1e-2
        C = nugget*np.eye(M) + (1-nugget)*np.exp(-sla.toeplitz(row1/d))
    else:
        raise KeyError

    return C
