########################################################################################################################
########################################################################################################################
"""A 1D emulator of chaotic atmospheric behaviour, including additive noise.

`bib.grudzien2020numerical`
"""


########################################################################################################################
########################################################################################################################
# Imports
import numpy as np
from .extras import LPs, d2x_dtdx, dstep_dx


########################################################################################################################
########################################################################################################################
# Method definitions 
########################################################################################################################
########################################################################################################################
# basic Lorenz 96 defintions and functions

__pdoc__ = {"demo": False}

# energy injected into the system
Force = 8.0

# diffusion in the SDE process
s = 0.05

# ?
Tplot = 10

def x0(M):
    return np.eye(M)[0]


def shift(x, n):
    return np.roll(x, -n, axis=-1)


def dxdt_autonomous(x):
    return (shift(x, 1)-shift(x, -2))*shift(x, -1) - x


def dxdt(x):
    return dxdt_autonomous(x) + Force


def step(x0, t, dt):
    return rk4(lambda t, x: dxdt(x), x0, np.nan, dt)

########################################################################################################################
# Lorenz 96 Jacobian function

def jacobian(x):
    """"This computes the Jacobian of the Lorenz 96, for arbitrary dimension, equation about the point x."""

    x_dim = len(x)

    dxF = np.zeros([x_dim, x_dim])

    for i in range(x_dim):
        i_m_2 = np.mod(i - 2, x_dim)
        i_m_1 = np.mod(i - 1, x_dim)
        i_p_1 = np.mod(i + 1, x_dim)

        dxF[i, i_m_2] = -x[i_m_1]
        dxF[i, i_m_1] = x[i_p_1] - x[i_m_2]
        dxF[i, i] = -1.0
        dxF[i, i_p_1] = x[i_m_1]

    return dxF


########################################################################################################################
# 2nd order strong taylor SDE step

def l96s_tay2_step(x, t, dt):
    """Steps forward state of L96s model by order 2.0 Taylor scheme

    This is the basic formulation which makes a Fourier truncation at p=1 for the simple form of the order 2.0 method."""

    # Infer system dimension
    sys_dim = len(x)

    # Compute the deterministic dxdt and the jacobian equations
    dx = dxdt(x) 
    dxF = jacobian(x)

    # coefficients defined based on the p=1 Fourier truncation
    rho = 1.0/12.0 - 0.5 * np.pi**(-2)
    alpha =(np.pi**2) / 180.0 - 0.5 * np.pi**(-2) 

    ### random variables
    # Vectors xi, mu, phi are sys_dim vectors of iid standard normal variables,
    # zeta and eta are sys_dim vectors of iid standard normal variables. Functional relationships describe each
    # variable W_j as the transformation of xi_j to be of variance given by the length of the time step dt. 
    # The functions of random Fourier coefficients a_i, b_i are given in terms mu/ eta and phi/zeta respectively.

    # draw standard normal samples
    rndm = np.random.standard_normal([sys_dim, 5])
    xi = rndm[:, 0]

    mu = rndm[:, 1]
    phi = rndm[:, 2]

    zeta = rndm[:, 3]
    eta = rndm[:, 4]

    ### define the auxiliary functions of random Fourier coefficients, a and b
    # vector of "a" terms
    a = -2.0 * np.sqrt(dt * rho) * mu - np.sqrt(2.0*dt) * zeta  / np.pi

    # vector of "b" terms
    b = np.sqrt(dt * alpha) * phi + np.sqrt(dt / (2.0 * np.pi**2) ) * eta 

    # vector of first order Stratonovich integrals
    J_pdelta = (dt / 2.0) * (np.sqrt(dt) * xi + a)

    def Psi(l, j):
        # psi will be a generic function of the indicies l and j, we will define psi plus and psi minus via this
        psi = dt**2 * xi[l] * xi[j] / 3.0 + dt * a[l] * a[j] / 2.0 + dt**(1.5) * (xi[l] * a[j] + xi[j] * a[l]) / 4.0 \
              - dt**(1.5) * (xi[l] * b[j] + xi[j] * b[l]) / (2.0 * np.pi) 
        return psi

    # we define the approximations of the second order Stratonovich integral
    psi_plus = np.array([Psi((i-1) % sys_dim, (i+1) % sys_dim) for i in range(sys_dim)])
    psi_minus = np.array([Psi((i-2) % sys_dim, (i-1) % sys_dim) for i in range(sys_dim)])

    # the final vectorized step forward is given as
    x  = x + dx * dt + dt**2 * 0.5 * dxF @ dx # deterministic taylor step
    x += s * np.sqrt(dt) * xi                 # stochastic euler step
    x += s * dxF @ J_pdelta                   # stochastic first order taylor step
    x += s**2 * (psi_plus - psi_minus)        # stochastic second order taylor step

    return x


########################################################################################################################


