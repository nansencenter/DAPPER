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

__pdoc__ = {"demo": False}

Force = 8.0
Tplot = 10


########################################################################################################################
########################################################################################################################
# Method definitions 
########################################################################################################################
########################################################################################################################
# auxiliary functions for the 2nd order taylor expansion
# these need to be computed once, only as a function of the order of truncation of the fourier series, p

def rho(p):
        return 1.0/12.0 - 0.5 * np.pi**(-2) * np.sum(1.0 / np.arange(1, p+1)**2)

def alpha(p):
        return (np.pi**2) / 180.0 - 0.5 * np.pi**(-2) * np.sum(1.0 / np.arange(1, p+1)**4)


########################################################################################################################
# 2nd order strong taylor SDE step

def l96s_tay2_step(x, **kwargs):
    """One step of integration rule for l96 second order taylor rule

    The rho and alpha are to be computed by the auxiliary functions, depending only on p, and supplied for all steps.
    This is the general formulation which includes, eg. dependence on the truncation of terms in the auxilliary
    function C with respect to the parameter p.  In general, truncation at p=1 is all that is necessary for order
    2.0 convergence, and in this case C below is identically equal to zero.  This auxilliary function can be removed
    (and is removed) in other implementations for simplicity."""

    # Infer system dimension
    sys_dim = len(x)

    dx_params = kwargs['dx_params']
    h = kwargs['h']
    diffusion = kwargs['diffusion']
    dx_dt = kwargs['dx_dt']
    p = kwargs['p']
    rho = kwargs['rho']
    alpha = kwargs['alpha']

    # Compute the deterministic dxdt and the jacobian equations, squeeze at the end
    # to eliminate the extra dimension from ensembles
    dx = np.squeeze(l96(x, dx_params))

    # x is always a single trajectory, we make into a 1 dimension array for the rest of code
    x = np.squeeze(x)
    Jac_x = l96_jacobian(x, dx_params)
    ### random variables

    # Vectors xi, mu, phi are sys_dim X 1 vectors of iid standard normal variables,
    # zeta and eta are sys_dim X p matrices of iid standard normal variables. Functional relationships describe each
    # variable W_j as the transformation of xi_j to be of variace given by the length of the time step h. The functions
    # of random Fourier coefficients a_i, b_i are given in terms mu/ eta and phi/zeta respectively.

    # draw standard normal samples
    rndm = np.random.standard_normal([sys_dim, 2*p + 3])
    xi = rndm[:, 0]

    mu = rndm[:, 1]
    phi = rndm[:, 2]

    zeta = rndm[:, 3: p+3]
    eta = rndm[:, p+3:]

    ### define the auxiliary functions of random fourier coefficients, a and b

    # denominators for the a series
    tmp = np.tile(1 / np.arange(1, p+1), [sys_dim, 1])

    # vector of sums defining a terms
    a = -2 * np.sqrt(h * rho) * mu - np.sqrt(2*h) * np.sum(zeta * tmp, axis=1) / np.pi

    # denominators for the b series
    tmp = np.tile(1 / np.arange(1, p+1)**2, [sys_dim, 1])

    # vector of sums defining b terms
    b = np.sqrt(h * alpha) * phi + np.sqrt(h / (2 * np.pi**2) ) * np.sum(eta * tmp, axis=1)

    # vector of first order Stratonovich integrals
    J_pdelta = (h / 2) * (np.sqrt(h) * xi + a)


    ### auxiliary functions for higher order stratonovich integrals ###

    # the triple stratonovich integral reduces in the lorenz 96 equation to a simple sum of the auxiliary functions, we
    # define these terms here abstractly so that we may efficiently compute the terms
    def C(l, j):
        C = np.zeros([p, p])
        # we will define the coefficient as a sum of matrix entries where r and k do not agree --- we compute this by a
        # set difference
        indx = set(range(1, p+1))

        for r in range(1, p+1):
            # vals are all values not equal to r
            vals = indx.difference([r])
            for k in vals:
                # and for row r, we define all columns to be given by the following, inexing starting at zero
                C[r-1, k-1] = (r / (r**2 - k**2)) * ((1/k) * zeta[l, r-1] * zeta[j, k-1] + (1/r) * eta[l, r-1] * eta[j, k-1] )

        # we return the sum of all values scaled by -1/2pi^2
        return -.5 * np.pi**(-2) * np.sum(C)

    def Psi(l, j):
        # psi will be a generic function of the indicies l and j, we will define psi plus and psi minus via this
        psi = h**2 * xi[l] * xi[j] / 3 + h * a[l] * a[j] / 2 + h**(1.5) * (xi[l] * a[j] + xi[j] * a[l]) / 4 \
              - h**(1.5) * (xi[l] * b[j] + xi[j] * b[l]) / (2 * np.pi) - h**2 * (C(l,j) + C(j,l))
        return psi

    # we define the approximations of the second order Stratonovich integral
    psi_plus = np.array([Psi((i-1) % sys_dim, (i+1) % sys_dim) for i in range(sys_dim)])
    psi_minus = np.array([Psi((i-2) % sys_dim, (i-1) % sys_dim) for i in range(sys_dim)])

    # the final vectorized step forward is given as
    x_step = x + dx * h + h**2 * .5 * Jac_x @ dx    # deterministic taylor step
    x_step += diffusion * np.sqrt(h) * xi           # stochastic euler step
    x_step += + diffusion * Jac_x @ J_pdelta        # stochastic first order taylor step
    x_step += diffusion**2 * (psi_plus - psi_minus) # stochastic second order taylor step

    return np.reshape(x_step, [sys_dim, 1])

########################################################################################################################


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
