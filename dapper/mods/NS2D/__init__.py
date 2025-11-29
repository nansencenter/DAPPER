"""The Navier-Stokes equations in 2D. Using the Streamfunction form of these equations:
laplacian(psi)_t = psi_y * laplacian(psi)_x - psi_x * laplacian(psi)_y + nu * laplacian(laplacian(psi))

See demo.py for an example of a Taylor-Green vortex"""

import numpy as np
from scipy.differentiate import jacobian
from dapper.dpr_config import DotDict
import dapper.tools.liveplotting as LP
import matplotlib as mpl

N_global=64 #This is awkward but necessary because otherwise spatial2D would have to be modified. If there is a better way to do this please change
def Model(N=32, Lxy=2 * np.pi, dt=0.002, nu=1/1600, T=100):
    """
    Parameters
    ----------
    N : int
        N = Nx* = Ny* = number of grid points in each dimension.
    Lxy : float
        Domain size in each dimension.
    dt : float
        Time step.
    nu : float
        Kinematic viscosity (dimensionless units); inversely proportional to Reynolds number.
    T : float
        Experiment time; total sim steps = T/dt.

    Returns
    -------
    dd : DotDict
        Model parameters and functions.
    
    Notes
    -----
    See the report pdf in the folder for further explanation of the implementation and mathematics. \n
    *Nx and Ny are not the same as the convention in DAPPER where Nx is the state size and Ny is the observation size; 
    here they just represent N. This is also the convention in the report. More confusingly, sometimes N represents the grid size and sometimes 
    represents ensemble size; each function will delineate what N is in its parameters if present.
    """
    assert N == N_global, f"Warning: change N_global in __init__.py to match your parameter ({N})"
   
    h = dt  # alias -- prevents o/w in step()   

        # Initialize grid and wavenumbers (N = number of grid points in each dimension)
    x = np.linspace(0, Lxy, N, False)
    y = np.linspace(0, Lxy, N, False)
    X, Y = np.meshgrid(x, y)

        # Wavenumbers
    kk = np.fft.fftfreq(N, Lxy/N) * 2 * np.pi
    kx = kk.reshape((N, 1))
    ky = kk.reshape((1, N))
    k2 = kx**2 + ky**2
    k2[0, 0] = 1  # avoid division by zero
    #initial conditions for a Taylor-Green vortex; to use decaying Kolmogorov, comment 56 and uncomment 57-59
    psi = np.sin(X) * np.sin(Y)
    # U = 1
    # k = 2
    # psi = -U/k * np.cos(k*Y)
    x0 = psi.copy()
    L = -nu * k2 #generalize
    L_flat = L.flatten()
    E = np.exp(h * L_flat)
    E2 = np.exp(h * L_flat/2)

    E = E.reshape((N, N))
    E2 = E2.reshape((N, N))
    def dealias(f_hat):
        """Implements the 2/3 rule for dealiasing (Orszag, 1971, https://doi.org/10.1175/1520-0469(1971)028%3C1074:OTEOAI%3E2.0.CO;2)\n
        See section 2.4 of report
        
        Parameters
        ----------
        f_hat : ndarray
            2D array in frequency space.
        
        Returns
        -------
        ndarray
            Dealiased 2D array in frequency space.
        
        """
        N = f_hat.shape[0]
        k_cutoff = N // 3

        k_int = np.fft.fftfreq(N) * N
        mask_1d = np.abs(k_int) < k_cutoff
        mask_2d = np.outer(mask_1d, mask_1d)
        return f_hat * mask_2d  # element-wise multiplication
    
#J = dpsi/dy * domega/dx - dpsi/dx * domega/dy
    def det_jacobian_equation(psi_hat):
        """
        Computes J in the streamfunction formulation of the 2D Navier-Stokes equations \n
        See section 2.1, equation 5 of report
        
        Parameters
        ----------
        psi_hat : ndarray
            2D array in frequency space.
        
        Returns
        -------
        ndarray
            J in physical space.
        """
        scale = 1 / (Lxy * Lxy)  # Rescale to match the original domain size
        dpsi_dy = np.fft.ifft2(1j * ky * psi_hat) * scale
        domega_dx = np.fft.ifft2(1j * kx * k2 * psi_hat) * scale
        dpsi_dx = np.fft.ifft2(1j * kx * psi_hat) * scale
        domega_dy = np.fft.ifft2(1j * ky * k2 * psi_hat) * scale

        # print(f"dpsi/dy: {dpsi_dy}\ndomega/dx: {domega_dx}\ndpsi/dx: {dpsi_dx}\ndomega/dy: {domega_dy}\n")
        return dpsi_dy * domega_dx - dpsi_dx * domega_dy
    

#ETD-RK4 method used in the KS model; see dapper/mods/KS/__init__.py
# Based on kursiv.m of Kassam and Trefethen, 2002,
# doi.org/10.1137/S1064827502410633.

# Adapted for the Navier-Stokes equations in 2D.
    def NL(psi_hat):
        """ Returns the nonlinear term `N` in equation 11 of report
        Parameters
        ----------
        psi_hat : ndarray
            2D array in frequency space.
        
        Returns
        -------
        ndarray
            Nonlinear term in frequency space.
        """
        J = det_jacobian_equation(dealias(psi_hat))
        J_hat = np.fft.fft2(J)
        return -J_hat
    
    # For the step function
    def f(psi): #consider redefining with psi_hat
        return np.fft.ifft2(NL(np.fft.fft2(psi)) + nu * k2 * k2 * np.fft.fft2(psi)).real
    def dstep_dx(psi, t, dt):
        return jacobian(f, psi)
    
    #Contour integral approximation and coefficients
    nRoots = 16
    roots = np.exp(1j * np.pi * (0.5+ np.arange(1, nRoots + 1)) / nRoots)

    CL = h * L_flat[:, None] + roots
    Q = h * ((np.exp(CL / 2) - 1) / CL).mean(axis=-1).real

    f1 = h * ((-4 - CL + np.exp(CL) * (4 - 3 * CL + CL**2) ) / CL**3).mean(axis=-1).real
    f2 = h * ((2 + CL + np.exp(CL) * (-2 + CL)) / CL**3).mean(axis=-1).real
    f3 = h * ((-4 - 3 * CL - CL**2 + np.exp(CL) * (4 - CL)) / CL**3).mean(axis=-1).real

   
    Q = Q.reshape((1, N, N))
    f1 = f1.reshape((1, N, N))
    f2 = f2.reshape((1, N, N))
    f3 = f3.reshape((1, N, N))
    def step_ETD_RK4(x, t, dt):
        """
        Step function using ETD-RK4; x = psi

        Parameters
        ----------
        x : ndarray
            flattened 2D array in physical space.
        t : float
        dt : float
            time step; must match initialized dt.

        Returns
        -------
        ndarray
            flattened 2D array in physical space after time step."""
        epsilon = 1e-6
        assert abs(dt-h) < epsilon, "dt must match the initialized dt"
        
        psi = x.reshape((N, N))
        p_hat = np.fft.fft2(psi)
        #omega = laplacian(psi)
        w_hat = k2 * p_hat
        N1 = NL(p_hat)
        v1 = E2 * w_hat + Q * N1
        N2a = NL(v1)
        v2a = E2 * w_hat + Q * N2a
        N2b = NL(v2a)
        v2b = E2 * v1 + Q * (2 * N2b - N1)
        N3 = NL(v2b)
        omega_hat_new = E * w_hat + N1 * f1 + 2 * (N2a + N2b) * f2 + N3 * f3
        psi_hat_new = omega_hat_new / k2 #i^2 = -1; -1 / - 1 = 1.
        psi_hat_new[0, 0] = 0  # Enforce zero mean for
        psi_new = np.fft.ifft2(psi_hat_new).real
        return psi_new.flatten()

# Vectorized versions of above functions for ensemble computations
    def det_jacobian_equation_vec(psi_hat_batch):
        # psi_hat_batch: (N_ens, N, N)
        scale = 1 / (Lxy * Lxy)
        dpsi_dy = np.fft.ifft2(1j * ky * psi_hat_batch, axes=(-2, -1)) * scale
        domega_dx = np.fft.ifft2(1j * kx * k2 * psi_hat_batch, axes=(-2, -1)) * scale
        dpsi_dx = np.fft.ifft2(1j * kx * psi_hat_batch, axes=(-2, -1)) * scale
        domega_dy = np.fft.ifft2(1j * ky * k2 * psi_hat_batch, axes=(-2, -1)) * scale
        return dpsi_dy * domega_dx - dpsi_dx * domega_dy

    def dealias_vec(f_hat_batch):
        # f_hat_batch: (N_ens, N, N)
        Nf = f_hat_batch.shape[-1]
        k_cutoff = Nf // 3
        k_int = np.fft.fftfreq(Nf) * Nf
        mask_1d = np.abs(k_int) < k_cutoff
        mask_2d = np.outer(mask_1d, mask_1d)
        return f_hat_batch * mask_2d  # broadcasting

    def nonlinear_vec(psi_hat):
        # psi_hat: (N_ens, N, N)
        J = det_jacobian_equation_vec(dealias_vec(psi_hat))
        J_hat = np.fft.fft2(J, axes=(-2, -1))
        return -J_hat

    def step_ETD_RK4_vec(X, t, dt):
        # X: (N_ens, N*N) or (N_ens, N, N)
        if X.ndim == 2 and X.shape[1] == N*N:
            X = X.reshape((-1, N, N))
        N_ens = X.shape[0]
        p_hat = np.fft.fft2(X, axes=(-2, -1))
        #omega = laplacian(psi)
        w_hat = k2 * p_hat
        N1 = nonlinear_vec(p_hat)
        v1 = E2 * w_hat + Q * N1
        N2a = nonlinear_vec(v1)
        v2a = E2 * w_hat + Q * N2a
        N2b = nonlinear_vec(v2a)
        v2b = E2 * v1 + Q * (2 * N2b - N1)
        N3 = nonlinear_vec(v2b)
        omega_hat_new = E * w_hat + N1 * f1 + 2 * (N2a + N2b) * f2 + N3 * f3
        psi_hat_new = omega_hat_new / k2 #i^2 = -1; -1 / - 1 = 1.
        psi_hat_new[0, 0] = 0  # Enforce zero mean for
        psi_new = np.fft.ifft2(psi_hat_new).real
        return psi_new.reshape((N_ens, N*N))

    

    def step_parallel(E, t, dt):
        """Parallelized step for ensemble (2D array) or single state (1D)."""
        if E.ndim == 1:
            return step_ETD_RK4(E, t, dt)
        if E.ndim == 2:
            # Use vectorized version for ensembles
            return step_ETD_RK4_vec(E, t, dt)


    dd = DotDict(
        dt=dt,
        nu=nu,
        DL=2,
        step=step_parallel,  # Use the parallelized step function when possible
        dstep_dx=dstep_dx,
        Nx=N,
        x0=x0,
        T=T
    )
    return dd

##Liveplotting mostly copied from QG model


def square(x):
    return x.reshape(N_global, N_global)


def ind2sub(ind):
    return np.unravel_index(ind, (N_global, N_global))

cm = mpl.cm.viridis
center = N_global * int(N_global / 2) + int(0.5 * N_global)
def LP_setup(jj):
    return [
        (1, LP.spatial2d(square, ind2sub, jj, cm, clims=((-1, 1), (-1, 1), (-1, 1), (-1, 1)), domainlims=(2 * np.pi, 2 * np.pi), periodic=(True, True))),
        (0, LP.spectral_errors),
        (0, LP.sliding_marginals),
    ]