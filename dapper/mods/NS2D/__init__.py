"""The Navier-Stokes equations in 2D. Using the Streamfunction form of these equations:
laplacian(psi)_t = psi_y * laplacian(psi)_x - psi_x * laplacian(psi)_y + nu * laplacian(laplacian(psi))

See demo.py for an example of a Taylor-Green vortex"""

import numpy as np
from scipy.differentiate import jacobian
from dapper.dpr_config import DotDict
import dapper.tools.liveplotting as LP
import matplotlib as mpl

N = 64
def Model(N=N, Lxy=2 * np.pi, dt=0.002, nu=1/1600, T=100, mp = True):
    def det_jacobian_equation_vec(psi_hat_batch):
        # psi_hat_batch: (N_ens, N, N)
        scale = 1 / (Lxy * Lxy)
        dpsi_dy = np.fft.ifft2(1j * ky * psi_hat_batch, axes=(-2, -1)).real * scale
        domega_dx = np.fft.ifft2(1j * kx * k2 * psi_hat_batch, axes=(-2, -1)).real * scale
        dpsi_dx = np.fft.ifft2(1j * kx * psi_hat_batch, axes=(-2, -1)).real * scale
        domega_dy = np.fft.ifft2(1j * ky * k2 * psi_hat_batch, axes=(-2, -1)).real * scale
        return dpsi_dy * domega_dx - dpsi_dx * domega_dy

    def dealias_vec(f_hat_batch):
        # f_hat_batch: (N_ens, N, N)
        Nf = f_hat_batch.shape[-1]
        k_cutoff = Nf // 3
        bool_mask = np.ones((Nf, Nf), dtype=bool)
        bool_mask[k_cutoff:-k_cutoff, :] = False
        bool_mask[:, k_cutoff:-k_cutoff] = False
        return f_hat_batch * bool_mask  # broadcasting

    def nonlinear_vec(psi_hat):
        # psi_hat: (N_ens, N, N)
        J = det_jacobian_equation_vec(psi_hat)
        J_hat = np.fft.fft2(J, axes=(-2, -1))
        J_hat = dealias_vec(J_hat)
        return -J_hat

    def step_ETD_RK4_vec(X, t, dt):
        # X: (N_ens, N*N) or (N_ens, N, N)
        if X.ndim == 2 and X.shape[1] == N*N:
            X = X.reshape((-1, N, N))
        N_ens = X.shape[0]
        psi_hat = np.fft.fft2(X, axes=(-2, -1))
        omega_hat = k2 * psi_hat
        a = E2 * omega_hat + dt * phi_E2 * nonlinear_vec(psi_hat)
        Na = nonlinear_vec(a)
        b = E2 * omega_hat + dt * phi_E2 * Na
        Nb = nonlinear_vec(b)
        c = E * omega_hat + dt * phi_E * (2 * Nb - nonlinear_vec(psi_hat))
        Nc = nonlinear_vec(c)
        omega_hat_new = E * omega_hat + dt * (phi_E * nonlinear_vec(psi_hat) + 2 * phi_E * (Na + Nb) + phi_E * Nc) / 6
        psi_hat_new = omega_hat_new / k2
        psi_hat_new[:, 0, 0] = 0  # Enforce zero mean for each member
        psi_new = np.fft.ifft2(psi_hat_new, axes=(-2, -1)).real
        return psi_new.reshape((N_ens, N*N))

    def step_1(x, t, dt):
        # x is a 1D flattened state; reshape to 2D for the model
        psi = x.reshape(N, N)
        psi_new = step(psi, t, dt)
        return psi_new.flatten()

    def step_parallel(E, t, dt):
        """Parallelized step for ensemble (2D array) or single state (1D)."""
        if E.ndim == 1:
            return step_1(E, t, dt)
        if E.ndim == 2:
            # Use vectorized version for ensembles
            return step_ETD_RK4_vec(E, t, dt)
    h = dt  # alias -- prevents o/w in step()   

        # Initialize grid and wavenumbers
    x = np.linspace(0, Lxy, N, False)
    y = np.linspace(0, Lxy, N, False)
    X, Y = np.meshgrid(x, y)

        # Wavenumbers
    kk = np.fft.fftfreq(N, Lxy/N) * 2 * np.pi
    kx = kk.reshape((N, 1))
    ky = kk.reshape((1, N))
    k2 = kx**2 + ky**2
    k2[0, 0] = 1  # avoid division by zero

    #initial conditions for a Taylor-Green vortex; change as needed
    psi = np.sin(X) * np.sin(Y)
    x0 = psi.copy()
    L = -nu * k2 #generalize
    E = np.exp(h * L)
    E2 = np.exp(h * L/2)
    def dealias(f_hat):
        N = f_hat.shape[0]
        k_cutoff = N // 3

        bool_mask = np.ones_like(f_hat, dtype=bool)
        bool_mask[k_cutoff:-k_cutoff, :] = False
        bool_mask[:, k_cutoff:-k_cutoff] = False

        dealiased = np.copy(f_hat)
        dealiased[~bool_mask] = 0
        return dealiased
#J = dpsi/dy * domega/dx - dpsi/dx * domega/dy
    def det_jacobian_equation(psi_hat):
        scale = 1 / (Lxy * Lxy)  # Rescale to match the original domain size
        dpsi_dy = np.fft.ifft2(1j * ky * psi_hat).real * scale
        domega_dx = np.fft.ifft2(1j * kx * k2 * psi_hat).real * scale
        dpsi_dx = np.fft.ifft2(1j * kx * psi_hat).real * scale
        domega_dy = np.fft.ifft2(1j * ky * k2 * psi_hat).real * scale

        # print(f"dpsi/dy: {dpsi_dy}\ndomega/dx: {domega_dx}\ndpsi/dx: {dpsi_dx}\ndomega/dy: {domega_dy}\n")
        return dpsi_dy * domega_dx - dpsi_dx * domega_dy
    

#ETD-RK4 method used in the KS model; see dapper/mods/KS/__init__.py
# Based on kursiv.m of Kassam and Trefethen, 2002,
# doi.org/10.1137/S1064827502410633.
# Doesn't use contour integration, but rather a Taylor expansion
# Adapted for the Navier-Stokes equations in 2D.
    def nonlinear(psi_hat):
        J = det_jacobian_equation(psi_hat)
        J_hat = dealias(np.fft.fft2(J))
        return -J_hat
    def phi1(z):
        phi = np.empty_like(z, dtype=np.complex128)
        small = np.abs(z) < 1e-6
        phi[small] = 1 + z[small] / 2 + z[small]**2 / 6 + z[small]**3 / 24 + z[small]**4 / 120
        phi[~small] = (np.exp(z[~small]) - 1)/z[~small]
        return phi
    
    phi_E = phi1(L * h)
    phi_E2 = phi1(L * h / 2)
    def f(psi): #consider redefining with psi_hat
        return np.fft.ifft2(nonlinear(np.fft.fft2(psi)) + nu * k2 * k2 * np.fft.fft2(psi)).real
    def dstep_dx(psi, t, dt):
        return jacobian(f, psi)
    def step_ETD_RK4(x, t, dt):
        """x = psi"""
        """serialize from 1D to 2D to 1D"""
        epsilon = 1e-6
        assert abs(dt-h) < epsilon, "dt must match the initialized dt"
        psi = x
        if psi.shape == (N * N,):
            psi = psi.reshape((N, N))
        psi_hat = np.fft.fft2(psi)
        #omega = laplacian(psi)
        omega_hat = k2 * psi_hat
        a = E2 * omega_hat + dt * phi_E2 * nonlinear(psi_hat)
        Na = nonlinear(a)
        b = E2 * omega_hat + dt * phi_E2 * Na
        Nb = nonlinear(b)
        c = E * omega_hat + dt * phi_E * (2 * Nb - nonlinear(psi_hat))
        Nc = nonlinear(c)
        omega_hat_new = E * omega_hat + dt * (phi_E * nonlinear(psi_hat) + 2 * phi_E * (Na + Nb) + phi_E * Nc) / 6
        psi_hat_new = omega_hat_new / k2 #i^2 = -1; -1 / - 1 = 1.
        psi_hat_new[0, 0] = 0  # Enforce zero mean for
        psi_new = np.fft.ifft2(psi_hat_new).real
        # assert abs(psi_new.mean()) < 1e-9, f"Mean of psi_hat_new is not approximately zero ({psi_new.mean()})"
        # assert abs(psi_hat_new[0,0]) < 1e-9, "Mean of psi_hat_new is not approximately zero"
        return np.clip(psi_new.flatten(), -1, 1)
        # return psi_new.flatten()
    step = step_ETD_RK4  # Single state step
    # for _ in range(num_steps):
    #     psi = step(psi, np.nan, h)
    dd = DotDict(
        dt=dt,
        DL=2,
        step=step_parallel,  # Use the parallelized step function
        dstep_dx=dstep_dx,
        Nx=N,
        x0=x0,
        T=T
    )
    return dd

##Liveplotting mostly copied from QG model


def square(x):
    return x.reshape(N, N)


def ind2sub(ind):
    return np.unravel_index(ind, (N, N))

cm = mpl.cm.viridis
center = N * int(N / 2) + int(0.5 * N)
def LP_setup(jj):
    return [
        (1, LP.spatial2d(square, ind2sub, jj, cm, clims=((-1, 1), (-1, 1), (-1, 1), (-1, 1)))),
        (0, LP.spectral_errors),
        (0, LP.sliding_marginals),
    ]