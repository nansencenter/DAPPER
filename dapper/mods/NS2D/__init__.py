"""The Navier-Stokes equations in 2D. Using the Streamfunction form of these equations:
laplacian(psi)_t = psi_y * laplacian(psi)_x - psi_x * laplacian(psi)_y + nu * laplacian(laplacian(psi))

See demo.py for an example of a Taylor-Green vortex"""

import numpy as np
from scipy.differentiate import jacobian
from dapper.dpr_config import DotDict


def Model(N=128, Lxy=2 * np.pi, dt=0.01, nu=0.5, T=1):
    h = dt  # alias -- prevents o/w in step()   
    num_steps = int(T / h)  # Number of time steps

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
        assert dt == h, "dt must match the initialized dt"
        psi = x
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
        return psi_new
    step = step_ETD_RK4  # Should anyone want to implement another step function
    # for _ in range(num_steps):
    #     psi = step(psi, np.nan, h)
    dd = DotDict(
        dt=dt,
        DL = 2,
        step=step,
        dstep_dx=dstep_dx,
        Nx=N,
        x0 = x0,
        T = T
    )
    return dd