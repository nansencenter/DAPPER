"""The Navier-Stokes equations in 2D. Using the Streamfunction form of these equations:
laplacian(psi)_t = psi_y * laplacian(psi)_x - psi_x * laplacian(psi)_y + nu * laplacian(laplacian(psi))

See demo.py for an example of a Taylor-Green vortex"""

import numpy as np
from dapper.dpr_config import DotDict

def Solver(N=128, Lxy=2 * np.pi, dt=0.001, nu=0.1, T=5):
    h = dt  # alias -- prevents o/w in step()   
    num_steps = int(T / h)  # Number of time steps

        # Initialize grid and wavenumbers
    xx = np.linspace(0, Lxy, N, False)
    yy = np.linspace(0, Lxy, N, False)
    XX, YY = np.meshgrid(xx, yy)

        # Wavenumbers
    kk = np.fft.fftfreq(N, Lxy/N) * 2 * np.pi
    kx = kk.reshape((N, 1))
    ky = kk.reshape((1, N))
    k2 = kx**2 + ky**2
    k2[0, 0] = 1  # avoid division by zero

    #initial conditions for a Taylor-Green vortex; change as needed
    psi = np.sin(XX) * np.sin(YY)
    psi_hat = np.fft.fft2(psi)
    omega = 2 * np.sin(XX) * np.sin(YY)
    omega_hat = np.fft.fft2(omega)

    L = nu * k2 #generalize
    E = np.exp(h * L)
    E2 = np.exp(h * L/2)
    def dealias(self, f_hat):
        N = f_hat.shape[0]
        k_cutoff = N // 3

        bool_mask = np.ones_like(f_hat, dtype=bool)
        bool_mask[k_cutoff:-k_cutoff, :] = False
        bool_mask[:, k_cutoff:-k_cutoff] = False

        dealiased = np.copy(f_hat)
        dealiased[~bool_mask] = 0
        return dealiased
#J = dpsi/dy * domega/dx - dpsi/dx * domega/dy
    def det_jacobian_equation(self, psi_hat):
        scale = 1 / (self.Lxy * self.Lxy)  # Rescale to match the original domain size
        dpsi_dy = np.fft.ifft2(1j * self.ky * psi_hat).real * scale
        domega_dx = np.fft.ifft2(1j * self.kx * self.k2 * psi_hat).real * scale
        dpsi_dx = np.fft.ifft2(1j * self.kx * psi_hat).real * scale
        domega_dy = np.fft.ifft2(1j * self.ky * self.k2 * psi_hat).real * scale

        # print(f"dpsi/dy: {dpsi_dy}\ndomega/dx: {domega_dx}\ndpsi/dx: {dpsi_dx}\ndomega/dy: {domega_dy}\n")
        return dpsi_dy * domega_dx - dpsi_dx * domega_dy

    def d2psi_dtdpsi(self, psi_hat):
        dL = np.fft.ifft2(L * np.fft.fft2(np.eye(N))).T
    
#ETD-RK4 method used in the KS model; see dapper/mods/KS/__init__.py
# Based on kursiv.m of Kassam and Trefethen, 2002,
# doi.org/10.1137/S1064827502410633.
# Doesn't use contour integration, but rather a Taylor expansion
# Adapted for the Navier-Stokes equations in 2D.
    def nonlinear(self, psi_hat):
        J = self.det_jacobian_equation(psi_hat)
        J_hat = self.dealias(np.fft.fft2(J))
        return -J_hat
    def phi1(self, z):
        phi = np.empty_like(z, dtype=np.complex128)
        small = np.abs(z) < 1e-6
        phi[small] = 1 + z[small] / 2 + z[small]**2 / 6 + z[small]**3 / 24 + z[small]**4 / 120
        phi[~small] = (np.exp(z[~small]) - 1)/z[~small]
        return phi
    
    phi_E = phi1(L * h)
    phi_E2 = phi1(L * h / 2)

    def step_ETD_RK4(psi, psi_hat, omega, omega_hat, dt):
        assert dt == h ("dt must match the initialized dt")
        a = E2 * omega_hat + dt * phi_E2 * nonlinear(psi_hat)
        Na = nonlinear(a)
        b = E2 * omega_hat + dt * phi_E2 * Na
        Nb = nonlinear(b)
        c = E * omega_hat + dt * phi_E * (2 * Nb - nonlinear(psi_hat))
        Nc = nonlinear(c)
        omega_hat_new = E * omega_hat + dt * (phi_E * nonlinear(psi_hat) + 2 * phi_E * (Na + Nb) + phi_E * Nc) / 6
        omega_new = np.fft.ifft2(omega_hat_new).real #/ (Lxy * Lxy)  # Rescale to match the original domain size
        psi_hat_new = omega_hat_new / k2 #i^2 = -1; -1 / - 1 = 1.
        psi_hat_new[0, 0] = 0  # Enforce zero mean for
        psi_new = np.fft.ifft2(psi_hat_new).real
        return psi_new, psi_hat_new, omega_hat_new, omega_new
    step = step_ETD_RK4  # Should anyone want to implement another step function
    for i in range(num_steps):
        psi, psi_hat, omega_hat, omega = step(psi, psi_hat, omega, omega_hat, h)
    dd = DotDict(

    )