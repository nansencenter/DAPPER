"""Kuramoto-Sivashinsky (KS) system: the simplest (?) PDE admitting chaos.

Defined by:

    u_t = -u*u_x - u_xx - u_xxxx

- See compare_schemes.py for a comparison of time-step integration schemes.
- See demo.py for further description.
"""

import functools

import numpy as np

from dapper.dpr_config import DotDict
from dapper.mods.integration import integrate_TLM, with_rk4


# To & from time/Fourier domain -- use reals-only fft
def fft(u): return np. fft. rfft(u, axis=-1)  # F
def ifft(v): return np.fft.irfft(v, axis=-1)  # F^{-1}


# Do fft to/from Fourier-domain for wrapped function.
def byFourier(func):
    @functools.wraps(func)
    def newfunc(u, *args, **kwargs):
        return ifft(func(fft(u), *args, **kwargs))
    return newfunc


def Model(dt=0.25, DL=32, Nx=128):
    """Define `step`, `x0`, `etc`. Alternative schemes (`step_XXX`) also implemented."""
    h = dt  # alias -- prevents o/w in step()

    # Fourier stuff
    # wave nums for rfft
    kk = np.append(np.arange(0, Nx/2), 0)*2/DL
    # wave nums for fft
    # kk = ccat([np.arange(0,Nx/2),[0], np.arange(-Nx/2+1,0)])*2/DL
    # Alternative method:
    # kk = np.fft.fftfreq(Nx, DL/Nx/2)
    # Operators
    D = 1j*kk          # Differentiation to compute: F[ u_x ]
    L = kk**2 - kk**4  # Linear operator for KS eqn: F[ - u_xx - u_xxxx]

    # NonLinear term (-u*u_x) in Fourier domain via time domain
    def NL(v):
        return -0.5 * D * fft(ifft(v).real ** 2)

    # Evolution equation
    @byFourier
    def dxdt(v):
        return NL(v) + L*v

    # Jacobian of dxdt(u)
    def d2x_dtdx(u):
        dL  = ifft(L * fft(np.eye(Nx))) . T
        dNL = - ifft(D * fft(np.diag(u))) . T
        return dL + dNL

    # dstep_dx = FD_Jac(step)
    def dstep_dx(x, t, dt):
        return integrate_TLM(d2x_dtdx(x), dt, method='analytic')

    # Runge-Kutta -- Requries dt<1e-2:
    # ------------------------------------------------
    step_RK4 = with_rk4(dxdt, autonom=True)           # Bad, not recommended.
    step_RK1 = with_rk4(dxdt, autonom=True, stages=1)  # Truly terrible.

    # "Semi-implicit RK3": explicit RK3 for nonlinear term,
    # ------------------------------------------------
    # implicit trapezoidal adjustment for linear term.
    # Based on github.com/jswhit/pyks (Jeff Whitaker),
    # who got it from doi.org/10.1175/MWR3214.1.
    @byFourier
    def step_SI_RK3(v, t, dt):
        v0 = v.copy()
        for n in range(3):
            dt3 = h/(3-n)
            v = v0 + dt3*NL(v)
            v = (v + 0.5*L*dt3*v0)/(1 - 0.5*L*dt3)
        return v

    # ETD-RK4 -- Integration factor (IF) technique, mixed with RK4.
    # ------------------------------------------------
    # Based on kursiv.m of Kassam and Trefethen, 2002,
    # doi.org/10.1137/S1064827502410633.
    #
    # Precompute ETDRK4 scalar quantities
    E  = np.exp(h*L)       # Integrating factor, eval at dt
    E2 = np.exp(h*L/2)     # Integrating factor, eval at dt/2
    # Roots of unity are used to discretize a circular countour...
    nRoots = 16
    roots = np.exp(1j * np.pi * (0.5+np.arange(nRoots))/nRoots)
    # ... the associated integral then reduces to the mean,
    # g(CL).mean(axis=-1) ~= g(L), whose computation is more stable.
    CL = h * L[:, None] + roots  # Contour for (each element of) L
    # E * exact_integral of integrating factor:
    Q  = h * ((np.exp(CL/2)-1)         / CL).mean(axis=-1).real
    # RK4 coefficients (modified by Cox-Matthews):
    f1 = h * ((-4-CL + np.exp(CL)*(4-3*CL+CL**2)) / CL**3).mean(axis=-1).real
    f2 = h * ((2+CL  + np.exp(CL)*(-2+CL))        / CL**3).mean(axis=-1).real
    f3 = h * ((-4-3*CL-CL**2+np.exp(CL)*(4-CL))   / CL**3).mean(axis=-1).real
    #

    @byFourier
    def step_ETD_RK4(v, t, dt):
        assert dt == h, \
            "Model is instantiated with a pre-set dt, " +\
            "which differs from the requested value"
        N1  = NL(v)
        v1  = E2*v  + Q*N1
        N2a = NL(v1)
        v2a = E2*v  + Q*N2a
        N2b = NL(v2a)
        v2b = E2*v1 + Q*(2*N2b-N1)
        N3  = NL(v2b)
        v   = E * v  + N1*f1 + 2*(N2a+N2b)*f2 + N3*f3
        return v

    # Select the "official" step method
    step = step_ETD_RK4

    # Generate IC as end-point of ex. from Kassam and Trefethen.
    # x0_Kassam isn't convenient, coz prefer {x0 ∈ attractor} to {x0 ∈ basin}.
    grid = DL*np.pi*np.linspace(0, 1, Nx+1)[1:]
    x0_Kassam = np.cos(grid/16) * (1 + np.sin(grid/16))
    x0 = x0_Kassam.copy()
    for _ in range(int(150/h)):
        x0 = step(x0, np.nan, h)

    # Return dict
    dd = DotDict(dt=dt,
                 DL=DL,
                 Nx=Nx,
                 x0=x0,
                 x0_Kassam=x0_Kassam,
                 grid=grid,
                 step=step,
                 step_ETD_RK4=step_ETD_RK4,
                 step_SI_RK3=step_SI_RK3,
                 step_RK4=step_RK4,
                 step_RK1=step_RK1,
                 dxdt=dxdt,
                 d2x_dtdx=d2x_dtdx,
                 dstep_dx=dstep_dx,
                 )
    return dd


Tplot = 10
