"""Demonstrate the Kuramoto-Sivashinsky (KS) system."""

# The Kuramoto-Sivashinsky (KS) system:
#    u_t = -u*u_x - u_xx - u_xxxx,
#    where x ∈ [0, L],  periodic BCs,
# is the simplest (?) PDE that admits chaos (requires L>=12?):
#
# Its numerical solution is best undertaken
# with Fourier decomposition for the spatial variable.
# According to kassam2005fourth:
# - The equation is stiff, due to higher-order linear terms:
#    - the diffusion term acts as an energy source,
#      causing instability of high-order (large-scale Fourier) modes.
#    - the hyper-diffusion term yields stability of the low-order modes.
# - The nonlinear term induces mixing of the (Fourier) modes.
#
# bocquet2019consistency use it with DA because:
# "it is characterised by sharp density gradients
# so that it may be expected that local EnKFs are prone to imbalance"
#
# hickmann2017multiresolution use it with DA becaues:
# "[The mixing allows us to] investigate the effect of
# propagating scale-dependent information through the EnKF."
#
# www.encyclopediaofmath.org/index.php/Kuramoto-Sivashinsky_equation:
# Number of unstable modes almost directly proportional to L?
#
# Applications:
# - modeling hydrodynamic stability of laminar flame fronts
# - instabilities in thin films and the flow of a viscous fluid down a vertical plane
# - etc
#
# It can be observed in the plots that sharpness from ICs
# remain in the system for a long time (for ever?).

import numpy as np
from matplotlib import pyplot as plt

from dapper.mods.KS import Model
from dapper.tools.viz import amplitude_animation

model = Model()

# Time settings
T = 150
dt = model.dt
K = round(T/dt)

# IC
N     = 3
tt    = np.zeros((K+1,))
EE    = np.zeros((K+1, N, model.Nx))
# x0    = x0_Kassam
EE[0] = model.x0 + 1e-3*np.random.randn(N, model.Nx)

# Integrate
for k in range(1, K+1):
    EE[k] = model.step(EE[k-1], np.nan, dt)
    tt[k] = k*dt


# Animate
ani = amplitude_animation(EE, dt, interval=20)

# Plot
plt.figure()
n = 0
plt.contourf(model.grid, tt, EE[:, n, :], 60)
plt.colorbar()
plt.set_cmap('seismic')
plt.axis('tight')
plt.title('Hovmoller for KS system, member %d' % n)
plt.ylabel('Time (t)')
plt.xlabel('Space (x)')
plt.show()
