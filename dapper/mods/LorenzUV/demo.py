"""Demonstrate the Lorenz two-speed/scale/layer model."""

import numpy as np
from matplotlib import pyplot as plt

import dapper.mods as modelling
from dapper.mods.LorenzUV.lorenz96 import LUV
from dapper.tools.progressbar import progbar
from dapper.tools.viz import setup_wrapping

nU, J = LUV.nU, LUV.J

dt = 0.005
K  = int(4/dt)

step_1 = modelling.with_rk4(LUV.dxdt, autonom=True)
step_K = modelling.with_recursion(step_1, prog='Simulating')

xx = step_K(LUV.x0, K, np.nan, dt)

# Grab parts of state vector
ii, wrapU = setup_wrapping(nU)
jj, wrapV = setup_wrapping(nU*J)

# Animate linear
plt.figure()
lhU = plt.plot(ii  , wrapU(xx[-1, :nU]), 'b', lw=3)[0]
lhV = plt.plot(jj/J, wrapV(xx[-1, nU:]), 'g', lw=2)[0]
for k in progbar(range(K), 'Plotting'):
    lhU.set_ydata(wrapU(xx[k, :nU]))
    lhV.set_ydata(wrapV(xx[k, nU:]))
    plt.pause(0.001)
