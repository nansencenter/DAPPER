"""Demonstrate the Vissio-Lucarini-20 model.

Reproduce Hovmoller diagram Fig 4. in `bib.vissio2020mechanics`.
"""
import numpy as np
from matplotlib import pyplot as plt

import dapper.mods as modelling
from dapper.mods.VL20 import model_instance

VL20 = model_instance(nX=36, F=10, G=0)
step = modelling.with_rk4(VL20.dxdt, autonom=True)
simulator = modelling.with_recursion(step, prog="Simulating")

x0 = np.random.rand(72)

dt = 0.05
xx = simulator(x0, k=2200, t0=0, dt=dt)

plt.figure(1)
plt.clf()
plt.contourf(xx[-200:, :], levels=100, cmap='jet')
plt.colorbar()
plt.xticks([0, 9, 19, 29, 36, 45, 55, 65], [1, 10, 20, 30, 1, 10, 20, 30])
plt.yticks(np.arange(0, 220, 20), np.arange(0, 11, 1))
plt.xlabel(r'$X_k, k = 1, ..., 36$'+' '*15+r'$\theta_k, k = 1, ..., 36$')
plt.ylabel('time')
plt.show()
