"""Demonstrate the Lorenz-63 model.

For a deeper introduction, see

"DA-tutorials/T4 - Dynamical systems, chaos, Lorenz.ipynb"
"""

import numpy as np
from matplotlib import pyplot as plt

import dapper.mods as modelling
from dapper.mods.Lorenz63 import step, x0

simulator = modelling.with_recursion(step, prog="Simulating")

xx = simulator(x0, k=5*10**3, t0=0, dt=0.01)

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot(*xx.T, lw=1)
ax.plot(*np.atleast_2d(x0    ).T, '*g', ms=14) # noqa
ax.plot(*np.atleast_2d(xx[-1]).T, '*r', ms=14)

fig.suptitle('Phase space evolution')
ax.set_facecolor('w')
for s in "xyz":
    eval(f"ax.set_{s}label('{s}')")
plt.show()
