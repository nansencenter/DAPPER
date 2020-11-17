"""For a deeper introduction, see
"DAPPER/tutorials/T4 - Dynamical systems, chaos, Lorenz.ipynb"
"""

from matplotlib import pyplot as plt
import numpy as np

from dapper.mods.Lorenz63 import step, x0
import dapper as dpr

simulator = dpr.with_recursion(step, prog="Simulating")

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
