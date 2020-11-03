"""Demonstrate the Lotka-Volterra model."""

from dapper import *
import dapper as dpr
from matplotlib import pyplot as plt
from dapper.mods.LotkaVolterra.core import step, x0
from dapper import with_recursion

simulator = with_recursion(step, prog="Simulating")

dt = 0.7
K  = int(1*10**3 / dt)
xx = simulator(x0, K, t0=0, dt=dt)

fig, ax = dpr.freshfig(2,(9,6))
ax.plot(np.linspace(0,K*dt,K+1),xx)

plt.show()
