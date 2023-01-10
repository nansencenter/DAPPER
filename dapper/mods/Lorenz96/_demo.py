"""Demonstrate the Lorenz-96 model."""

# For a deeper introduction, see
# "DA-tutorials/T4 - Dynamical systems, chaos, Lorenz.ipynb"

from matplotlib import pyplot as plt
from numpy import eye

import dapper.mods as modelling
from dapper.mods.Lorenz96 import step, x0
from dapper.tools.viz import amplitude_animation

simulator = modelling.with_recursion(step, prog="Simulating")

M = 40
N = 3

x0 = x0(M)
E0 = x0 + 1e-3*eye(M)[:N]

dt = 0.05
xx = simulator(E0, k=500, t=0, dt=dt)

ani = amplitude_animation(xx, dt=dt, interval=70)

plt.show()
