"""Demonstrate the Lorenz-05 model."""

from matplotlib import pyplot as plt
from numpy import eye

import dapper.mods as modelling
from dapper.mods.Lorenz05 import Model
from dapper.tools.viz import amplitude_animation

model = Model()
simulator = modelling.with_recursion(model.step, prog="Simulating")

N = 3
M = len(model.x0)
E0 = model.x0 + 1e-2*eye(M)[:N]

dt = 0.004
xx = simulator(E0, k=2000, t=0, dt=dt)

ani = amplitude_animation(xx, dt=dt, interval=10)
plt.show()
