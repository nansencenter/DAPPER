"""Demonstrate the Linear Advection (LA) model."""
from matplotlib import pyplot as plt

import dapper.mods as modelling
from dapper.mods.LA.raanes2015 import X0, step
from dapper.tools.viz import amplitude_animation

simulator = modelling.with_recursion(step, prog="Simulating")

x0 = X0.sample(1).squeeze()
dt = 1
xx = simulator(x0, k=500, t=0, dt=dt)

anim = amplitude_animation(xx, dt)
plt.show()
