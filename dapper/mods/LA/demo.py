"""Demonstrate the Linear Advection (LA) model."""
from dapper import with_recursion
from matplotlib import pyplot as plt
from dapper.mods.LA.raanes2015 import step, X0
from dapper.tools.viz import amplitude_animation

simulator = with_recursion(step, prog="Simulating")

x0 = X0.sample(1).squeeze()
dt = 1
xx = simulator(x0, k=500, t=0, dt=dt)

anim = amplitude_animation(xx,dt)
plt.show()
