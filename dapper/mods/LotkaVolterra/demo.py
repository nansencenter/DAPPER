"""Demonstrate the Lotka-Volterra model."""

import numpy as np
from matplotlib import pyplot as plt

import dapper.mods as modelling
from dapper.mods.LotkaVolterra import step, x0

simulator = modelling.with_recursion(step, prog="Simulating")

dt = 0.7
K  = int(1*10**3 / dt)
xx = simulator(x0, K, t0=0, dt=dt)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(np.linspace(0, K*dt, K+1), xx)

plt.show()
