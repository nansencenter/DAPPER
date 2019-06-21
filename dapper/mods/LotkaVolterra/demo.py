
# For a deeper introduction, see
# "DAPPER/tutorials/T4 - Dynamical systems, chaos, Lorenz.ipynb"
##

from dapper import *
from dapper.mods.LotkaVolterra.core import step, x0

##
simulator = with_recursion(step, prog="Simulating")

dt = 0.7
K  = int(1*10**4 / dt)
xx = simulator(x0, K, t0=0, dt=dt)

##
fig, ax = freshfig(2,(9,6))
ax.plot(linspace(0,K*dt,K+1),xx)

##
