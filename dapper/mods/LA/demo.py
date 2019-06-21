# 


from dapper import *
from dapper.mods.LA.raanes2015 import step, X0
from dapper.mods.Lorenz95.demo import amplitude_animation

##
simulator = with_recursion(step, prog="Simulating")

x0 = X0.sample(1).squeeze()
xx = simulator(x0, k=500, t=0, dt=1)

##
amplitude_animation(xx,periodic=True,skip=3)

##


