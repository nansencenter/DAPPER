"""Demonstrate the Ikeda map."""
##
from dapper import *
import core

## Various parameter cases
case = "c3"
if case == "c1":
    core.u = 0.918
    s = 5
    N = 2000
    as_points = False
elif case == "c2":
    core.u = 0.99
    s = 1
    N = 200
    as_points = True
elif case == "c3":
    core.u = 0.9
    s = 0
    N = 1
    as_points = True

## Computations
simulator = with_recursion(core.step, prog="Simulating")

# Initial ensemble
E0 = core.x0 + s*randn((N,2))

# Simulate
K = 10**5 // N
EE = simulator(E0, K, 0, 1)

##
# fig, ax = plt.subplots()
plt.ion()
fig, ax = freshfig(1)
fig.suptitle('Phase space')
[eval("ax.set_%slabel('%s')"%(s,s)) for s in "xy"]

# Length (from ending) of trajectories to plot
L = 0 # if 0: include all

# Re-order axes for plotting
EE = EE[-L:].transpose((2,0,1))

if as_points: ax.plot(*EE, "b.", ms=1.0, alpha=1.0)
else:         ax.plot(*EE, "b-", lw=.02, alpha=0.1)

# Plot start/end-ing points
# ax.plot(*EE[0] .T, '*g', ms=14)
# ax.plot(*EE[-1].T, '*r', ms=14)


##

