# For a deeper introduction, see
# "DAPPER/tutorials/T4 - Dynamical systems, chaos, Lorenz.ipynb"

from numpy import eye
from dapper import with_recursion, amplitude_animation, plt
plt.ion()

from dapper.mods.Lorenz95.core import step, x0

##
if __name__ == "__main__":
  simulator = with_recursion(step, prog="Simulating")

  x0 = x0(40)
  E0 = x0 + 1e-3*eye(len(x0))[:3]

  dt = 0.05
  xx = simulator(E0, k=500, t=0, dt=dt)

  ani = amplitude_animation(xx,dt=dt,interval=70)

##

