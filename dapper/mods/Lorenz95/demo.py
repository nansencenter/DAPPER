# For a deeper introduction, see
# "DAPPER/tutorials/T4 - Dynamical systems, chaos, Lorenz.ipynb"

from dapper import *
from dapper.mods.Lorenz95.core import step


##
def amplitude_animation(xx,periodic=False,skip=1):
  fig, ax = plt.subplots()
  fig.suptitle('Amplitudes')
  ax.set_ylim(*stretch(*xtrema(xx),1.1))
  K,Nx = xx.shape

  ii,wrap = setup_wrapping(Nx,periodic)

  lh, = ax.plot(ii, wrap(xx[0]))
  ax.set_xlim(*xtrema(ii))

  for x in progbar(xx[::skip],"Animating"):
    lh.set_ydata(wrap(x))
    plt.pause(0.01)

  return fig, ax, lh

##
if __name__ == "__main__":
  simulator = with_recursion(step, prog="Simulating")

  x0 = zeros(40); x0[0] = 1
  xx = simulator(x0, k=500, t=0, dt=0.05)

  amplitude_animation(xx,periodic=True)

##

