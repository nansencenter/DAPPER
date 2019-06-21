# https://en.wikipedia.org/wiki/Double_pendulum

from dapper import *
from dapper.mods.DoublePendulum.core import step, x0, energy, L1, L2

##
E0 = x0 + 0.01*randn((3,4))
simulator = with_recursion(step)
dt = 0.01
EE = simulator(E0, k=10**4, t0=0, dt=dt)

# Energy evolution. Should be constant, but for numerical errors:
# plt.plot(energy(EE).T)

##
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))

lines = []
for x in E0:
  lines += [ax.plot([], [], 'o-', lw=2)[0]]

# Need to hide text handle among "lines"
time_template = 'time = %.1fs'
lines += [ax.text(0.05, 0.9, '', transform=ax.transAxes)]

def init():
  for line in lines[:-1]:
    line.set_data([],[])
  return lines

x012 = lambda x: (0,  L1*sin(x[0]),  L1*sin(x[0]) + L2*sin(x[2]) )
y012 = lambda x: (0, -L1*cos(x[0]), -L1*cos(x[0]) - L2*cos(x[2]) )

def animate(i):
  for x, line in zip(EE[i],lines[:-1]):
    line.set_data(x012(x), y012(x))
  lines[-1].set_text(time_template % (i*dt))
  return lines

import matplotlib.animation as animation
ani = animation.FuncAnimation(fig, animate, np.arange(1, len(EE)),
                              interval=1, blit=True, init_func=init)


##
