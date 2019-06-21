# 


from dapper import *
from dapper.mods.Lorenz84.core import step, x0

##
simulator = with_recursion(step, prog="Simulating")

N  = 400
K  = 10
xx = simulator(x0, k=N*K, t0=0, dt=0.01)

##
fig, ax = plt.subplots(subplot_kw={'projection':'3d'})

cc = plt.cm.winter(linspace(0,1,N))
for n in range(N):
  ax.plot(*xx[n*K : (n+1)*K+1].T, lw=1, c=cc[n])

fig.suptitle('Phase space evolution')
ax.set_facecolor('w')
[eval("ax.set_%slabel('%s')"%(s,s)) for s in "xyz"]

ax.view_init(0, 0)

##
