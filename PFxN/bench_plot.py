# Plot benchmarks relating to the Particle Filter and PFxN
# with settings from mods/Lorenz95/boc10.py.
# Some of these are obtained with shorter experimental lengths,
# and are not fully reliable.

from common import *
plt.style.use('AdInf/paper.mplstyle')

OD = OrderedDict
nn = OD()
yy = OD()

nn['EnKF'] = [5    , 6    , 7    , 8    , 20   , 100  , 4000]
yy['EnKF'] = [0.36 , 0.35 , 0.32 , 0.31 , 0.27 , 0.27 , 0.27]

nn['iEnKS'] = [5    , 8    , 20   , 100  , 4000]
yy['iEnKS'] = [0.27 , 0.25 , 0.24 , 0.23 , 0.23]

nn['PF'] = [10  , 30  , 50  , 100  , 800  , 4000]
yy['PF'] = [4.8 , 4.8 , 1.0 , 0.36 , 0.23 , 0.22]

nn['PF Impl'] = [20   , 30   , 50   , 100  , 800  , 4000]
yy['PF Impl'] = [4.8  , 0.51 , 0.45 , 0.37 , 0.23 , 0.21]

nn['PF xN'] = [20   , 30   , 50   , 100  , 400  , 800  , 4000]
yy['PF xN'] = [4.8  , 0.43 , 0.34 , 0.27 , 0.24 , 0.22 , 0.22]

fig, ax = plt.subplots()

hh = OD()
for label, ens in nn.items():
  hh[label], = ax.plot(ens,yy[label],marker='o',label=label,lw=3.5)

ax.set_xscale('log')
ax.set_title('Benchmarks')
ax.set_ylabel('Average RMSE')
ax.set_xlabel('Ensemble size ($N$)')
ax.set_ylim(0.2,0.6)
ax.legend()

# Hide all elements
toggle_viz(hh.values(),prompt=False) # legend=0 also nice

savefig_n.index = 1
def tog(h,save=True,*a,**b):
  toggle_viz(h,*a,**b)
  if save: savefig_n()

tog(hh['EnKF'])
tog(hh['iEnKS'])
tog(hh['PF'])
tog(hh['PF Impl'])
tog(hh['PF xN'])
