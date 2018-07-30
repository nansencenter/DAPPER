# Illustrate how to use DAPPER
# to tune parameters of the DA system
# This script builds on example_3, which must be understood first.

from common import *

sd0 = seed_init(8) # Base random seed.

from   mods.Lorenz95.boc15loc import setup
import mods.Lorenz95.core as core

CtrlVar = sys.argv[1] # command-line argument #1
#CtrlVar = 'N'

# Define range of the experiment control variable.
if CtrlVar == 'N': # Ensemble size.
  xticks = ccat(arange(2,20),[20, 22, 25, 30, 40, 50, 70, 100]) 

# Experiments duplication (random seeds will be varied).
xticks = array(xticks).repeat(16)

# Parallelization
xticks, save_path, rep_inds = distribute(__file__,sys.argv,xticks,CtrlVar)

##############################
# DA Configurations
##############################
cfgs  = List_of_Configs()

tuning_tag = 'loc_rad'

#INFL = 1.02
INFL = 1.0

if tuning_tag=='rot':
  for infl in [1]:
    for rot in linspace(0,1,7):
      cfgs += EnKF('Sqrt',N='?',infl=INFL,rot=rot)

elif tuning_tag=='Lag':
  for Lag in ccat(arange(1,10),11,14,20):
    cfgs += iLEnKS('-N',N='?',Lag=Lag)

elif tuning_tag=='loc_rad':
  for r in [0.2, 0.5, 1, 2, 3, 5, 7, 10, 20]: 
    cfgs += LETKF(N='?',loc_rad=r,infl=INFL)


##############################
# Assimilate
##############################
avrgs = np.empty((len(xticks),1,len(cfgs)),dict)

for iX,(X,iR) in enumerate(zip(xticks,rep_inds)):
  with coloring(): print('\n'+"xticks[",iX,'/',len(xticks)-1,"] ",CtrlVar,': ',X,sep="")

  seed(sd0+iR)
  xx,yy = simulate(setup)

  for iC,C in enumerate(cfgs):
    C = C.update_settings(N=X)
    seed(sd0+iR)
    stat = C.assimilate(setup,xx,yy)
    avrgs[iX,0,iC] = stat.average_in_time()

  print_averages(cfgs,avrgs[iX,0])

np.savez(save_path,
    avrgs      = avrgs,            # 3D array of dicts whose fields are the averages.
    xlabel     = CtrlVar,          # The control variable tag (string).
    xticks     = xticks,           # xticks (array).
    tuning_tag = tuning_tag,       # Tag to search for within labels.
    labels     = cfgs.gen_names()) # List of strings.


##############################
# Results load & presentation
##############################
if 'WORKER' in sys.argv: sys.exit(0) # quit if script is running as worker.

R = ResultsTable(save_path)

# Plot as bunch of lines
fig, ax = plt.subplots()
R.plot_1d()
ax.set_yscale('log')
yt = [0.1, 0.2, 0.5, 1, 2, 5]
ax.set_yticks(yt); ax.set_yticklabels(yt)
ax.legend()

# Plot field in 2D: (x-axis: control-var, y-axis: tuning-var)
fig, ax = plt.subplots()
R.plot_2d(log=True)
#R.plot_2d(log=False,cMax=0.8)


##

