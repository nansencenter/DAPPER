"""A land-ocean setup for Lorenz-96 from Anderson's 2009 Tellus A article:
'Spatially and temporally varying adaptive covariance inflation for ensemble filters'
"""


from dapper.mods.Lorenz95.sakov2008 import *
from dapper.tools.localization import general_localization, pairwise_distances

t = Chronology(0.05, dtObs=0.05, KObs=4000, Tplot=Tplot, BurnIn=2000*0.05)

# Define obs sites
obs_sites = 0.395 + 0.01*arange(1,21)
obs_sites *= 40
# Surrounding inds
ii_below = obs_sites.astype(int)
ii_above = ii_below + 1              
# Linear-interpolation weights
w_above = obs_sites - ii_below
w_below = 1 - w_above
# Define obs matrix
H = zeros((20,40))
H[arange(20),ii_below] = w_below
H[arange(20),ii_above] = w_above
# Measure obs-state distances
y2x_dists = pairwise_distances(obs_sites[:,None], arange(Nx)[:,None], (Nx,), periodic=True)
batches = arange(40)[:,None]
# Define operator
Obs = {
      'M'     : len(H),
      'model' : lambda E,t: E @ H.T,
      'linear': lambda E,t: H,
      'noise' : 1,
      'localizer': general_localization(lambda t: y2x_dists, batches),
      }

HMM = HiddenMarkovModel(Dyn,Obs,t,X0,LP=LPs(jj),
        sectors={'land':np.arange(*xtrema(obs_sites)).astype(int)})

####################
# Suggested tuning
####################

# Reproduce Anderson Figure 2
# -----------------------------------------------------------------------------------
# config = SL_EAKF(N=6, infl=sqrt(1.1), loc_rad=0.2/1.82*40)
# for lbl in ['err','std']:
#     stat = getattr(config.stats,lbl).f[HMM.t.maskObs_BI]
#     plt.plot(sqrt(mean(stat**2, axis=0)),label=lbl)
#
# Note: for this config, one must to be lucky with the random seed to avoid
#       blow up in the ocean sector (which is not constrained by obs) due to infl.
#       Instead, I recommend lowering dt (as in Miyoshi 2011) to stabilize integration.
