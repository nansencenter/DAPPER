"""A land-ocean setup from `bib.anderson2009spatially`."""

import numpy as np

import dapper.mods as modelling
from dapper.mods.Lorenz96.sakov2008 import X0, Dyn, LPs, Nx, Tplot
from dapper.tools.localization import localization_setup, pairwise_distances
from dapper.tools.viz import xtrema

tseq = modelling.Chronology(0.05, dto=0.05, Ko=4000, Tplot=Tplot, BurnIn=2000*0.05)

# Define obs sites
obs_sites = 0.395 + 0.01*np.arange(1, 21)
obs_sites *= 40
# Surrounding inds
ii_below = obs_sites.astype(int)
ii_above = ii_below + 1
# Linear-interpolation weights
w_above = obs_sites - ii_below
w_below = 1 - w_above
# Define obs matrix
H = np.zeros((20, 40))
H[np.arange(20), ii_below] = w_below
H[np.arange(20), ii_above] = w_above
# Measure obs-state distances
y2x_dists = pairwise_distances(obs_sites[:, None], np.arange(Nx)[:, None], domain=(Nx,))
batches = np.arange(40)[:, None]
# Define operator
Obs = {
    'M': len(H),
    'model': lambda E, t: E @ H.T,
    'linear': lambda E, t: H,
    'noise': 1,
    'localizer': localization_setup(lambda t: y2x_dists, batches),
}

HMM = modelling.HiddenMarkovModel(
    Dyn, Obs, tseq, X0, LP=LPs(),
    sectors={'land': np.arange(*xtrema(obs_sites)).astype(int)})

####################
# Suggested tuning
####################

# Reproduce Anderson Figure 2
# -----------------------------------------------------------------------------------
# xp = SL_EAKF(N=6, infl=sqrt(1.1), loc_rad=0.2/1.82*40)
# for lbl in ['err', 'spread']:
#     stat = getattr(xp.stats,lbl).f[HMM.tseq.masko]
#     plt.plot(sqrt(np.mean(stat**2, axis=0)),label=lbl)
#
# Note: for this xp, one must to be lucky with the random seed to avoid
#       blow up in the ocean sector (which is not constrained by obs) due to infl.
#       Instead, I recommend lowering dt (as in Miyoshi 2011) to stabilize integration.
