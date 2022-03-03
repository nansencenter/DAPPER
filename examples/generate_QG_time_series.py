"""Truth-twin and observation time series for QG (quasi-geostrophic) model."""

import numpy as np
import scipy.ndimage.filters as filters
from matplotlib import pyplot as plt
from dapper.mods.QG import square
import dapper.tools.progressbar as pb
from dapper.mods.QG.sakov2008 import HMM
from dapper.mods.QG.sakov2008 import obs_inds

###########
# Auxiliary plotting function
###########

def show(x0, psi=True, ax=None):
    #
    def psi_or_q(x):
        return x if psi else compute_q(x)
    #
    if ax == None:
        fig, ax = plt.subplots()

    im = ax.imshow(psi_or_q(square(x0)))

    if psi:
        im.set_clim(-30, 30)
    else:
        im.set_clim(-28e4, 25e4)

    def update(x):
        im.set_data(psi_or_q(square(x)))
    return update

###########
# Main
###########

## generate truth-twin and observation time series as per standard configuration
## of Sakov2008, save the double time series
xx, yy = HMM.simulate()

# save data for easy re-use and debugging
np.savez("QG_truth_obs_time_series.npz", xx=xx, yy=yy)

# these lines can be used in place of above steps after first run,
# comment out above
tmp = np.load("QG_truth_obs_time_series.npz")
xx = tmp['xx'][1:,]
yy = tmp['yy']
tmp.close()

# generate observations on the same gridding as the state vector

# initialize the storage on the parent state dimension
yy_xx = np.empty(np.shape(xx))
yy_xx[:] = np.NaN

tseq = HMM.tseq
for k, ko, t, dt in pb.progbar(tseq.ticker, "Truth & Obs"):
    if ko is not None:
        indx = obs_inds(t)
        yy_xx[ko, indx] = yy[ko, :]

# create figure
fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 4))
for ax in (ax1, ax2):
    ax.set_aspect('equal', 'box')
ax1.set_title(r'Truth-twin stream function: $\psi$')
ax2.set_title(r'Noisy observations: $\mathcal{H}(\psi) + \epsilon$')

setter1 = show(xx[0], psi=True, ax=ax1)
setter2 = show(yy_xx[0], psi=True, ax=ax2)

# create dual iterable for the animated plot of truth / obs
ts = zip(xx, yy_xx)

# run over the dual time series
for k, val in pb.progbar(list(enumerate(ts)), "Animating"):
    if k % 2 == 0:
        fig.suptitle("k: "+str(k))
        setter1(val[0])
        setter2(val[1])
        plt.pause(0.01)

