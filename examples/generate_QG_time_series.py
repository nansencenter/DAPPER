"""Stream function and observation time series for QG (quasi-geostrophic) model."""

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

def show(x0, ax=None):
    if ax == None:
        fig, ax = plt.subplots()

    im = ax.imshow(square(x0))
    im.set_clim(-30, 30)
    
    def update(x):
        im.set_data(square(x))
    return update

###########
# Main
###########

# try to load saved data 
try:
    tmp = np.load("QG_truth_obs_time_series.npz")
    xx = tmp['xx'][1:]
    yy = tmp['yy']
    tmp.close()
except:
    # else, generate stream funciton and observation time series as per standard
    # configuration  of Sakov2008, save the double time series
    xx, yy = HMM.simulate()
    np.savez("QG_truth_obs_time_series.npz", xx=xx, yy=yy)

# generate observations on the same gridding as the state vector
# initialize the storage on the parent state dimension
yy_xx = np.empty_like(xx)
yy_xx[:] = np.NaN

tseq = HMM.tseq
for k, ko, t, dt in pb.progbar(tseq.ticker, "Truth & Obs"):
    if ko is not None:
        indx = obs_inds(t)
        yy_xx[ko, indx] = yy[ko, :]

# create figure
fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(12, 6))
for ax in (ax1, ax2):
    ax.set_aspect('equal', 'box')
ax1.set_title(r'Stream function: $\psi$')
ax2.set_title(r'Noisy observations: $\mathcal{H}(\psi) + \epsilon$')

setter1 = show(xx[0], ax=ax1)
setter2 = show(yy_xx[0], ax=ax2)

# create dual iterable for the animated plot of stream function / obs
ts = zip(xx, yy_xx)

# run over the dual time series
for k, (xx, yy_xx) in pb.progbar(list(enumerate(ts)), "Animating"):
    if k % 2 == 0:
        fig.suptitle("k: "+str(k))
        setter1(xx)
        setter2(yy_xx)
        plt.pause(0.01)
