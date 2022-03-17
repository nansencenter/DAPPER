"""Ensemble generation for stream function time series for QG (quasi-geostrophic) model."""

from matplotlib import pyplot as plt
import numpy as np
import dapper.mods as modelling
import dapper as dpr
from dapper.mods.QG import square, model_config, default_prms, shape
import dapper.tools.progressbar as pb

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


#########################
# Free ensemble run
#########################
def gen_ensemble_sample(model, nSamples, nEnsemble, SpinUp, Spacing):
    simulator = modelling.with_recursion(model.step, prog="Simulating")
    K         = SpinUp + nSamples*Spacing
    Nx        = np.prod(shape)  # total state length
    init      = np.random.normal(loc=0.0, scale=0.1, size=[nEnsemble, Nx])
    sample    = simulator(init, K, 0.0, model.prms["dtout"])
    return sample[SpinUp::Spacing,:, :]


###########
# Main
###########
# Load or generate time-series data of a simulated state and obs:
fname = dpr.rc.dirs.data / "QG-ts-en.npz"
np.random.seed(123)

# ensemble size needs to be at least Ne=2 for plotting to be true
plotting = True
Ne = 2

try:
    with np.load(fname) as data:
        E1 = np.squeeze(data['ens'][:, 0, :])
        E2 = np.squeeze(data['ens'][:, 1, :])
except FileNotFoundError:
    sample = gen_ensemble_sample(model_config("sample_generation", {}), 
                                 400, Ne, 10, 10)
    E1 = np.squeeze(sample[:, 0, :])
    E2 = np.squeeze(sample[:, 1, :])
    np.savez(fname, ens=sample)

if plotting == True:
    # Create figure
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(12, 6))
    for ax in (ax1, ax2):
        ax.set_aspect('equal', 'box')
    ax1.set_title(r'Ensemble member 1')
    ax2.set_title(r'Ensemble member 2')
    
    # Define plot updating functions
    setter1 = show(E1[0], ax=ax1)
    setter2 = show(E2[0], ax=ax2)
    
    # Create double iterable for the animation
    ts = zip(E1, E2)
    
    # Animate
    for k, (E1, E2) in pb.progbar(list(enumerate(ts)), "Animating"):
        if k % 2 == 0:
            fig.suptitle("k: "+str(k))
            setter1(E1)
            setter2(E2)
            plt.pause(0.01)
