"""Stream function and observation time series for QG (quasi-geostrophic) model."""

if __name__ == "__main__":  # dont run if pdoc (sample may not be avail/generate-able)
    import numpy as np
    from matplotlib import pyplot as plt

    import dapper as dpr
    from dapper.mods.QG.demo import show
    from dapper.mods.QG.sakov2008 import HMM, obs_inds
    from dapper.tools.progressbar import progbar

    # Load (or generate) time-series data of a simulated state AND obs.
    # (Alternative: re-use sample of QG/__init__.py and observe it)
    fname = dpr.rc.dirs.data / "QG-ts.npz"
    try:
        with np.load(fname) as data:
            xx = data['xx'][1:]
            yy = data['yy']
    except FileNotFoundError:
        xx, yy = HMM.simulate()
        np.savez(fname, xx=xx, yy=yy)

    # Insert obs on the same "grid" as the state vector
    # Allocate the storage on the parent state dimension
    yy_xx = np.full_like(xx, np.NaN)
    for k, ko, t, dt in progbar(HMM.tseq.ticker, "Truth & Obs"):
        if ko is not None:
            indx = obs_inds(t)
            yy_xx[ko, indx] = yy[ko, :]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(12, 6))
    for ax in (ax1, ax2):
        ax.set_aspect('equal', 'box')
    ax1.set_title(r'Stream function: $\psi$')
    ax2.set_title(r'Noisy observations: $\mathcal{H}(\psi) + \epsilon$')

    # Define plot updating functions
    setter1 = show(xx[0]   , psi=True, ax=ax1)
    setter2 = show(yy_xx[0], psi=True, ax=ax2)

    # Create double iterable for the animation
    ts = zip(xx, yy_xx)

    # Animate
    for k, (xx, yy_xx) in progbar(list(enumerate(ts)), "Animating"):
        if k % 2 == 0:
            fig.suptitle("k: "+str(k))
            setter1(xx)
            setter2(yy_xx)
            plt.pause(0.01)
