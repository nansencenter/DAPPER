"""Demonstrate the AR model: simulate and plot each (rescaled) component."""

import matplotlib.pyplot as plt

from dapper.mods.AR.tutorials import HMM
from dapper.tools.seeding import set_seed

set_seed(4)
HMM.tseq.Ko = 200
xx, yy = HMM.simulate()

fig, axs = plt.subplots(nrows=HMM.Nx, sharex=True, figsize=(8, 6))
for i, ax in enumerate(axs):
    ax.plot(HMM.tseq.tt, xx[:, i], c="k")
    ax.set_ylabel(f"$x_{i}$")
axs[0].plot(HMM.tseq.tto, [y[0] for y in yy], ".", ms=3)
axs[-1].set_xlabel("t")
plt.show()
