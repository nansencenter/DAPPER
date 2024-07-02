# Test resample function

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from dapper.da_methods.particle import resample

from statsmodels.nonparametric.kernel_density import KDEMultivariate as kde  # noqa

f, axs = plt.subplots(7, 1, sharex=True, sharey=True)

N = 5 * 10**2

# Target distribution
dof = 3
P = np.random.chisquare(dof, N)


def pdf(x, k):
    return 2 ** (-k / 2) / sp.special.gamma(k / 2) * x ** (k / 2 - 1) * np.exp(-x / 2)


# Proposal distribution
scl = 5
q = np.random.exponential(scl, N)


def qdf(x, scl):
    return 1 / scl * np.exp(-x / scl)


# Weights
w = np.array([pdf(q[i], dof) / qdf(q[i], scl) for i in range(N)])
w = w / w.sum()

# Resaple
r, _ = resample(w, "Residual")
r = q[r]
s, _ = resample(w, "Systematic")
s = q[s]
t, _ = resample(w, "Stochastic")
t = q[t]

XL = 15
xx = np.linspace(0, XL, 201)
bins = np.linspace(0, XL, 50)

# Illustrate
axs[0].hist(P, bins, density=True, label="Example sample")
axs[1].hist(q, bins, density=True, label="Proposal sample and pdf")
axs[2].hist(q, bins, density=True, label="Proposal sample - weighted", weights=w)
axs[3].hist(r, bins, density=True, label="resmpl: Residual")
axs[4].hist(s, bins, density=True, label="resmpl: Systematic")
axs[5].hist(t, bins, density=True, label="resmpl: Stochastic")

# Add actual pdfs
axs[0].plot(xx, pdf(xx, dof), label="pdf: Target")
axs[1].plot(xx, qdf(xx, scl), label="pdf: Proposal")

# kde pdf comparison
axs[6].plot(xx, pdf(xx, dof), c="k", lw=3, label="pdf: Target")
axs[6].plot(xx, qdf(xx, scl), c="k", lw=2, label="pdf: Proposal")
axs[6].plot(xx, kde(r, "c", bw=[0.1]).pdf(xx), label="kde: Residual")
axs[6].plot(xx, kde(s, "c", bw=[0.1]).pdf(xx), label="kde: Systematic")
axs[6].plot(xx, kde(t, "c", bw=[0.1]).pdf(xx), label="kde: Stochastic")


axs[0].set_yticklabels([])
for ax in f.axes:
    ax.legend()

plt.pause(0.1)
