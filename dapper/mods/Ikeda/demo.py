"""Demonstrate the Ikeda map.

Plot settings inspired by Wikipedia.
"""

import numpy as np
from matplotlib import pyplot as plt

import dapper.mods as modelling
import dapper.mods.Ikeda as core


def demo(u, s0, N, as_points, ends):
    core.u = u

    # Simulation length
    K = 10**5 // N

    # Computations
    simulator = modelling.with_recursion(core.step, prog="Simulating")

    # Initial ensemble
    E0 = core.x0 + s0*np.random.randn(N, 2)

    # Simulate
    EE = simulator(E0, K, 0, 1)

    # Plot
    fig, ax = plt.subplots()
    fig.suptitle('Phase space' + f"\nu={core.u}, N={N}, $σ_0$={s0}")
    ax.set(xlabel="x", ylabel="y")

    # Re-order axes for plotting
    tail_length = 0  # 0 => infinite
    ET = EE[-tail_length:].transpose((2, 0, 1))

    if as_points:
        ax.plot(*ET, "b.", ms=1.0, alpha=1.0)
    else:
        ax.plot(*ET, "b-", lw=.02, alpha=0.1)

    if ends:
        ax.plot(*EE[0] .T, '*g', ms=7)
        ax.plot(*EE[-1].T, '*r', ms=7)


cases = [
    (.9, 0, 1, True, True),
    (.918, 5, 2000, False, False),
    (.99, 2, 200, True, True),
]

for case in cases:
    demo(*case)

plt.show()
