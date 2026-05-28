"""Liveplotting tests that run headless (CI-compatible) on the Agg backend.

These check that the plotting code path produces non-empty figure content,
not just that it doesn't crash. The old test_plotting_interactive.py
is preserved for manual visual testing with an interactive backend.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")

plt.switch_backend("agg")  # headless; must come before dapper import

import dapper as dpr
import dapper.da_methods as da

# dpr_config disables liveplotting at import time when the backend is
# non-interactive. Re-enable it: Agg renders fine for headless testing.
dpr.rc.liveplotting = True

_NO_PAUSE = {"pause_a": 0, "pause_f": 0, "pause_s": 0, "pause_i": 0}


@pytest.fixture(autouse=True)
def _close_figs():
    plt.close("all")
    yield
    plt.close("all")


def _check_figures(before):
    """New figures were created, each has finite data in a plausible range.

    Collects numeric content from lines (sliding_diagnostics, spatial1d, …),
    bar patches (weight_histogram), and images (correlations imshow). Figures
    that are text-only "not available" placeholders — e.g. correlations in
    replay mode, where the ensemble is not stored — are skipped.

    Exact DA output values are intentionally not checked here: that is the
    job of the DA tests (test_example_1.py etc.), and exact values are
    seed- and platform-dependent. The `< 1e6` bound catches only catastrophic
    failures (wrong array shape, blown-up stat) without prescribing correctness.
    """
    new = [n for n in plt.get_fignums() if n not in before]
    assert new, "No figures were created by liveplotting"
    for num in new:
        fig = plt.figure(num)
        label = fig.get_label()
        arrays = []
        for ax in fig.axes:
            for line in ax.get_lines():
                yd = line.get_ydata()
                if len(yd):
                    arrays.append(yd)
            for patch in ax.patches:
                arrays.append([patch.get_height()])
            for image in ax.images:
                arr = image.get_array()
                if arr is not None and arr.size:
                    arrays.append(np.asarray(arr, dtype=float).ravel())
        if not arrays:
            continue  # text-only "not available" figure — nothing numeric to check
        ydata = np.concatenate(arrays)
        finite = ydata[np.isfinite(ydata)]
        # Boundary NaNs are used as line-break separators in phase-space plots.
        assert len(finite) > 0, f"Figure {label!r}: all values are NaN or inf"
        assert (
            np.max(np.abs(finite)) < 1e6
        ), f"Figure {label!r}: implausibly large values"


# ---------------------------------------------------------------------------
# Lorenz-63
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def L63_HMM():
    from dapper.mods.Lorenz63.sakov2012 import HMM as _HMM

    HMM = _HMM.copy()
    HMM.tseq.BurnIn = 0
    HMM.tseq.Ko = 1
    return HMM


def test_L63_liveplot(L63_HMM):
    xps = dpr.xpList()
    xps += da.EnKF("Sqrt", N=10, infl=1.02, rot=True)
    xps += da.PartFilt(N=20, reg=2.4, NER=0.3)
    before = set(plt.get_fignums())
    xps.launch(
        L63_HMM,
        free=False,
        liveplots="all",
        store_i=False,
        fail_gently=False,
        save_as=False,
        LP_kwargs=_NO_PAUSE,
    )
    _check_figures(before)


def test_L63_replay(L63_HMM):
    xps = dpr.xpList()
    xps += da.EnKF("Sqrt", N=10, infl=1.02, rot=True)
    xps.launch(
        L63_HMM,
        free=False,
        liveplots=False,
        store_i=False,
        fail_gently=False,
        save_as=False,
    )
    before = set(plt.get_fignums())
    xps[-1].stats.replay("all", speed=np.inf)
    _check_figures(before)


# ---------------------------------------------------------------------------
# Lorenz-96
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def L96_HMM():
    from dapper.mods.Lorenz96.sakov2008 import HMM as _HMM

    HMM = _HMM.copy()
    HMM.tseq.BurnIn = 0
    HMM.tseq.Ko = 2
    return HMM


def test_L96_liveplot(L96_HMM):
    xps = dpr.xpList()
    xps += da.EnKF("PertObs", N=40, infl=1.06)
    xps += da.LETKF(N=6, rot=True, infl=1.05, loc_rad=4, taper="Step")
    before = set(plt.get_fignums())
    xps.launch(
        L96_HMM,
        free=False,
        liveplots="all",
        store_i=False,
        fail_gently=False,
        save_as=False,
        LP_kwargs=_NO_PAUSE,
    )
    _check_figures(before)


def test_L96_replay(L96_HMM):
    xps = dpr.xpList()
    xps += da.EnKF("PertObs", N=40, infl=1.06)
    xps.launch(
        L96_HMM,
        free=False,
        liveplots=False,
        store_i=False,
        fail_gently=False,
        save_as=False,
    )
    before = set(plt.get_fignums())
    xps[-1].stats.replay("all", speed=np.inf)
    _check_figures(before)
