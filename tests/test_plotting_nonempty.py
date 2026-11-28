"""Liveplotting tests that run headless (CI-compatible) on the Agg backend.

These check that the plotting code path produces non-empty figure content,
not just that it doesn't crash. The old test_plotting_interactive.py
is preserved for manual visual testing with an interactive backend.
"""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")

plt.switch_backend("agg")  # headless; must come before dapper import

import dapper as dpr
import dapper.da_methods as da
from dapper.tools.liveplotting import correlations

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
        assert np.max(np.abs(finite)) < 1e6, (
            f"Figure {label!r}: implausibly large values"
        )


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


@pytest.mark.parametrize("half", [True, False])
def test_correlation_panels(half, monkeypatch):
    key = (0, 0, "a")
    stats = SimpleNamespace(mu={key: np.zeros(3)})
    ensemble = np.array([[0, 1, 0], [1, 0, 0], [0, 2, 1], [0, 0, 3]])
    monkeypatch.setattr(correlations, "half", half)
    plot = correlations("Correlations", stats, key, True, ensemble, None)
    fig = plot.ax.figure

    for use_ensemble in (True, False):
        if use_ensemble:
            plot(key, ensemble, None)
        else:
            fig.set_size_inches(9, 4)
            ensemble = ensemble * [1, -1, 1]
            plot(key, None, np.cov(ensemble, rowvar=False))
        fig.canvas.draw()

        expected = np.corrcoef(ensemble, rowvar=False)
        image = plot.im.get_array()
        mask = np.triu(np.ones_like(expected, dtype=bool), k=1) & half
        np.testing.assert_array_equal(np.ma.getmaskarray(image), mask)
        np.testing.assert_allclose(image.compressed(), expected[~mask])
        off_diag = expected[np.triu_indices(3, k=1)]
        signed = off_diag.mean()
        absolute = abs(off_diag).mean()
        np.testing.assert_allclose(plot.line_AC.get_ydata(), [1, signed, signed])
        np.testing.assert_allclose(plot.line_AA.get_ydata(), [1, absolute, absolute])

        assert plot.ax.get_shared_x_axes().joined(plot.ax, plot.ax2)
        assert not any(
            tick.label1.get_visible() or tick.label2.get_visible()
            for tick in plot.ax.xaxis.get_major_ticks()
        )
        assert any(t.get_text() for t in plot.ax2.get_xticklabels())
        points = [[0, 0], [2, 0]]
        np.testing.assert_allclose(
            plot.ax.transData.transform(points)[:, 0],
            plot.ax2.transData.transform(points)[:, 0],
        )
