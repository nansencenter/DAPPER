"""Tests for dapper.tools.chronos: Chronology and Ticker."""

import pytest

from dapper.tools.chronos import Chronology

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base() -> Chronology:
    """A simple reference Chronology: dt=0.1, dko=5, K=50 (T=5.0, Ko=9)."""
    return Chronology(dt=0.1, dko=5, K=50)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_dt_dko_K(self):
        c = Chronology(dt=0.1, dko=5, K=50)
        assert c.dt == pytest.approx(0.1)
        assert c.dko == 5
        assert c.K == 50

    def test_dt_dko_T(self):
        c = Chronology(dt=0.1, dko=5, T=5.0)
        assert c.K == 50
        assert c.T == pytest.approx(5.0)

    def test_dt_dko_Ko(self):
        c = Chronology(dt=0.1, dko=5, Ko=9)
        assert c.Ko == 9
        assert c.K == 50

    def test_dt_dto_K(self):
        c = Chronology(dt=0.1, dto=0.5, K=50)
        assert c.dko == 5
        assert c.dto == pytest.approx(0.5)

    def test_T_K_dko(self):
        c = Chronology(T=5.0, K=50, dko=5)
        assert c.dt == pytest.approx(0.1)

    def test_T_K_dto(self):
        c = Chronology(T=5.0, K=50, dto=0.5)
        assert c.dt == pytest.approx(0.1)
        assert c.dko == 5

    def test_wrong_param_count_raises(self):
        with pytest.raises((AssertionError, ValueError)):
            Chronology(dt=0.1, dko=5)  # only 2 params

    def test_wrong_param_count_too_many_raises(self):
        with pytest.raises((AssertionError, ValueError)):
            Chronology(dt=0.1, dko=5, K=50, T=5.0)  # 4 params


# ---------------------------------------------------------------------------
# Computed properties
# ---------------------------------------------------------------------------


class TestProperties:
    def setup_method(self):
        self.c = _base()  # dt=0.1, dko=5, K=50, T=5.0, Ko=9

    def test_T(self):
        assert self.c.T == pytest.approx(0.1 * 50)

    def test_Ko(self):
        assert self.c.Ko == 9  # K/dko - 1 = 50/5 - 1

    def test_dto(self):
        assert self.c.dto == pytest.approx(0.1 * 5)

    def test_kk(self):
        assert len(self.c.kk) == 51  # K+1
        assert self.c.kk[0] == 0
        assert self.c.kk[-1] == 50

    def test_kko(self):
        kko = self.c.kko
        assert len(kko) == 10  # Ko+1
        assert kko[0] == 5  # first obs at k=dko
        assert kko[-1] == 50  # last obs at k=K

    def test_tt(self):
        assert len(self.c.tt) == 51
        assert self.c.tt[0] == pytest.approx(0.0)
        assert self.c.tt[-1] == pytest.approx(5.0)

    def test_tto(self):
        tto = self.c.tto
        assert len(tto) == 10
        assert tto[0] == pytest.approx(0.5)

    def test_mask(self):
        c = Chronology(dt=0.1, dko=5, K=50, BurnIn=2.0)
        mask = c.mask
        assert mask.sum() == (50 - 20)  # steps after t=2.0

    def test_masko(self):
        c = Chronology(dt=0.1, dko=5, K=50, BurnIn=2.0)
        masko = c.masko
        assert masko.sum() == (10 - 4)  # obs times after t=2.0

    def test_iBurnIn(self):
        c = Chronology(dt=0.1, dko=5, K=50, BurnIn=2.0)
        assert c.tt[c.iBurnIn] == pytest.approx(2.1)  # first t > 2.0

    def test_ioBurnIn(self):
        c = Chronology(dt=0.1, dko=5, K=50, BurnIn=2.0)
        assert c.tto[c.ioBurnIn] == pytest.approx(2.5)  # first obs time > 2.0


# ---------------------------------------------------------------------------
# Ticker
# ---------------------------------------------------------------------------


class TestTicker:
    def test_ticker_length(self):
        c = _base()
        items = list(c.ticker)
        assert len(items) == 50  # K steps (ticker starts at k=1)

    def test_ticker_first_item(self):
        c = _base()
        k, ko, t, dt = next(iter(c.ticker))
        assert k == 1
        assert ko is None
        assert t == pytest.approx(0.1)
        assert dt == pytest.approx(0.1)

    def test_ticker_obs_times(self):
        c = _base()
        obs_ks = [k for k, ko, t, dt in c.ticker if ko is not None]
        assert obs_ks == list(c.kko)

    def test_ticker_ko_values(self):
        c = _base()
        obs_kos = [ko for k, ko, t, dt in c.ticker if ko is not None]
        assert obs_kos == list(range(10))  # ko = 0..Ko

    def test_ticker_no_obs_at_k0(self):
        # Convention: no obs at t=0 (k=0); ticker starts at k=1
        c = _base()
        first_obs_k = next(k for k, ko, t, dt in c.ticker if ko is not None)
        assert first_obs_k == c.dko

    def test_ticker_dt_values(self):
        c = _base()
        dts = [dt for k, ko, t, dt in c.ticker]
        assert all(dt == pytest.approx(0.1) for dt in dts)


# ---------------------------------------------------------------------------
# Setters
# ---------------------------------------------------------------------------


class TestSetters:
    def test_set_K(self):
        c = _base()
        c.K = 100
        assert c.K == 100
        assert c.T == pytest.approx(10.0)
        assert c.Ko == 19

    def test_set_T(self):
        c = _base()
        c.T = 10.0
        assert c.K == 100
        assert c.T == pytest.approx(10.0)

    def test_set_Ko(self):
        c = _base()
        c.Ko = 19
        assert c.Ko == 19
        assert c.K == 100

    def test_set_dko(self):
        c = _base()
        c.dko = 10
        assert c.dko == 10
        assert c.T == pytest.approx(10.0)  # T scales with dko

    def test_set_dt(self):
        c = _base()
        c.dt = 0.05
        assert c.dt == pytest.approx(0.05)
        assert c.dko == 10  # dko doubles to keep dto fixed


# ---------------------------------------------------------------------------
# copy and equality
# ---------------------------------------------------------------------------


class TestCopyEq:
    def test_eq_same(self):
        c1 = _base()
        c2 = _base()
        assert c1 == c2

    def test_eq_different(self):
        c1 = _base()
        c2 = Chronology(dt=0.1, dko=5, K=60)
        assert c1 != c2

    def test_copy_independent(self):
        c1 = _base()
        c2 = c1.copy()
        assert c1 == c2
        c2.K = 100
        assert c1 != c2

    def test_copy_preserves_values(self):
        c = _base()
        c2 = c.copy()
        assert c2.dt == pytest.approx(c.dt)
        assert c2.dko == c.dko
        assert c2.K == c.K
