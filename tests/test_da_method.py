"""Tests for the @da_method decorator."""

import pytest

from dapper.da_methods import da_method
from dapper.mods.Lorenz63.sakov2012 import HMM as L63_HMM
from dapper.tools.seeding import set_seed


def _tiny_hmm():
    """Lorenz-63 HMM trimmed to 2 obs times for speed."""
    HMM = L63_HMM.copy()
    HMM.tseq.BurnIn = 0
    HMM.tseq.Ko = 1
    return HMM


def _simulate(HMM):
    set_seed(3)
    return HMM.simulate()


# ---------------------------------------------------------------------------
# Decorator produces a dataclass-like class
# ---------------------------------------------------------------------------


class TestDecoratorStructure:
    def test_init_with_fields(self):
        @da_method()
        class Dummy:
            alpha: float = 1.0

            def assimilate(self, HMM, xx, yy):
                pass

        assert Dummy(alpha=2.5).alpha == 2.5

    def test_default_field(self):
        @da_method()
        class Dummy:
            alpha: float = 1.0

            def assimilate(self, HMM, xx, yy):
                pass

        assert Dummy().alpha == 1.0

    def test_repr_includes_class_name(self):
        @da_method()
        class Dummy:
            alpha: float = 1.0

            def assimilate(self, HMM, xx, yy):
                pass

        assert "Dummy" in repr(Dummy())

    def test_equality(self):
        @da_method()
        class Dummy:
            alpha: float = 1.0

            def assimilate(self, HMM, xx, yy):
                pass

        assert Dummy(alpha=1.0) == Dummy(alpha=1.0)
        assert Dummy(alpha=1.0) != Dummy(alpha=2.0)

    def test_da_method_class_attr(self):
        @da_method()
        class MyFilter:
            N: int = 10

            def assimilate(self, HMM, xx, yy):
                pass

        assert MyFilter.da_method == "MyFilter"

    def test_missing_assimilate_raises(self):
        with pytest.raises(AttributeError):

            @da_method()
            class Bad:
                pass


# ---------------------------------------------------------------------------
# Inherited defaults via default_dataclasses
# ---------------------------------------------------------------------------


class TestInheritedDefaults:
    def test_inherited_defaults_present(self):
        class shared:
            infl: float = 1.0
            rot: bool = False

        @da_method(shared)
        class FilterA:
            N: int

            def assimilate(self, HMM, xx, yy):
                pass

        f = FilterA(N=10)
        assert f.infl == 1.0
        assert f.rot is False

    def test_own_field_takes_precedence(self):
        class shared:
            infl: float = 1.0

        @da_method(shared)
        class FilterB:
            N: int
            infl: float = 2.0

            def assimilate(self, HMM, xx, yy):
                pass

        assert FilterB(N=5).infl == 2.0


# ---------------------------------------------------------------------------
# assimilate() creates self.stats and self.stat("duration", ...)
# ---------------------------------------------------------------------------


class TestAssimilateIntegration:
    def setup_method(self):
        self.HMM = _tiny_hmm()
        self.xx, self.yy = _simulate(self.HMM)

    def _make_noop(self):
        @da_method()
        class NoOp:
            N: int = 5

            def assimilate(self, HMM, xx, yy):
                E = HMM.X0.sample(self.N)
                self.stats.assess(0, E=E)
                for k, ko, _t, _dt in HMM.tseq.ticker:
                    E = HMM.X0.sample(self.N)
                    self.stats.assess(k, ko, E=E)

        return NoOp()

    def test_stats_attribute_created(self):
        xp = self._make_noop()
        xp.assimilate(self.HMM, self.xx, self.yy)
        assert hasattr(xp, "stats")

    def test_duration_non_negative(self):
        xp = self._make_noop()
        xp.assimilate(self.HMM, self.xx, self.yy)
        assert xp.stats.duration >= 0

    def test_average_in_time_populates_avrgs(self):
        xp = self._make_noop()
        xp.assimilate(self.HMM, self.xx, self.yy)
        xp.stats.average_in_time()

        # Top-level keys from Stats.new_series() calls
        expected_top = {
            "err",
            "spread",
            "mu",
            "gscore",
            "mad",
            "skew",
            "kurt",
            "duration",
        }
        assert expected_top <= set(xp.avrgs)

        # Field-summary sub-keys on vector stats (from field_summaries)
        assert {"m", "rms", "ma"} <= set(xp.avrgs.err)
        assert {"m", "rms", "ma"} <= set(xp.avrgs.spread)

        # Analysis values are accessible and finite
        # (.a is UncertainQtty, .val is the float)
        import math

        assert math.isfinite(xp.avrgs.err.rms.a.val)
        assert math.isfinite(xp.avrgs.spread.rms.a.val)
        assert math.isfinite(xp.avrgs.gscore.rms.a.val)
        assert math.isfinite(xp.avrgs.mad.a.val)

        # Spread must be positive
        assert xp.avrgs.spread.rms.a.val > 0

        # Scalar: duration is a plain float, not a mean_with_conf
        assert isinstance(xp.avrgs.duration, float)
        assert xp.avrgs.duration > 0

        # Ensemble-specific keys present (NoOp has N=5)
        assert "rh" in xp.avrgs  # rank histogram


# ---------------------------------------------------------------------------
# fail_gently behaviour
# ---------------------------------------------------------------------------


class TestFailGently:
    def setup_method(self):
        self.HMM = _tiny_hmm()
        self.xx, self.yy = _simulate(self.HMM)

    def _crasher(self):
        @da_method()
        class Crasher:
            def assimilate(self, HMM, xx, yy):
                raise RuntimeError("intentional crash")

        return Crasher()

    def test_fail_gently_true_does_not_raise(self):
        self._crasher().assimilate(self.HMM, self.xx, self.yy, fail_gently=True)

    def test_fail_gently_sets_crashed(self):
        xp = self._crasher()
        xp.assimilate(self.HMM, self.xx, self.yy, fail_gently="silent")
        assert xp.crashed is True

    def test_fail_gently_false_raises(self):
        with pytest.raises(RuntimeError, match="intentional crash"):
            self._crasher().assimilate(self.HMM, self.xx, self.yy, fail_gently=False)
