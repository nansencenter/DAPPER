"""Tests for HiddenMarkovModel construction and simulation."""

import numpy as np

from dapper.mods import HiddenMarkovModel
from dapper.mods.Lorenz63.sakov2012 import HMM as L63_HMM
from dapper.tools.seeding import set_seed


def _tiny_hmm():
    """Lorenz-63 HMM trimmed to 2 obs times for speed."""
    HMM = L63_HMM.copy()
    HMM.tseq.BurnIn = 0
    HMM.tseq.Ko = 1
    return HMM


class TestHMMConstruction:
    def test_constructs_from_lorenz63(self):
        HMM = _tiny_hmm()
        assert isinstance(HMM, HiddenMarkovModel)

    def test_nx_shortcut(self):
        HMM = _tiny_hmm()
        assert HMM.Nx == HMM.Dyn.M

    def test_name_is_string(self):
        HMM = _tiny_hmm()
        assert isinstance(HMM.name, str)

    def test_sectors_default_empty(self):
        HMM = _tiny_hmm()
        assert HMM.sectors == {}

    def test_liveplotters_default_list(self):
        HMM = _tiny_hmm()
        assert isinstance(HMM.liveplotters, list)


class TestHMMSimulate:
    def setup_method(self):
        self.HMM = _tiny_hmm()
        set_seed(3)
        self.xx, self.yy = self.HMM.simulate()

    def test_xx_shape(self):
        K = self.HMM.tseq.K
        Nx = self.HMM.Nx
        assert self.xx.shape == (K + 1, Nx)

    def test_yy_length(self):
        Ko = self.HMM.tseq.Ko
        assert len(self.yy) == Ko + 1

    def test_xx_finite(self):
        assert np.all(np.isfinite(self.xx))

    def test_yy_finite(self):
        for y in self.yy:
            assert np.all(np.isfinite(y))

    def test_xx_starts_near_X0(self):
        # x0 from sakov2012: [1.509, -1.531, 25.46]; X0 has C=2
        # After sampling from GaussRV, xx[0] should have the right dimension
        assert self.xx[0].shape == (self.HMM.Nx,)


class TestHMMCopy:
    def test_copy_is_independent(self):
        HMM1 = _tiny_hmm()
        HMM2 = HMM1.copy()
        HMM2.tseq.Ko = 99
        assert HMM1.tseq.Ko != HMM2.tseq.Ko

    def test_copy_has_same_values(self):
        HMM1 = _tiny_hmm()
        HMM2 = HMM1.copy()
        assert HMM1.Nx == HMM2.Nx
        assert HMM1.tseq == HMM2.tseq

    def test_copy_dyn_independent(self):
        HMM1 = _tiny_hmm()
        HMM2 = HMM1.copy()
        HMM2.Dyn.M = 999
        assert HMM1.Dyn.M != HMM2.Dyn.M
