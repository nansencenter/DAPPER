"""Tests of randvars module"""

import numpy as np

from dapper.tools.randvars import GaussRV


def test_gauss_rv():
    M = 4
    nsamples = 5
    grv = GaussRV(mu=0, C=0, M=M)
    assert (grv.sample(nsamples) == np.zeros((nsamples, M))).all()


test_gauss_rv()
