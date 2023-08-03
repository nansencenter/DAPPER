"""Configures pytest (beyond the ini file)."""
import matplotlib as mpl
import numpy
import pytest
from matplotlib import pyplot as plt

from dapper.dpr_config import rc


@pytest.fixture(autouse=True)
def add_sci(doctest_namespace):
    """Add numpy as np for doctests."""
    doctest_namespace["np"] = numpy
    doctest_namespace["mpl"] = mpl
    doctest_namespace["plt"] = plt
    doctest_namespace["rc"] = rc
