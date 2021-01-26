"""Configures pytest (beyond the ini file)."""
import numpy
import pytest


@pytest.fixture(autouse=True)
def add_sci(doctest_namespace):
    """Add numpy as np for doctests."""
    doctest_namespace["np"] = numpy
    doctest_namespace["rnd"] = numpy.random
