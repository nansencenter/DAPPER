"""Test localization module."""
import numpy as np
from numpy import sqrt

from dapper.tools.localization import pairwise_distances

A2 = [[0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
      [2, 2]]


def test_pairwise_distances_2():
    """Test with a 2-dimensional collection."""
    dists = pairwise_distances(A2)
    answer = np.array([
        [0.     , 1.     , 1.     , sqrt(2), sqrt(8)],
        [1.     , 0.     , sqrt(2), 1.     , sqrt(5)],
        [1.     , sqrt(2), 0.     , 1.     , sqrt(5)],
        [sqrt(2), 1.     , 1.     , 0.     , sqrt(2)],
        [sqrt(8), sqrt(5), sqrt(5), sqrt(2), 0.     ]])
    assert np.all(dists == answer)


A3 = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [2, 0, 0],
    [0, 0, 2],
    [0, 0, 3],
])


def test_pairwise_distances_3():
    """Test with a 3-dimensional collection."""
    dists = pairwise_distances(A3)
    answer = np.array([
        [0.      , 1.      , 2.      , 2.      , 3.      ],
        [1.      , 0.      , 1.      , 2.236068, 3.162278],
        [2.      , 1.      , 0.      , 2.828427, 3.605551],
        [2.      , 2.236068, 2.828427, 0.      , 1.      ],
        [3.      , 3.162278, 3.605551, 1.      , 0.      ]])
    assert np.allclose(dists, answer)


def test_pairwise_distances_3_periodic():
    """Test with a 3-dimensional collection, assuming a periodic domain."""
    dists = pairwise_distances(A3, domain=(10, 10, 3))
    answer = np.array([
        [0.      , 1.      , 2.      , 1.      , 0.      ],
        [1.      , 0.      , 1.      , 1.414214, 1.      ],
        [2.      , 1.      , 0.      , 2.236068, 2.      ],
        [1.      , 1.414214, 2.236068, 0.      , 1.      ],
        [0.      , 1.      , 2.      , 1.      , 0.      ]])
    assert np.allclose(dists, answer)
