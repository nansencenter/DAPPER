"""Tests for `round2` and `round2sigfig`."""

import numpy as np
import pytest

from dapper.tools.rounding import round2, round2sigfig


class ca(float):
    """Make `==` approximate.

    Example:
    >>> ca(1 + 1e-6) == 1
    True

    This might be a roundabout way to execute `np.isclose`,
    but it is a fun way.
    """

    def __new__(cls, val, tol=1e-5):
        """From <https://stackoverflow.com/q/35943789>."""
        self = super().__new__(cls, val)
        self.tol = tol
        return self

    def __eq__(self, other):
        """Make equality comparison approximate."""
        return np.isclose(self, other, self.tol)


# Test cases for round2sigfig
lst1 = [
    (1   , 0, 0)   ,
    (11  , 0, 0)   ,
    (111 , 0, 0)   ,
    (1   , 1, 1)   ,
    (11  , 1, 10)  ,
    (111 , 1, 100) ,
    (1   , 2, 1)   ,
    (11  , 2, 11)  ,
    (111 , 2, 110) ,
    (1   , 3, 1)   ,
    (11  , 3, 11)  ,
    (111 , 3, 111) ,
    (1   , 4, 1)   ,
    (11  , 4, 11)  ,
    (111 , 4, 111) ,
    (1.11, 1, 1)   ,
    (1.11, 2, 1.1) ,
    (1.11, 3, 1.11),
    (1.11, 4, 1.11),
]

# Test cases for round2
lst2 = [
    (1.2345, 1.0  , 1)     ,
    (12.345, 1.0  , 12)    ,
    (123.45, 1.0  , 123)   ,
    (1234.5, 1.0  , 1234)  ,
    (12345., 1.0  , 12345) ,

    (1.2345, 9.0  , 1)     ,
    (12.345, 9.0  , 12)    ,
    (123.45, 9.0  , 123)   ,
    (1234.5, 9.0  , 1234)  ,
    (12345., 9.0  , 12345) ,

    (1.2345, 10.  , 0)     ,
    (12.345, 10.  , 10)    ,
    (123.45, 10.  , 120)   ,
    (1234.5, 10.  , 1230)  ,
    (12345., 10.  , 12340) ,

    (1.2345, 0.1  , 1.2)   ,
    (12.345, 0.1  , 12.3)  ,
    (123.45, 0.1  , 123.4) ,
    (1234.5, 0.1  , 1234.5),
    (12345., 0.1  , 12345) ,

    (1.2345, 0.2  , 1.2)   ,
    (12.345, 0.2  , 12.3)  ,
    (123.45, 0.2  , 123.4) ,
    (1234.5, 0.2  , 1234.5),
    (12345., 0.2  , 12345) ,

    (0.1   , 0.3  , 0.1)   ,
    (0.2   , 0.3  , 0.2)   ,
    (0.3   , 0.3  , 0.3)   ,

    (1.65  , 1.234, 2.0)   ,
    (1.65  , 0.543, 1.6)   ,
    (1.87  , 0.543, 1.9)   ,
    (1.92  , 0.543, 1.9)   ,
]


# For inspiration:
# p = 99.3
# print(p)
# for x in range(10):
#     x = 148.5 + .1*x
#     print(f"x: {x:.1f}:", round2(x, p))


@pytest.mark.parametrize("x, p, y", lst1)
def test_round2sigfig(x, p, y):
    rounded = round2sigfig(x, p)
    desired = ca(y, 1e-9)
    assert rounded == desired


@pytest.mark.parametrize("x, p, y", lst2)
def test_round2(x, p, y):
    rounded = round2(x, p)
    desired = ca(y, 1e-9)
    assert rounded == desired


if __name__ == "__main__":
    # Demonstrate `ca`.
    numbers = [
        1 + 1e-6,
        1 + 1e-4,
    ]
    for x in numbers:
        print("\nx:", x)
        values = {
            "=": x,
            "≈": ca(x),
        }
        for mode, x2 in values.items():
            print(f"x {mode} 1 {mode} x:", x2 == 1 == x2)
