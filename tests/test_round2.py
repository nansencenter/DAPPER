import numpy as np

import dapper.tools.math as dm


class ca(float):
    """Make `==` approximate.

    Example:
    >>> ca(1 + 1e-6) == 1
    True

    This might be a roundabout way to execute `np.isclose`,
    but it is a fun way.
    """

    def __new__(cls, val, tol=1e-5):
        self = super().__new__(cls, val)
        self.tol = tol
        return self

    def __eq__(self, other):
        return np.isclose(self, other, self.tol)


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


print()
lst = [
    (1.11  , 4  , 1.11) ,
    (1.11  , 3  , 1.11) ,
    (1.11  , 2  , 1.1)  ,
    (1.11  , 1  , 1)    ,
    (1.11  , 1.0, 1)    ,
    (1     , 0  , 0)    ,
    (11    , 0  , 0)    ,
    (111   , 0  , 0)    ,
    (1     , 1  , 1)    ,
    (11    , 1  , 10)   ,
    (111   , 1  , 100)  ,
    (1     , 2  , 1)    ,
    (11    , 2  , 11)   ,
    (111   , 2  , 110)  ,
    (1     , 3  , 1)    ,
    (11    , 3  , 11)   ,
    (111   , 3  , 111)  ,
    (1     , 4  , 1)    ,
    (11    , 4  , 11)   ,
    (111   , 4  , 111)  ,
    (1.2345, 1.0, 1)    ,
    (12.345, 1.0, 12)   ,
    (123.45, 1.0, 123)  ,
    (1234.5, 1.0, 1234) ,
    (12345., 1.0, 12345),
    (1.2345, 2.0, 2)    ,
    (12.345, 2.0, 12)   ,
    (123.45, 2.0, 124)  ,
    (1234.5, 2.0, 1234) ,
    (12345., 2.0, 12344),
    (1.2345, 10., 0)    ,
    (12.345, 10., 10)   ,
    (123.45, 10., 120)  ,
    (1234.5, 10., 1230) ,
    (12345., 10., 12340),
    (1.2345, 0.1, 1.2)    ,
    (12.345, 0.1, 12.3)   ,
    (123.45, 0.1, 123.4)  ,
    (1234.5, 0.1, 1234.5) ,
    (12345., 0.1, 12345),
    (1.2345, 0.2, 1.2)    ,
    (12.345, 0.2, 12.4)   ,
    (123.45, 0.2, 123.4)  ,
    (1234.5, 0.2, 1234.4) ,
    (12345., 0.2, 12345),

    (0.1, 0.3, 0),
    (0.2, 0.3, 0.3),
    (0.3, 0.3, 0.3),
    (0.4, 0.3, 0.3),
    (0.5, 0.3, 0.6),

    (18.4, 12.3, 10),
    (18.5, 12.3, 20),

    (148.7, 99.2, 100),
    (148.8, 99.2, 200),
    (150.4, 100.3, 100),
    (150.5, 100.3, 200),
]


def test_all():
    for x, p, y in lst:
        y1 = dm.round2(x, p)
        cond = y1 == ca(y)
        print(f"{x:8} rounded by {p:<6} ≈ {y:<6} ?",
              cond, "" if cond else f"({y1})")
        assert cond

# For inspiration:
# p = 99.3
# print(p)
# for x in range(10):
    # x = 148.5 + .1*x
    # print(f"x: {x:.1f}:", dm.round2(x, p))
