"""Tests for dapper module.

To run tests run `pytest`. For ex., if you want to avoid the slow plotting tests, do:
```sh
pytest --ignore=tests/test_plotting.py
```

Assuming you have `pytest-xdist` installed, you can do
`pytest -n auto` for multiprocessing.
However, this does not work smoothly with `test_example_2`,
and the output is a little ugly.

Assuming you have `pytest-clarity` installed,
which is activated by passing `-vv` to pytest,
some of the tests are easier to interpret (upon failure),
but otherwise the output is a little too verbose.

What about `pytest --doctest-modules` ?
"""
