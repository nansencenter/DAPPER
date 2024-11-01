# Examples

Here are some example scripts using DAPPER.
They all consist of one (or more) synthetic experiments.

Run them using `python docs/examples/the_script.py`,
or with the `%run` command inside `ipython`.

Some of the scripts have also been converted to Jupyter notebooks (`.ipynb`).
You can try them out without installing anything
by pressing this button (but note that some plotting features won't work,
and that it requires a Google login): [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/nansencenter/DAPPER)

## Description

When adapting the scripts to your needs,
you should begin with `basic_1.py`
before incorporating the aspects of `basic_2` and `basic_3`.

- [`basic_1.py`](basic_1): A single experiment, with Liveplotting.
- [`basic_2.py`](basic_2): Comparison of several DA methods.
- `basic_3.py`: Comparison of *many* DA methods and other experiment settings.
- `time-dep-obs-operator.py`: Similar to `basic_1`, but with "double" Lorenz-63 systems
  evolving independently, and observations of each "half" at alternating times.
- `param_estim.py`: Similar to `basic_2`, but with parameter estimation.
- `stoch_model1.py`: A perfect-yet-random model, with various integration schemes.
- `stoch_models.py`: As above, but studies the relationship between
  model uncertainty and numerical discretization error.
