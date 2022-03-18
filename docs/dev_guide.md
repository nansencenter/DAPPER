# Developer guide

## Conventions

- Ensemble (data) matrices are `np.ndarrays` with shape `N-by-Nx`.
  This shape (orientation) is contrary to the EnKF literature,
  but has the following advantages:
    - Improves speed in row-by-row accessing,
      since that's `np`'s default orientation.
    - Facilitates broadcasting for, e.g. centring the matrix.
    - Fewer indices: `[n,:]` yields same as `[n]`
    - Beneficial operator precedence without `()`.
      E.g. `dy @ Rinv @ Y.T @ Pw` (where `dy` is a vector)
    - Less transposing for for ensemble space formulae.
    - It's the standard for data matrices in the broader statistical literature.
- Naming:
    - `E`: ensemble matrix
    - `w`: ensemble weights or coefficients
    - `X`: centred ensemble ("anomalies")
    - `N`: ensemble size
    - `Nx`: state size
    - `Ny`: observation size
    - *Double letters* means a sequence of something.
      For example:
        - `xx`: Time series of truth; shape `(K+1, Nx)`
        - `yy`: Time series of observations; shape `(Ko+1, Nx)`
        - `EE`: Time series of ensemble matrices
        - `ii`, `jj`: Sequences of indices (integers)
    - `xps`: an `xpList` or `xpDict`,
      where `xp` abbreviates "experiment".


## Install for development

Make sure you included the dev tools as part of the installation
(detailed in the README):

```sh
pip install -e .[dev]
```

## Run tests

By default, only `doctests` are run when executing `pytest`.
To run the main tests, do this:

```sh
pytest tests
```

You can also append `test_plotting.py` for example,
which is otherwise ignored for being slow.

If the test with the `QG` model in `test_HMM.py` fails
(simply because you have not compiled it) that is fine
(that test does not run in CI either).

## Pre-commit and linting

Pull requests (PR) to DAPPER are checked with continuous integration (CI),
which runs the tests, and also linting, plus some `pre-commit` hooks.
To avoid having to wait for the CI server to run all of this,
you'll want to run them on your own computer:

```sh
pre-commit install
pre-commit run --all-files
```

Now every time you commit, these tests will run on the staged files.
For detailed linting messages, run

```sh
flakeheaven lint
```

You may also want to display linting issues in your editor as you code.
Below is a suggested configuration of VS-Code with the pylance plug-in
or Vim (with the coc.nvim plug-in with the pyright extension)

```jsonc
// Put this in settings.json (VS-Code) or ~/.vim/coc-settings.json (For Vim)
{
    "python.analysis.typeCheckingMode": "off",
    "python.analysis.useLibraryCodeForTypes": true,
    "python.analysis.extraPaths": ["scripts"],
    "python.formatting.provider": "autopep8",
    "python.formatting.autopep8Path":  "~/.local/bin/autopep8",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.flake8Args": ["lint"],
    "python.linting.flake8Path": "${env:CONDA_PREFIX}/bin/flakeheaven",
    // Without VENV (requires `pip3 install --user flakeheaven flake8-docstrings flake8-bugbear ...`)
    // "python.linting.flake8Path": "[USE PATH PRINTED BY PIP ABOVE]/Python/3.8/bin/flakeheaven",
}
```

## Adding to the examples

Example scripts are very useful, and contributions are very desirable.  As well
as showcasing some feature, new examples should make sure to reproduce some
published literature results.  After making the example, consider converting
the script to the Jupyter notebook format (or vice versa) so that the example
can be run on Colab without users needing to install anything (see
`examples/README.md`). This should be done using the `jupytext` plug-in (with
the `lightscript` format), so that the paired files can be kept in synch.

## Documentation

The documentation may be generated with `pdoc3`, e.g.

```sh
pdoc --force --html --template-dir docs/templates -o ./docs \
docs/bib/bib.py docs/dev_guide.py dapper
open docs/index.html # preview
```

This is done automatically by a GitHub workflow whenever
the `master` branch gets updated,
and the generated docs are pushed into the `gh-pages` branch,
to which the Github `Pages` settings of points to,
which hosts the website.

In order to add new references,
insert their bibtex into `docs/bib/refs.bib`,
then run `docs/bib/make_bib.py`,
which will format and add entries to `docs/bib/bib.py`.

A live preview of the documentation (that updates whenever you save
the python source file) can be had with

```sh
cd docs
pdoc --http : dapper dev_guide.py
```

## Profiling

- Launch your python script using `kernprof -l -v my_script.py`
- *Functions* decorated with `profile` will be timed, line-by-line.
- If your script is launched regularly, then `profile` will not be
  present in the `builtins.` Instead of deleting your decorations,
  you could also define a pass-through fallback.

## Publishing a release on PyPI

`cd DAPPER`

Bump version number in `__init__.py`

Merge `dev1` into `master`

```sh
git checkout master
git merge --no-commit --no-ff dev1
# Fix conflicts, e.g
# git rm <unwanted-file>
git commit
```

Make docs (including bib)
Tag

```sh
git tag -a v$(python setup.py --version) -m 'My description'
git push origin --tags
```

Clean

```sh
rm -rf build/ dist *.egg-info .eggs
```

Add new files to `package_data` and `packages` in `setup.py`

Build

```sh
./setup.py sdist bdist_wheel
```

Upload to PyPI

```sh
twine upload --repository pypi dist/*
```


Upload to Test.PyPI

```sh
twine upload --repository testpypi dist/*
```

where ~/.pypirc contains

```ini
[distutils]
index-servers=
                pypi
                testpypi

[pypi]
username: myuser
password: mypass

[testpypi]
repository: https://test.pypi.org/legacy/
username: myuser
password: mypass
```

Upload to `Test.PyPI`

```sh
git checkout dev1
```

### Test installation


Install from `Test.PyPI`

```sh
pip install --extra-index-url https://test.pypi.org/simple/ dapper
```

Install from `PyPI`

```sh
pip install dapper
```

- Install into specific dir (includes all of the dependencies)  
  `pip install dapper -t MyDir`

- Install with options  
  `pip install dapper[dev,Qt]`

Install from local (makes installation accessible from everywhere)

```sh
pip install -e .
```

