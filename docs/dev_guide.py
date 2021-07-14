"""# Developer guide

## Install for development

Make sure you included the dev tools as part of the installation
(detailed in the README):

```sh
pip install -e .[dev]
```

## Run tests

By default, only doctests are run when executing `pytest`.
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
flakehell lint
```

## Adding to the examples

Example scripts are very useful, and contributions are very desirable.  As well
as showcasing some feature, new examples should make sure to reproduce some
published literature results.  After making the example, consider converting
the script to the Jupyter notebook format (or vice versa) so that the example
can be run on Colab without users needing to install anything (see
`examples/README.md`). This should be done using the `jupytext` plug-in (with
the "lightscript" format), so that the paired files can be kept in synch.

## Documentation

### Update bib

Copy new bibtex items into `docs/bib/refs.bib`,
then add it to `docs/bib/bib.py` using

```sh
docs/bib/make_bib.py
```

### Run pdoc

```sh
pdoc --force --html --template-dir docs/templates -o ./docs \
docs/bib/bib.py docs/dev_guide.py dapper
open docs/index.html # preview
```

### Hosting

Push updated docs to github.
In the main github settings of the repo,
go to the "GitHub Pages" section,
and set the source to the docs folder.

## Profiling

- Launch your python script using `kernprof -l -v my_script.py`
- *Functions* decorated with `profile` will be timed, line-by-line.
- If your script is launched regularly, then `profile` will not be
  present in the `builtins.` Instead of deleting your decorations,
  you could also define a pass-through fallback.

## Making a release

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
"""
# vim: ft=markdown
