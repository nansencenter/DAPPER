"""
## Run tests
See `tests`.

## Documentation gen.

### Update bib
Copy new bibtex items into `docs/bib/refs.bib`,
then convert to bib.py using  
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

## TODOs

- There are also TODO labels scattered in the code,
  including a priority number (1-9, lower is more important).

- The documentation always needs improvement.
  The documentation uses pdoc3 to auto-generate API reference,
  so improving function and class docstrings is very helpful.

- Make Colab work for notebooks in examples.
  This requires that Colab upgrades to python 3.7,
  which I don't know when will happen.

- Write an example script (and/or make changes to DAPPER) to show how to:
    - do parameter estimation.
    - use different models for Truth and DA-methods.
    - work with real data.
    - pretend variable-length state or obs are fixed-length.

- Right now each column in `tabulate_avrgs` and `xpSpace.print`
  is treated independently, so that they may be aligned on the decimal point.
  But ideally the number of decimals printed in uq.val is determined by uq.prec.
  This is already the case, *somewhat*, since `unpack_uqs` uses `uq.round`
  But, trailing zeros will still get truncated. I.e. 0.3023 +/- 0.01
  gets tabulate-printed as 0.3 instead of 0.30. Should be fixed.

- Merge UncertainQtty with the ca class, including its __eq__ ?
  Rename Ca ?

- Simplify, improve and generalize (total rewrite?) time sequence management.
    - At the moment, `t` (absolute time) is "too important" in the code,
     compared to `k` (the time index). For example,

        - `dxdt` have signature `dxdt(x,t,dt)` instead of `dxdt(x,k,dt)`.
          But a common situation is that `dxdt` needs to look-up some
          property (e.g. parmeter value) from a pre-defined table.
          For that purpose, `k` is better suited that `t` because it is
          and integer, not a fload.
        - If models simply have the signature `HMM.dyn(x,k)`
          (i.e. `ticker` would only yield `k` and `kObs`)
          then variable dt is effectively supported.

    - Change KObs to KObs-1 ?
    - Highlight the fact that `t=0` is special
        - There shoult not be any obs then
        - Some of the plotting functionality are maybe
          assuming that `tt[0] == 0`.

- Simplify and/or improve (total rewrite?) the CovMat class.

- Make superclasses for the filter, smoother, and iterative smoother.

- Pause after Enter doesn't work on Ubuntu?

- For xpList, could also subclass collections.UserList,
  although I don't know if it would be any better
  <https://stackoverflow.com/q/25464647>

- Make autoscaler.py work also for other users
  (I dont know if it's condor_- that outputs user-specific numbers,
  or glcoud resize command that is user-specific, or what.)


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

#### Test installation

Install from `Test.PyPI`

```sh
pip install --extra-index-url https://test.pypi.org/simple/ DA-DAPPER
```

Install from `PyPI`

```sh
pip install DA-DAPPER
```

  - Install into specific dir (includes all of the dependencies)  
    `pip install DA-DAPPER -t MyDir`

  - Install with options  
    `pip install DA-DAPPER[Qt,MP]`

Install from local (makes installation accessible from everywhere)

```sh
pip install -e .
```
"""
