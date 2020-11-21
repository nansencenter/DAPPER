"""
## Documentation generation

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
This generates a lot of warnings of the type
"UserWarning: __pdoc__-overriden key ... does not exist in module".
AFAICT that's fine. https://github.com/pdoc3/pdoc/issues/206
Alternative: Insert this at top of each script to exclude
and run pdoc with --skip-errors.

>>> if __name__ != "__main__":
>>>     raise RuntimeError("This module may only be run as script.")

### Hosting:
Push updated docs to github.
In the main github settings of the repo,
go to the "GitHub Pages" section,
and set the source to the docs folder.

## TODOs

Minor:

- Badges
- Add to publication list
- Publish version

Major:

- Improve docs
- Make autoscaler.py work also for other users
  (I dont know if it's condor_- that outputs user-specific numbers,
  or glcoud resize command that is user-specific, or what.)
- Pause after Enter doesn't work on Ubuntu?

- Make superclasses for the filter, smoother, and iterative smoother.
- Simplify and/or generalize time management?

- k,kObs only, yielded by ticker
- Change KObs to KObs-1
- Note (somewhere) the implicit dependence on t=0 being special

- Simplify and/or improve cov matrix stuff.

- For xpList, could also subclass collections.UserList,
  although I don't know if it would be any better
  <https://stackoverflow.com/q/25464647>

Note: 

There are also TODO labels scattered in the code,
including a priority number (1-9, lower is more important).

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
