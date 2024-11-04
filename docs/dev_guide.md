---
hide:
- navigation
---

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

Include the dev tools as part of the installation (detailed in the README):

```sh
pip install -e '.[dev]'
```

PS: *If* you want to be able to use static analysis tools (`pyright`) with dapper
all the while working *from outside its directory*,
you should also **append** `--config-settings editable_mode=compat` to the above command.
Ref [pyright doc](https://github.com/microsoft/pyright/blob/main/docs/import-resolution.md#editable-installs)
and [pyright issue](https://github.com/microsoft/pylance-release/issues/78).
Alternatively, there is the `extraPaths` setting.

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
ruff check --output-format=concise
```

You may also want to display linting issues in your editor as you code.

## Writing documentation

The documentation is built with `mkdocs` and should be written in [markdown syntax](https://www.markdownguide.org/cheat-sheet/).
You can preview the rendered html docs by running

```sh
mkdocs serve
```

- Temporarily disable `mkdocs-jupyter` in `mkdocs.yml` to speed up build reloads.
- Set `validation: unrecognized_links: warn` to get warnings about linking issues.

Docstrings should be written in the [style of numpy](https://mkdocstrings.github.io/griffe/reference/docstrings/#numpydoc-style).
Additional details on the documentation system are collected in the following subsection.

### Linking to pages

You should use relative page links, including the `.md` extension.
For example, `[link label](sibling-page.md)`.

The following works, but does not get validated! `[link label](../sibling-page)`

!!! hint "Why not absolute links?"

    The downside of relative links is that if you move/rename source **or** destination,
    then they will need to be changed, whereas only the destination needs be watched
    when using absolute links.

    Previously, absolute links were not officially supported by MkDocs, meaning "not modified at all".
    Thus, if made like so `[label](/DAPPER/references)`,
    i.e. without `.md` and including `/DAPPER`,
    then they would **work** (locally with `mkdocs serve` and with GitHub hosting).
    Since [#3485](https://github.com/mkdocs/mkdocs/pull/3485) you can instead use `[label](/references)`
    i.e. omitting `DAPPER` (or whatever domain sub-dir is applied in `site_url`)
    by setting `mkdocs.yml: validation: absolute_links: relative_to_docs`.
    A different workaround is the [`mkdocs-site-url` plugin](https://github.com/OctoPrint/mkdocs-site-urls).

    !!! tip "Either way"
        It will not be link that your editor can follow to the relevant markdown file
        (unless you create a symlink in your file system root?)
        nor will GitHub's internal markdown rendering manage to make sense of it,
        so my advise is not to use absolute links.

### Linking to headers/anchors

Thanks to the `autorefs` plugin,
links to **headings** (including page titles) don't even require specifying the page path!
Syntax: `[visible label][link]` i.e. double pairs of _brackets_. Shorthand: `[link][]`.
!!! info
    - Clearly, non-unique headings risk being confused with others in this way.
    - The link (anchor) must be lowercase!

This facilitates linking to

- **API (code reference)** items.
  For example, ``[`da_methods.ensemble`][]``,
  where the backticks are optional (makes the link _look_ like a code reference).
- **References**. For example ``[`bocquet2016`][]``,

### Docstring injection

Use the following syntax to inject the docstring of a code object.

```markdown
::: da_methods.ensemble
```

But we generally don't do so manually.
Instead it's taken care of by the reference generation via `docs/gen_ref_pages.py`.

### Including other files

The `pymdown` extension ["snippets"](https://facelessuser.github.io/pymdown-extensions/extensions/snippets/#snippets-notation)
enables the following syntax to include text from other files.

`--8<-- "/path/from/project/root/filename.ext"`

### Adding to the examples

Example scripts are very useful, and contributions are very desirable.  As well
as showcasing some feature, new examples should make sure to reproduce some
published literature results.  After making the example, consider converting
the script to the Jupyter notebook format (or vice versa) so that the example
can be run on Colab without users needing to install anything (see
`docs/examples/README.md`). This should be done using the `jupytext` plug-in (with
the `lightscript` format), so that the paired files can be kept in synch.

### Bibliography

In order to add new references,
insert their bibtex into `docs/bib/refs.bib`,
then run `docs/bib/bib2md.py`
which will format and add entries to `docs/references.md`
that can be cited with regular cross-reference syntax, e.g. `[bocquet2010a][]`.

### Hosting

The above command is run by a GitHub Actions workflow whenever
the `master` branch gets updated.
The `gh-pages` branch is no longer being used.
Instead [actions/deploy-pages](https://github.com/actions/deploy-pages)
creates an artefact that is deployed to Github Pages.

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
where `~/.pypirc` contains

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

