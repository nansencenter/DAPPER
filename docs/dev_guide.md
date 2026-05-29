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

!!! note Devs should use [`uv`](https://docs.astral.sh/uv/) for reproducible installs.
    It also has high-performance dependency resolution,
    and a streamlined workflow for building and publishing to PyPI.

`uv.lock` is checked in. Use `uv sync` to install from it (DAPPER gets installed as "editable"):

```sh
uv sync --extra dev
```

To add or remove dependencies, let `uv` manage both `pyproject.toml` and `uv.lock` together:

```sh
uv add --optional lint package-to-be-installed
uv remove package-to-uninstall
```

Or edit `pyproject.toml` by hand and then run `uv lock` to update the lockfile.

## Dependency management

DAPPER being aimed at research experiments, it'd be best (reproducibility)
if we could pin all its dependencies (like an "App" and not a "library").
However, this might make it overly hard for people's to bring their models/methods.
Meanwhile, we want to run on Colab; while `!pip install <dep>` sometimes works seamlessly,
it can be slow due to dependency resolution in an already crowded environment,
it requires kernel restarting (highly undesirable) for `matplotlib` and other packages that have been pre-imported.
Therefore some leniency should be granted where possible in `pyproject.toml` (used by `pip`)
by using minimum version bounds (`>=`, if the API is expected to be stable) or
compatible release (`~=`) bounds.

Using cutting-edge dependencies is not necessary.
The main reason to upgrade is if the Colab tests break.
Again, use `uv` to keep lockfile in sync:

```sh
uv lock --upgrade-package scipy   # upgrade one package
uv lock --upgrade                 # upgrade everything
```

!!! note For `dill` specifically: also upgrade the version on any remote computing servers,
    and check whether existing serialised test data needs to be re-saved.

The workflow `.github/workflows/colab-compat.yml` tests (`basic_1`, `basic_2`) against Google's
[public Colab Docker image](https://us-docker.pkg.dev/colab-images/public/runtime).
It runs automatically on the 1st of each month, but can also be triggered via
`GitHub UI → Actions → Colab compat → Run workflow`.
To test it locally, get [Podman](https://podman.io/),
setup the `podman` VM with at least 80 GB of virtual disk (`podman machine init --disk-size 80`)
then do (is slow on first pull because ~5 GB image download):

```sh
mise run test:colab
```

## Run tests

By default, only `doctests` are run when executing `pytest`.
To run the main tests, do this:

```sh
pytest tests
```

You can also append `test_plotting_interactive.py` for example,
which is otherwise ignored for being slow.

To test across multiple Python versions, either rely on CI (already in place),
or do `uv run --isolated --python 3.13 --extra test pytest tests`.  
In combination with a task runner, like `mise` (see `mise.toml`),
this does away with any need for `tox`.

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
ty check
```

You may also want to display linting issues in your editor as you code.
Both `ruff` and `ty` ship with language server support for in-editor diagnostics.

## Writing documentation

The documentation is built with `mkdocs` and should be written in [markdown syntax](https://www.markdownguide.org/cheat-sheet/).
You can preview the rendered html docs by running

```sh
mkdocs serve
```

- Temporarily disable `mkdocs-jupyter` in `properdocs.yml` to speed up build reloads.
- Set `validation: unrecognized_links: warn` to get warnings about linking issues.
- Filter [spurious pandoc/version warnings](https://github.com/danielfrg/mkdocs-jupyter/issues/231):
  (TODO 4: checkup on bug resolved?):

  ```sh
  mkdocs serve 2>&1 | grep -Ev '^\[WARNING\] Div at|^[0-9]+\.[0-9]+\.[0-9]'
  ```

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
    by setting `properdocs.yml: validation: absolute_links: relative_to_docs`.
    A different workaround is the [`mkdocs-site-url` plugin](https://github.com/OctoPrint/mkdocs-site-urls).

    !!! tip "Either way"
        It will not be link that your editor can follow to the relevant markdown file
        (unless you create a symlink in your file system root?)
        nor will GitHub's internal markdown rendering manage to make sense of it,
        so my advise is not to use absolute links.

### Linking to headers/anchors

Thanks to the `autorefs` plugin,
links to **headings** (including page titles) don't even require specifying the page path!
Syntax: `[visible label][link]` i.e. double pairs of *brackets*. Shorthand: `[link][]`.
!!! info
    - Clearly, non-unique headings risk being confused with others in this way.
    - The link (anchor) must be lowercase!

This facilitates linking to

- **API (code reference)** items.
  For example, `[`da_methods.ensemble`][]`,
  where the backticks are optional (makes the link *look* like a code reference).
- **References**. For example `[`bocquet2016`][]`,

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

Bump version number in `dapper/__init__.py`

Tag

```sh
git tag -a v$(python -c "import dapper; print(dapper.__version__)") -m 'My description'
git push origin --tags
```

Build

```sh
python -m build
```

Upload to PyPI

```sh
uv publish dist/*
```

Upload to Test.PyPI

```sh
uv publish --publish-url https://test.pypi.org/legacy/ dist/*
```

Credentials are read from `~/.pypirc` or passed via `--token`.

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

