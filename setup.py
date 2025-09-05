#!/usr/bin/env python3
"""DAPPER benchmarks the performance of data assimilation (DA) methods.

It is usually best to install from source (github),
so that you the code is readily available to play with.
See full README on [github](https://github.com/nansencenter/DAPPER).
"""

import os
import re
import sys

from setuptools import find_packages, setup

# with open("README.md", "r") as fh:
#     long_description = fh.read()
DOCLINES = __doc__.split("\n")

# Dependencies
# Why pin?: https://github.com/nansencenter/DAPPER/issues/41#issuecomment-1381616971
INSTALL_REQUIRES = [
    # NB: Colab comes with several packages pre-installed, and we might want to avoid
    # re-installing these (for compatibility, and startup time).
    # Some are even pre-imported, and for these `!pip install` won't take effect
    # (restarting kernel is workaround but we want to be able to just "run all").
    # ⇒ try not to strictly pin.
    # TODO 4: implement CI with colab environment: https://github.com/googlecolab/backend-info/blob/main/pip-freeze.txt
    "scipy>=1.14",
    "numpy~=2.0",
    "matplotlib>=3.10",
    "pyyaml>=6.0.2",
    "ipython>=7.34",
    "ipdb",
    "jupyter",
    "notebook<7",  # only nbclassic supports nbAgg (liveplotting in Jupyter) backend
    "mpl-tools==0.4.1",
    "tqdm~=4.67",
    "colorama~=0.4.1",
    "tabulate~=0.8.3",
    "pathos~=0.3",
    "dill==0.3.8",  # NB: must be same on remote computing servers
    "patlib==0.3.7",
    "struct-tools==0.2.5",
    "threadpoolctl>=3.0.0,<4.0.0",
]

EXTRAS = {
    "Qt": ["PyQt5", "qtpy"],
    "debug": ["line_profiler", "pre-commit"],
    "test": [
        "tox",
        "coverage>=5.1",
        "pytest",
        "pytest-cov",
        "pytest-sugar",
        "pytest-benchmark",
        "pytest-clarity",
        "pytest-xdist",
        "pytest-timeout",
    ],
    "lint": ["ruff"],
    "doc": [
        "mkdocs-material",
        "mkdocstrings",
        "mkdocstrings-python",
        "mkdocs-gen-files",
        "mkdocs-literate-nav",
        "mkdocs-section-index",
        "mkdocs-glightbox",
        "mkdocs-jupyter",
        "pybtex",
    ],
    # 'flake8-docstrings', 'flake8-bugbear', 'flake8-comprehensions'],
    "build": [
        "twine",
        "jupytext<=1.15",  # menu item disappeared for 1.16
    ],
}
EXTRAS["dev"] = (
    EXTRAS["debug"] + EXTRAS["test"] + EXTRAS["lint"] + EXTRAS["build"] + EXTRAS["doc"]
)


def find_version(*file_paths):
    """Get version by regex'ing a file."""
    # Source: packaging.python.org/guides/single-sourcing-package-version

    def read(*parts):
        here = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(here, *parts)) as fp:
            return fp.read()

    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# 'python setup.py publish' shortcut.
if sys.argv[-1] == "publish":
    os.system("python setup.py sdist bdist_wheel")
    os.system("twine upload dist/*")
    sys.exit(0)


setup(
    # Basic meta
    name="dapper",
    version=find_version("dapper", "__init__.py"),
    author="Patrick N. Raanes",
    author_email="patrick.n.raanes@gmail.com",
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    long_description_content_type="text/markdown",
    # >=3.5 for @.
    # >=3.6 for mpl>=3.1.
    # >=3.7 for dataclass, capture_output, dict ordering, np>=1.20.
    # ==3.7 for Colab
    # ==3.9 for the DAPPER/GCP cluster, since dill isn't compat. across versions.
    python_requires=">=3.12",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS,
    packages=find_packages(),
    # py_modules=["examples.basic_1", "examples.basic_2", "examples.basic_3"],
    package_data={
        "": ["*.txt", "*.md", "*.png", "*.yaml"],
        "dapper.mods.QG.f90": ["*.txt", "*.md", "*.png", "Makefile", "*.f90"],
    },
    # Detailed meta
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords=(
        "data-assimilation enkf kalman-filtering"
        " state-estimation particle-filter kalman"
        " bayesian-methods bayesian-filter chaos"
    ),
    project_urls={
        "Documentation": "https://nansencenter.github.io/DAPPER/",
        "Source": "https://github.com/nansencenter/DAPPER",
        "Tracker": "https://github.com/nansencenter/DAPPER/issues",
        # 'Funding': 'https://donate.pypi.org',
        # 'Say Thanks!': 'http://saythanks.io/to/example',
    },
    # url="https://github.com/nansencenter/DAPPER",
)
