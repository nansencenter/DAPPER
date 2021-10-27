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
DOCLINES = __doc__.split('\n')

# Dependencies
INSTALL_REQUIRES = [
    'scipy>=1.1',
    'ipython>=5.1',
    'jedi<0.18',  # ipython/issues/12740
    'jupyter',
    'matplotlib~=3.2.2',
    #    >=3.1 to avoid Mac's framework-build issues.
    #    But 3.4 does not work on Colab.
    'mpl-tools==0.2.50',
    'tqdm~=4.31',
    'pyyaml',
    'ipdb',
    'colorama~=0.4.1',
    'tabulate~=0.8.3',
    'dill==0.3.2',  # >=0.3.1.1 for dataclass. Pin vers. to equal GCP.
    'patlib==0.3.5',
    'struct-tools==0.2.5',
    # TODO 5: replace by p-tqdm?
    'multiprocessing-on-dill==3.5.0a4',
    'threadpoolctl==1.0.0',
]

EXTRAS = {
    'Qt': ['PyQt5', 'qtpy'],
    'dev': ['line_profiler', 'pre-commit'],
    'test': ['tox', 'coverage>=5.1', 'pytest',
             'pytest-cov', 'pytest-sugar', 'pytest-benchmark',
             'pytest-clarity', 'pytest-xdist', 'pytest-timeout'],
    'lint': ['flake8', 'flakehell'],
    # 'flake8-docstrings', 'flake8-bugbear', 'flake8-comprehensions'],
    'build': ['twine', 'pdoc3', 'jupytext'],
}
EXTRAS['dev'] += EXTRAS['test'] + EXTRAS['lint'] + EXTRAS['build']


def find_version(*file_paths):
    """Get version by regex'ing a file."""
    # Source: packaging.python.org/guides/single-sourcing-package-version

    def read(*parts):
        here = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(here, *parts), 'r') as fp:
            return fp.read()

    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# 'python setup.py publish' shortcut.
if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist bdist_wheel')
    os.system('twine upload dist/*')
    sys.exit(0)


setup(
    # Basic meta
    name="dapper",
    version=find_version("dapper", "__init__.py"),
    author="Patrick N. Raanes",
    author_email="patrick.n.raanes@gmail.com",
    description=DOCLINES[0],
    long_description='\n'.join(DOCLINES[2:]),
    long_description_content_type="text/markdown",

    # >=3.5 for @.
    # >=3.6 for mpl>=3.1.
    # >=3.7 for dataclass, capture_output, dict ordering, np>=1.20.
    # ==3.7 for Colab
    # ==3.9 for the DAPPER/GCP cluster, since dill isn't compat. across versions.
    python_requires='>=3.7',
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS,

    packages=find_packages(),
    py_modules=["examples.basic_1", "examples.basic_2", "examples.basic_3"],
    package_data={
        '': ['*.txt', '*.md', '*.png', '*.yaml'],
        'dapper.mods.QG.f90': ['*.txt', '*.md', '*.png', 'Makefile', '*.f90'],
    },

    # Detailed meta
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',

        'Programming Language :: Python :: 3',

        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords=("data-assimilation enkf kalman-filtering"
              " state-estimation particle-filter kalman"
              " bayesian-methods bayesian-filter chaos"),
    project_urls={
        'Documentation': 'https://nansencenter.github.io/DAPPER/',
        'Source': 'https://github.com/nansencenter/DAPPER',
        'Tracker': 'https://github.com/nansencenter/DAPPER/issues',
        # 'Funding': 'https://donate.pypi.org',
        # 'Say Thanks!': 'http://saythanks.io/to/example',
    },
    # url="https://github.com/nansencenter/DAPPER",
)
