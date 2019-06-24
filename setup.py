#!/usr/bin/env python3

import setuptools, os, re

with open("README.md", "r") as fh:
    long_description = fh.read()

def filter_dirs(x):
  val  = x!='__pycache__'
  val &= x!='.DS_Store'
  val &= x!='pyks'
  val &= not x.endswith('.py')
  return val



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



setuptools.setup(

    # Basic meta
    name="DA-DAPPER",
    version=find_version("dapper", "__init__.py"),
    author="Patrick N. Raanes",
    author_email="patrick.n.raanes@gmail.com",
    description="Data Assimilation with Python: a Package for Experimental Research.",


    # Dependencies. Use pipdeptree and pipreqs tools.
    python_requires='~=3.5',
    # TODO: improve version pinning
    install_requires=[
      'scipy>=1.1',
      'matplotlib>=3.0.3',
      'ipython>=5.1',
      'tqdm>=4.18',
      'colorama>=0.3.7',
      'tabulate>=0.7.7',
      ],
    extras_require={
      'MP': [
        'threadpoolctl==1.0.0',
        'multiprocessing-on-dill>=3.5.0a4',
        'psutil',
        ],
      'Qt':  ['qtpy'],
      },
    # Tutorials:
    # -----------
    # 'jupyter>=1.0.0',
    # 'Markdown>=2.6',
    #
    # Site-packages -- implicit
    # -------------
    # numpy # comes with scipy, etc
    # cycler # comes with mpl
    # pkg_resources # comes with setuptools
    #
    # Stdlib
    # -------------
    # contextlib
    # subprocess
    # socket
    # textwrap
    # copy
    # itertools
    # os.path
    # warnings
    # re
    # traceback
    # inspect
    # signal
    # functools
    # glob
    # collections
    # termios
    # getpass
    #
    # Built-ins:
    # -------------
    # builtins
    # time
    # sys
    # msvcrt


    # File inclusion
    # Note: find_packages() only works on __init__ dirs.
    packages=setuptools.find_packages()+\
      ['dapper.da_methods','dapper.tools','dapper.mods']+\
      ['dapper.mods.'+x for x in os.listdir('dapper/mods') if filter_dirs(x)]+\
      ['dapper.mods.QG.f90'],
    package_data={
        '': ['*.mplstyle','*.txt','*.md','*.png'],
        'dapper.mods.QG.f90': ['*.txt','*.md','*.png','Makefile','*.f90'],
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
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nansencenter/DAPPER",
    keywords='data-assimilation enkf kalman-filtering state-estimation particle-filter kalman bayesian-methods bayesian-filter chaos',
    # project_urls={
        # 'Documentation': 'https://packaging.python.org/tutorials/distributing-packages/',
        # 'Funding': 'https://donate.pypi.org',
        # 'Say Thanks!': 'http://saythanks.io/to/example',
        # 'Source': 'https://github.com/pypa/sampleproject/',
        # 'Tracker': 'https://github.com/pypa/sampleproject/issues',
    # },
)


