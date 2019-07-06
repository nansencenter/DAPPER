#!/usr/bin/env python3

import setuptools, os, re

# with open("README.md", "r") as fh:
    # long_description = fh.read()
long_description = """DAPPER is a set of templates for benchmarking the performance of data assimilation (DA).
See full description at github.com/nansencenter/DAPPER.
"""

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


    # Dependencies. Use pipdeptree and pipreqs tools to list.
    # We pin smaller libraries (coz stackoverflow.com/a/28510546),
    # but allow ranges for bigger ones (coz stackoverflow.com/a/43421973).
    python_requires='~=3.6', # (need >=3.6 for mpl 3.1)
    install_requires=[
      'scipy>=1.1',
      'ipython>=5.1',
      'matplotlib~=3.1', # >=3.1 to avoid Mac's framework-build issues
      'tqdm~=4.31',
      'colorama~=0.4.1',
      'tabulate~=0.8.3',
      ],
    # Optional
    extras_require={
      'MP': [
        'threadpoolctl==1.0.0',
        'multiprocessing-on-dill==3.5.0a4',
        'psutil',
        ],
      'Qt':  ['PyQt5','qtpy'],
      },
    # Other packages used, but not explicitly required
    #
    # Already implicitly required:
    # ----------------------------
    # numpy # comes with scipy, etc
    # cycler # comes with mpl
    # pkg_resources # comes with setuptools
    #
    # Stdlib
    # ----------------------------
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
    # ----------------------------
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
        '': ['dpr_config.ini','*.mplstyle','*.txt','*.md','*.png'],
        'dapper.mods.QG.f90': ['*.txt','*.md','*.png','Makefile','*.f90'],
    },

    py_modules=["example_1","example_2","example_3"],

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


