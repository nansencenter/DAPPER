#!/bin/bash
# This script is to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# This script is inspired by Scikit-Learn (http://scikit-learn.org/)

set -e

if [[ "$DISTRIB" == "conda" ]]; then
    if [[ -f "$HOME/miniconda/bin/conda" ]]; then
        echo "Skip install conda [cached]"
    else
        # By default, travis caching mechanism creates an empty dir in the
        # beginning of the build, but conda installer aborts if it finds an
        # existing folder, so let's just remove it:
        rm -rf "$HOME/miniconda"

        # Use the miniconda installer for faster download / install of conda
        # itself
        if [ "$RUNNER_OS" != "macOS" ]; then
            wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
                -O miniconda.sh
        else
            wget http://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh \
                -O miniconda.sh
        fi
        chmod +x miniconda.sh && ./miniconda.sh -b -p $HOME/miniconda
    fi
    export PATH=$HOME/miniconda/bin:$PATH
    # Make sure to use the most updated version
    conda update --yes conda

    # Configure the conda environment and put it in the path using the
    # provided versions
    # (prefer local venv, since the miniconda folder is cached)
    conda create -p ./venv --yes python=${PYTHON_VERSION} pip virtualenv
    source activate ./venv
fi

# For all
pip install -U pip setuptools

if [[ "$COVERAGE" == "true" ]]; then
    pip install -U coverage coveralls
fi
