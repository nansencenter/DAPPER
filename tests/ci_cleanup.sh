#!/bin/bash

printf "Cleaning up environments ... "  # printf avoids new lines
if [[ "$DISTRIB" == "conda" ]]; then
    # Force the env to be recreated next time, for build consistency
    source deactivate
    conda remove -p ./.venv --all --yes
    rm -rf ./.venv
fi
echo "DONE"