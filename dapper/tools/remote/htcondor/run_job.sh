#!/usr/bin/env bash

# Args
index=$1
rundir=$2

# Stuff that's likely to change when updating cluster/deployment images
__conda_root="/opt/anaconda"
__conda_env="py3.9.6-2021-10-14"
__py_cmd="python3"

# HTCondor (ref `UID_DOMAIN` and `SOFT_UID_DOMAIN) will set
# USER i.e. $(whoami) to "nobody".
# HOME will be unset.
# But matplotlib will try to write to $HOME/.config/matplotlib
# so we need to set it to somewhere `nobody` has permissions.
# The scratch dir (eg /var/lib/condor/execute/dir_2083) seems a safe bet:
export HOME=`pwd`

# NB: Careful so you don't start the path with :,
# which would entail adding PWD to sys.path,
# even when python is launched with a script.
export PYTHONPATH="$PWD/DAPPER"
export PYTHONPATH="$PYTHONPATH:$PWD/extra_files"

# Debug info
echo "pwd (scratch dir):" $(pwd)
echo "whoami:" $(whoami)
echo "HOME:" $HOME
echo "PYTHONPATH:" $PYTHONPATH
echo ""
echo "find . -type f -maxdepth 3:"
find . -type f -maxdepth 3
echo ""

echo "Activating conda"
__conda_setup=$($__conda_root/bin/conda shell.bash hook)
eval "$__conda_setup"
conda activate $__conda_env

echo "Python version:"
$__py_cmd -c "import sys; print(sys.version,'\n')"

echo "Mv dpr_config.yaml to PWD"
mv $PWD/extra_files/dpr_config.yaml $PWD

echo "Running experiment"
$__py_cmd $PWD/extra_files/load_and_run.py 2>&1
