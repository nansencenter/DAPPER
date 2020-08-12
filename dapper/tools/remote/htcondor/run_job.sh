#!/usr/bin/env bash

# Args
index=$1
rundir=$2


# NB: USER $(whoami) is "nobody", so cannot use "export HOME=/home/pnr".
# (ref `UID_DOMAIN` and `SOFT_UID_DOMAIN)
#export HOME=`pwd` # scratch dir, eg /var/lib/condor/execute/dir_2083
export SCRATCH=`pwd`


# Debug
echo "pwd:" $(pwd)
echo "SCRATCH:" $SCRATCH
echo "whoami:" $(whoami)
echo ""


echo "Activating conda"
__conda_setup=$(/opt/anaconda3/bin/conda shell.bash hook)
eval "$__conda_setup"
conda activate dpr_2020-08-07
python -c "import sys; print('Python v.', sys.version)"


echo "Running experiment"
export PYTHONPATH="$SCRATCH/DAPPER"
python DAPPER/dapper/tools/remote/load_and_run.py 2>&1
