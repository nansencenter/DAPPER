#!/usr/bin/env bash

# Args
index=$1
rundir=$2

# NB: USER $(whoami) is "nobody", so cannot use "export HOME=/home/pnr".
# (ref `UID_DOMAIN` and `SOFT_UID_DOMAIN)
#export HOME=`pwd` # scratch dir, eg /var/lib/condor/execute/dir_2083
export SCRATCH=`pwd`

# NB: Careful so you don't start the path with :,
# which would entail adding PWD to sys.path,
# even when python is launched w/ a script.
export PYTHONPATH="$SCRATCH/DAPPER"
export PYTHONPATH="$PYTHONPATH:$SCRATCH/extra_files"


# Debug info
echo "pwd:" $(pwd)
echo "SCRATCH:" $SCRATCH
echo "whoami:" $(whoami)
echo "PYTHONPATH:" $PYTHONPATH
echo "find . -type f -maxdepth 3:"
find . -type f -maxdepth 3
echo ""


echo "Activating conda"
__conda_setup=$(/opt/anaconda3/bin/conda shell.bash hook)
eval "$__conda_setup"
conda activate dpr_2020-08-07
python -c "import sys; print('Python v.', sys.version,'\n')"


echo "Running experiment"
python load_and_run.py 2>&1
