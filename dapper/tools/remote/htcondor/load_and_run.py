"""Loads PWD/xp_{var,com} and calls run_experiment()."""

from dapper import *

# Load
with open("xp_com", "rb") as FILE: com = dill.load(FILE)
with open("xp_var", "rb") as FILE: var = dill.load(FILE)

# Startup
script = com.pop("exec")
exec(script)

# Run
result = run_experiment(var['xp'], **com, savepath="./")
