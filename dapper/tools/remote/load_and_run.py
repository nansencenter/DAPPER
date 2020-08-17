"""Loads PWD/xp_{var,com} and calls run_experiment()."""

from dapper import *

# Load
with open("xp.com", "rb") as xp_com: com = dill.load(xp_com)
with open("xp.var", "rb") as xp_com: var = dill.load(xp_com)

# Startup
script = com.pop("exec")
exec(script)

# Run
result = run_experiment(var['xp'], None, Path("."), **com)
