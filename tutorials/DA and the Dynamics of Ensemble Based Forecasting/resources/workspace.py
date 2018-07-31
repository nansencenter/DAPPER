# CD to DAPPER folder
from IPython import get_ipython
IP = get_ipython()
if IP.magic("pwd").endswith('tutorials'):
    IP.magic("cd ..")
elif IP.magic("pwd").endswith('DA and the Dynamics of Ensemble Based Forecasting'):
    IP.magic("cd ../..")
else:
    assert IP.magic("pwd").endswith("DAPPER")

# Load DAPPER
from common import *

# Load answers
from tutorials.resources.answers import answers, show_answer

# Load widgets
from ipywidgets import *

import markdown


