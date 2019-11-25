"""
	Principal module
	======================

	Python 3.5 implementation of the modular arbitrary-order ocean-atmosphere model MAOOAM
	This module actually contains the Heun algorithm routines.

	.. note :: The python code is available here : https://github.com/nansencenter/DAPPER/tree/max1/mods/MAOOAM

	:Example:

	>>>from maooam import *

	Global variable
	-------------------

	ic.X0
	ic.X1
	X
	t
	t_trans,t_run,tw,dt
	T

	Dependencies
	-------------------

	import numpy as np
	import params2
	from params2 import ndim,tw,t_run,t_trans,dt
	import aotensor
	import integrator
	from plot import *
	import time
	import ic_def
	import ic
"""

import numpy as np
import params2
from params2 import ndim,tw,t_run,t_trans,dt
import aotensor
import integrator
from plot import *
import time
import ic_def
import ic

print ("MODEL MAOOAM")
print ("Initialization")

# ic_def.load_IC()
X= ic.X0
print ("Starting the transient time evolution...")

t=0.
T=time.clock()
while(t<t_trans) :
	X=integrator.step(X,t,dt)
	t += dt

print ("Starting the  time evolution...")
#we write in the file on this section
#ndim+1 columns : 1 for time and one for the ndim coordinates
fichier = open("evol_field_python_"+str(t_trans)+" "+str(t_run)+".dat", "w")
t=0.

while (t<t_run) :
	#one step
	X=integrator.step(X,t,dt)
	t += dt
	if(t%(tw) <dt) :
		fichier.write(str(t)+" ")
		for i in range(ndim) :
			fichier.write(str(X[i])+" ")
		fichier.write("\n")
fichier.close()
print ("Evolution finished ")
print ("Time clock :",time.clock()-T)
# print ("Calculating mean and variance")
#calculate_stat()
# print ("Plot thetao2 psio2 psia1")
#plotD()

