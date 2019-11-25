import numpy as np
import params
from params import ndim,tw,t_run,t_trans,dt
import aotensor
import integrator
from plot import *
import time
import ic_def

class bcolors:
#to color the instructions in the console
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'


print (bcolors.OKBLUE + "MODEL MAOOAM" + bcolors.ENDC)
print (bcolors.OKBLUE + "Initialization" + bcolors.ENDC)

ic_def.load_IC()
import ic
X= ic.X0
print (X)
print (bcolors.OKBLUE + "Starting the transient time evolution..." + bcolors.ENDC)

t=0.
T=time.clock()
while(t<t_trans) :
	X=integrator.step(X,t,dt)
	print (t)
	t += dt

print (bcolors.OKBLUE + "Starting the  time evolution..." + bcolors.ENDC)
#we write in the file on this section
#ndim+1 columns : 1 for time and one for the ndim coordinates
fichier = open("evol_field_python_"+str(t_trans)+" "+str(t_run)+".dat", "w")
t=0.

while (t<t_run) :
	#one step
	X=integrator.step(X,t,dt)
	print (t)
	t += dt
	if(t%(tw) <dt) :
		fichier.write(str(t)+" ")
		for i in range(1,ndim+1) :
			fichier.write(str(X[i])+" ")
		fichier.write("\n")
fichier.close()
print (bcolors.OKBLUE + "Evolution finished "+ bcolors.ENDC)

d=time.clock()-T

print (bcolors.OKBLUE + "Time clock :" + bcolors.ENDC)
print (d)

print (bcolors.OKBLUE + "Calculating mean and variance" + bcolors.ENDC)
#calculate_stat()

print (bcolors.OKBLUE + "Plot thetao2 psio2 psia1" + bcolors.ENDC)

#plotD()

