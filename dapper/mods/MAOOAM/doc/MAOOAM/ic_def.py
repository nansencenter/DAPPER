"""
	Initial conditions generator module
	===================================

	This module generates initial conditions for the model if it doesn't exist with the good dimensions.
	
	It is possible to change the dimensions of the system in params.py the parameters file.
	Then delete ic.py and ic_def.py will regenerates it.

	.. note :: The python code is available here : `ic_def.py <../_modules/ic_def.html>`_ and on `Git <https://github.com/nansencenter/DAPPER/blob/max1/mods/MAOOAM/ic_def.py>`_.

	:Example:

	>>> from ic_def import load_IC
	>>> load_IC()

	Global file
	-------------------

	the file ic.py in the same directory.


	Dependancies
	-------------------

	uses the modules to know the dimensions :

	>>> from params2 import natm,noc,ndim,t_trans,t_run
	>>> from inprod_analytic import awavenum,owavenum,init_inprod

	Functions
	-------------------

	Here is the function in the module :

"""

from params2 import natm,noc,ndim,t_trans,t_run
from inprod_analytic import awavenum,owavenum,init_inprod
import os.path


def load_IC() :
	""" Check if ic.py exists, if not creates it with good dimensions and zero initial conditions """
	if (ndim==0):
		exit('Number of dimensions is 0!')
	if (os.path.exists('ic.py')):
		#no need to create one
		print ("ic already defined")
	else :
		#init_inprod()
		fichier = open("ic.py", "w")
		
		fichier.write("import numpy as np")
		fichier.write("\n")
		fichier.write("\n")
		
		fichier.write("X0=np.zeros("+str(ndim)+")")
		fichier.write("\n")
		fichier.write("\n")
		
		
		fichier.write("#psi variables")
		fichier.write("\n")
		for i in range(0,natm):
			fichier.write("X0["+str(i)+"]=0.0 # typ="+str(awavenum[i].typ)+", NX0="+str(awavenum[i].Nx)+", Ny= "+str(awavenum[i].Ny))
			fichier.write("\n")

		fichier.write("\n")
		fichier.write("#theta variables")
		fichier.write("\n")
		
		for i in range(0,natm):
			fichier.write("X0["+str(i+natm)+"]=0.0 # typ="+str(awavenum[i].typ)+", NX0="+str(awavenum[i].Nx)+", Ny= "+str(awavenum[i].Ny))
			fichier.write("\n")
			
		fichier.write("\n")
		fichier.write("#A variables")
		fichier.write("\n")
		
		for i in range(0,noc):
			fichier.write("X0["+str(i+natm+natm)+"]=0.0 # NX0="+str(owavenum[i].Nx)+", Ny= "+str(owavenum[i].Ny))
			fichier.write("\n")

		fichier.write("\n")
		fichier.write("#T variables")
		fichier.write("\n")
		
		for i in range(0,noc):
			fichier.write("X0["+str(i+natm+natm+noc)+"]=0.0 # NX0="+str(owavenum[i].Nx)+", Ny= "+str(owavenum[i].Ny))
			fichier.write("\n")
		fichier.close()

if __name__ == "__main__":
	load_IC()
