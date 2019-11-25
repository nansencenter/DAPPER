"""
	Integration module
	======================

	Module with the integration
	This module actually contains the Heun algorithm routines.

	.. note :: The python code is available here : https://github.com/nansencenter/DAPPER/tree/max1/mods/MAOOAM

	:Example:

	>>> from integrator import step
	>>> step(y,t,dt)

	Global variable
	-------------------

	aotensor

	Dependencies
	-------------------

	>>> from mods.MAOOAM.params2 import ndim
	>>> import mods.MAOOAM.aotensor as aotensor
	>>> import numpy as np
	>>> from scipy.sparse import dok_matrix
	>>> from scipy.sparse import csr_matrix
	>>> import time

	Functions
	-------------------

	* sparse_mul3
	* tendencies
	* step
"""

from params2 import ndim
import aotensor as aotensor
import numpy as np
from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix
import time

aotensor=aotensor.init_aotensor()


def sparse_mul3(arr) :
	""" Calculate for each i the sums on j,k of the product
	tensor(i,j,k)* arr(j) * arr(k) """
	if np.ndim(arr) is 1:
		arr = arr.reshape((1,len(arr)))
	n=arr.shape[0]
	arr = np.hstack((np.array([[1]*n]).reshape(n,1),arr))
	res=np.zeros((n,ndim+1))
	for (i,j,k,v) in aotensor :
		res[:,i]=res[:,i]+v*arr[:,j]*arr[:,k]
	if res.shape[0] is 1:
		res = res.squeeze()
		return res[1:]
	return res[:,1:]



def tendencies(y) :
	""" Calculate the tendencies thanks to the product of the tensor and the vector y"""
	return sparse_mul3(y)



def step(y,t,dt) :
	""" Heun method integration"""
	n=y.shape[0]
	buf_f0=np.zeros((n,ndim+1))
	buf_f1=np.zeros((n,ndim+1))
	buf_y1=np.zeros((n,ndim+1))

	buf_f0=tendencies(y)
	buf_y1= y + dt * buf_f0
	buf_f1=tendencies(buf_y1)
	Y=y + 0.5 * (buf_f0 + buf_f1) * dt
	return Y

if __name__ == "__main__":
	import ic
	X=ic.X0
	#for i in range(0,80):
	X=sparse_mul3(X)
	print (sparse_mul3(X))

