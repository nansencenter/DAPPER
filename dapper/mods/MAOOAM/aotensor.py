from mods.MAOOAM.params2 import *
from mods.MAOOAM.inprod_analytic import *
import numpy as np
from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix
real_eps = 2.2204460492503131e-16

t=np.zeros( ((ndim+1),(ndim+1),(ndim+1)),dtype=float)


def psi(i) :
	return i
def theta(i):
	return i+ natm
def A(i):
	return i+2*natm
def T(i) :
	return i+2*natm+noc
def kdelta(i,j) :
	if (i==j) :
		return 1
	else :
		return 0
		

def compute_aotensor() :
	#func est sous la forme f(i,j,k,v)
	t[theta(1),0,0] = (Cpa / (1 - atmos.a[0,0] * sig0))
	for i in range(1, natm+1) :
		for j in range(1, natm+1) :
			t[psi(i)  ,psi(j)  ,0] = -(((atmos.c[(i-1),(j-1)] * betp) / atmos.a[(i-1),(i-1)])) - (kd * kdelta((i-1),(j-1))) / 2
			t[theta(i),psi(j)  ,0] = (atmos.a[(i-1),(j-1)] * kd * sig0) / (-2 + 2 * atmos.a[(i-1),(i-1)] * sig0)
			t[psi(i)  ,theta(j),0] = (kd * kdelta((i-1),(j-1))) / 2
			t[theta(i),theta(j),0] = (-((sig0 * (2. * atmos.c[(i-1),(j-1)] * betp + atmos.a[(i-1),(j-1)] * (kd + 4. * kdp))))+ 2. * (LSBpa +1. * Lpa)* kdelta((i-1),(j-1))) / (-2. + 2. * atmos.a[(i-1),(i-1)] * sig0)
			#I change sc to 1. (-((sig0 * (2. * atmos.c[i,j] * betp + atmos.a[i,j] * (kd + 4. * kdp))))+ 2. * (LSBpa + sc * Lpa)* kdelta(i,j)) / (-2. + 2. * atmos.a[i,i] * sig0)"""
			for k in range(1, natm+1) :
				t[psi(i)  ,psi(j)  ,psi(k)]  = -(atmos.b[(i-1),(j-1),(k-1)] / atmos.a[(i-1),(i-1)])
				t[psi(i)  ,theta(j),theta(k)] =-((atmos.b[(i-1),(j-1),(k-1)] / atmos.a[(i-1),(i-1)]))
				t[theta(i),psi(j)  ,theta(k)]= (atmos.g[(i-1),(j-1),(k-1)] - atmos.b[(i-1),(j-1),(k-1)] * sig0) / (-1 + atmos.a[(i-1),(i-1)] *sig0)
				t[theta(i),theta(j),psi(k)]  =(atmos.b[(i-1),(j-1),(k-1)] * sig0) / (1 - atmos.a[(i-1),(i-1)] * sig0)
		for j in range(1, noc+1) :
			t[psi(i)  ,A(j),0] = kd * atmos.d[(i-1),(j-1)] / (2 * atmos.a[(i-1),(i-1)])
			t[theta(i),A(j),0] = kd * (atmos.d[(i-1),(j-1)] * sig0) / (2 - 2 * atmos.a[(i-1),(i-1)] * sig0)
			t[theta(i),T(j),0] = atmos.s[(i-1),(j-1)] * (2 * LSBpo + Lpa) / (2 - 2 * atmos.a[(i-1),(i-1)] * sig0)
	for i in range(1, noc+1) :
		for j in range(1, natm+1) :
			t[A(i),psi(j),0]   = ocean.K[(i-1),(j-1)] * dp / (ocean.M[(i-1),(i-1)] + G)
			t[A(i),theta(j),0] = -(ocean.K[(i-1),(j-1)]) * dp / (ocean.M[(i-1),(i-1)] + G)
		for j in range(1, noc+1) :
			t[A(i),A(j),0] = -((ocean.N[(i-1),(j-1)] * betp + ocean.M[(i-1),(i-1)] * (rp + dp) * kdelta((i-1),(j-1)))) / (ocean.M[(i-1),(i-1)] + G)
			for k in range(1, noc+1) :
				t[A(i),A(j),A(k)] = -(ocean.C[(i-1),(j-1),(k-1)]) / (ocean.M[(i-1),(i-1)] + G)

	for i in range(1, noc+1) :
		t[T(i),0,0] = Cpo * ocean.W[(i-1),0]
		for j in range(1, natm+1) :
			t[T(i),theta(j),0] = ocean.W[(i-1),(j-1)] * (2 * 1. * Lpo + sbpa)
			#I change sc to 1. ocean.W[i,j] * (2 * sc * Lpo + sbpa)
		for j in range(1, noc+1) :
			t[T(i),T(j),0] = -((Lpo + sbpo)) * kdelta((i-1),(j-1))
			for k in range(1, noc) :
				t[T(i),A(j),T(k)] = -(ocean.O[(i-1),(j-1),(k-1)])

def coeff(i,j,k,v) :
	if( abs(v) >= real_eps) :
		if (j<=k) :
			t[i,j,k]=v
		else :
			t[i,k,j]=v
			

def simplify() :
	for i in range(1,ndim+1):
		for j in range(0,ndim+1):
			for k in range(0,j):
				if t[i,j,k]!=0 :
					t[i,k,j]=t[i,j,k]+t[i,k,j]
					t[i,j,k]=0.



def init_aotensor() :
	
	init_inprod()
	compute_aotensor()
	simplify()
	global tensor_liste
	global tensor
	tensor_liste=[]
	for i in range(0,ndim+1):
		Xbis=csr_matrix(t[i])
		#Xbis.eliminate_zeros()
		#print (Xbis)
		X=Xbis
		tensor_liste.append(X)
		fichier = open("mods/MAOOAM/tensor/tenseur"+str(i)+".dat", "w")
		Y=Xbis.nonzero()
		for m in range(0,len(Y[0])) :
			j=Y[0][m]
			k=Y[1][m]
			fichier.write("("+str(j)+","+str(k)+"):"+str(Xbis[j,k])+"\n")
		fichier.close()
	tensor=np.array(tensor_liste)
	
	global aotensor
	aotensor=[]
	for i in range(1,ndim+1) :
		X=tensor[i].nonzero()
		for m in range(0,tensor[i].nnz) :
			j=X[0][m]
			k=X[1][m]
			aotensor.append((i,j,k, tensor[i][j,k]))
	return aotensor

		

	
