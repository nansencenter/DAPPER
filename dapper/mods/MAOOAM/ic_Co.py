import numpy as np

X0=np.zeros(37)

X0[0]=1. #one dimension more to have ndim+1 to support the multiplication with the tensor (j and k are in [|0,ndim|]

#psi variables
X0[1]=0.03 # typ=A, NX0=0, Ny= 1
X0[2]=0 # typ=K, NX0=1, Ny= 1
X0[3]=0 # typ=L, NX0=1, Ny= 1
X0[4]=0.0 # typ=A, NX0=0, Ny= 2
X0[5]=0.055 # typ=K, NX0=1, Ny= 2
X0[6]=0.04 # typ=L, NX0=1, Ny= 2
X0[7]=0.0 # typ=K, NX0=2, Ny= 1
X0[8]=0.0 # typ=L, NX0=2, Ny= 1
X0[9]=0.032 # typ=K, NX0=2, Ny= 2
X0[10]=0.01 # typ=L, NX0=2, Ny= 2

#theta variables
X0[11]=0.03 # typ=A, NX0=0, Ny= 1
X0[12]=0.0 # typ=K, NX0=1, Ny= 1
X0[13]=0.0 # typ=L, NX0=1, Ny= 1
X0[14]=0.0 # typ=A, NX0=0, Ny= 2
X0[15]=0.4 # typ=K, NX0=1, Ny= 2
X0[16]=0.056 # typ=L, NX0=1, Ny= 2
X0[17]=0.0 # typ=K, NX0=2, Ny= 1
X0[18]=0.0 # typ=L, NX0=2, Ny= 1
X0[19]=0.32 # typ=K, NX0=2, Ny= 2
X0[20]=0.005 # typ=L, NX0=2, Ny= 2

#A variables
X0[21]=0.0 # NX0=0.5, Ny= 1
X0[22]=0.003 # NX0=0.5, Ny= 2
X0[23]=0.0 # NX0=0.5, Ny= 3
X0[24]=0.034 # NX0=0.5, Ny= 4
X0[25]=0.0 # NX0=1.0, Ny= 1
X0[26]=0.033 # NX0=1.0, Ny= 2
X0[27]=0.0 # NX0=1.0, Ny= 3
X0[28]=0.011 # NX0=1.0, Ny= 4

#T variables
X0[29]=0.0 # NX0=0.5, Ny= 1
X0[30]=0.06666 # NX0=0.5, Ny= 2
X0[31]=0.0 # NX0=0.5, Ny= 3
X0[32]=0.0454 # NX0=0.5, Ny= 4
X0[33]=0.0 # NX0=1.0, Ny= 1
X0[34]=0.0004 # NX0=1.0, Ny= 2
X0[35]=0.0 # NX0=1.0, Ny= 3
X0[36]=0.0 # NX0=1.0, Ny= 4
