import numpy as np

X0=np.zeros(37)

X0[0]=1. #one dimension more to have ndim+1 to support the multiplication with the tensor (j and k are in [|0,ndim|]

#psi variables
X0[1]=-0.00309# typ=A, NX0=0, Ny= 1
X0[2]= 0.24808# typ=K, NX0=1, Ny= 1
X0[3]=-0.02521 # typ=L, NX0=1, Ny= 1
X0[4]=0.01096 # typ=A, NX0=0, Ny= 2
X0[5]=0.15825 # typ=K, NX0=1, Ny= 2
X0[6]= -0.09092# typ=L, NX0=1, Ny= 2
X0[7]= -0.05916# typ=K, NX0=2, Ny= 1
X0[8]= 0.01876# typ=L, NX0=2, Ny= 1
X0[9]= -0.03299# typ=K, NX0=2, Ny= 2
X0[10]= -0.11928# typ=L, NX0=2, Ny= 2

#theta variables
X0[11]= -0.02049# typ=A, NX0=0, Ny= 1
X0[12]= -0.03588# typ=K, NX0=1, Ny= 1
X0[13]= 0.06035# typ=L, NX0=1, Ny= 1
X0[14]= -0.16648# typ=A, NX0=0, Ny= 2
X0[15]= -0.07002# typ=K, NX0=1, Ny= 2
X0[16]= 0.11514# typ=L, NX0=1, Ny= 2
X0[17]= 0.18573# typ=K, NX0=2, Ny= 1
X0[18]= -0.15112# typ=L, NX0=2, Ny= 1
X0[19]= 0.06448# typ=K, NX0=2, Ny= 2
X0[20]= -0.09806# typ=L, NX0=2, Ny= 2

#A variables
X0[21]= -0.08569# NX0=0.5, Ny= 1
X0[22]= -0.08719# NX0=0.5, Ny= 2
X0[23]= -0.04225# NX0=0.5, Ny= 3
X0[24]= 0.09964# NX0=0.5, Ny= 4
X0[25]= 0.07124# NX0=1.0, Ny= 1
X0[26]= 0.00591# NX0=1.0, Ny= 2
X0[27]= -0.03633# NX0=1.0, Ny= 3
X0[28]= 0.00033# NX0=1.0, Ny= 4

#T variables
X0[29]= -0.01059# NX0=0.5, Ny= 1
X0[30]= 0.07931# NX0=0.5, Ny= 2
X0[31]= -0.06316# NX0=0.5, Ny= 3
X0[32]= -0.00062# NX0=0.5, Ny= 4
X0[33]= -0.01011# NX0=1.0, Ny= 1
X0[34]= -0.00523# NX0=1.0, Ny= 2
X0[35]= 0.02492# NX0=1.0, Ny= 3
X0[36]=0.01977 # NX0=1.0, Ny= 4
