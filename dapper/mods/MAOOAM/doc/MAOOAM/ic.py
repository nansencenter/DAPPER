"""
        Initial conditions module
        =========================

        This file defines the initial conditions of the model. To be deleted if the dimensions are changed.

        .. note :: The Code is available on `Git <https://github.com/nansencenter/DAPPER/blob/max1/mods/MAOOAM/ic.py>`_.

        :Example:

        >>> from ic import X1
        >>> from ic import X0

        Global variables (state vectors)
        --------------------------------

        * X1 ic computed with python after 1000 years with step time of 0.1 and GRL set of parameters
        * X0 random (non-null) initial conditions.

        Dependencies
        ------------

        >>> import numpy as np

"""
import numpy as np
X1=np.zeros(36)
#psi variables
X1[0]= 0.044407409655     # typ=A, NX0=0, Ny= 1
X1[1]= 0.0213038976163    # typ=K, NX0=1, Ny= 1
X1[2]= 0.0395841278924    # typ=L, NX0=1, Ny= 1
X1[3]= 0.021479220755     # typ=A, NX0=0, Ny= 2
X1[4]= 0.0259565877263    # typ=K, NX0=1, Ny= 2
X1[5]=-0.000808394007743  # typ=L, NX0=1, Ny= 2
X1[6]= 0.0162785217921    # typ=K, NX0=2, Ny= 1
X1[7]= 0.0135781336365    # typ=L, NX0=2, Ny= 1
X1[8]= 0.00481734414887   # typ=K, NX0=2, Ny= 2
X1[9]=-0.0282931570559    # typ=L, NX0=2, Ny= 2

#theta variables
X1[10]= 0.0392889233995   # typ=A, NX0=0, Ny= 1
X1[11]= 0.0214746401094   # typ=K, NX0=1, Ny= 1
X1[12]= 0.00666175087627  # typ=L, NX0=1, Ny= 1
X1[13]= 0.00888581734515  # typ=A, NX0=0, Ny= 2
X1[14]= 0.0121927084913   # typ=K, NX0=1, Ny= 2
X1[15]= 0.00112473801586  # typ=L, NX0=1, Ny= 2
X1[16]= 0.00321828848754  # typ=K, NX0=2, Ny= 1
X1[17]= 0.00915025970638  # typ=L, NX0=2, Ny= 1
X1[18]=-0.00311478827961  # typ=K, NX0=2, Ny= 2
X1[19]=-0.012961505861    # typ=L, NX0=2, Ny= 2

#A variables
X1[20]= 3.47065339287e-06 # NX0=0.5, Ny= 1
X1[21]= 3.8208710196e-05  # NX0=0.5, Ny= 2
X1[22]= 2.70101289878e-06 # NX0=0.5, Ny= 3
X1[23]= 7.26400542358e-07 # NX0=0.5, Ny= 4
X1[24]=-2.18591496349e-07 # NX0=1.0, Ny= 1
X1[25]=-1.61491684347e-06 # NX0=1.0, Ny= 2
X1[26]= 2.89874536568e-06 # NX0=1.0, Ny= 3
X1[27]= 6.36277279258e-07 # NX0=1.0, Ny= 4

#T variables
X1[28]= 0.000637509625393 # NX0=0.5, Ny= 1
X1[29]= 0.216459954971    # NX0=0.5, Ny= 2
X1[30]=-0.00180715409124  # NX0=0.5, Ny= 3
X1[31]= 0.0655838907249   # NX0=0.5, Ny= 4
X1[32]= 0.00249747959713  # NX0=1.0, Ny= 1
X1[33]= 0.0221262215281   # NX0=1.0, Ny= 2
X1[34]= 0.00056686960192  # NX0=1.0, Ny= 3
X1[35]= 2.68544791147e-05 # NX0=1.0, Ny= 4


X0=np.zeros(36)
#psi variables
X0[0]=0.03 # typ=A, NX0=0, Ny= 1
X0[1]=0.005 # typ=K, NX0=1, Ny= 1
X0[2]=0 # typ=L, NX0=1, Ny= 1
X0[3]=0.0 # typ=A, NX0=0, Ny= 2
X0[4]=0.0 # typ=K, NX0=1, Ny= 2
X0[5]=0.0 # typ=L, NX0=1, Ny= 2
X0[6]=0.0 # typ=K, NX0=2, Ny= 1
X0[7]=0.0 # typ=L, NX0=2, Ny= 1
X0[8]=0.0 # typ=K, NX0=2, Ny= 2
X0[9]=0.0 # typ=L, NX0=2, Ny= 2

#theta variables
X0[10]=0.0 # typ=A, NX0=0, Ny= 1
X0[11]=0.0 # typ=K, NX0=1, Ny= 1
X0[12]=0.0 # typ=L, NX0=1, Ny= 1
X0[13]=0.0 # typ=A, NX0=0, Ny= 2
X0[14]=0.0 # typ=K, NX0=1, Ny= 2
X0[15]=0.0 # typ=L, NX0=1, Ny= 2
X0[16]=0.0 # typ=K, NX0=2, Ny= 1
X0[17]=0.0 # typ=L, NX0=2, Ny= 1
X0[18]=0.0 # typ=K, NX0=2, Ny= 2
X0[19]=0.0 # typ=L, NX0=2, Ny= 2

#A variables
X0[20]=0.0 # NX0=0.5, Ny= 1
X0[21]=0.0 # NX0=0.5, Ny= 2
X0[22]=0.0 # NX0=0.5, Ny= 3
X0[23]=0.0 # NX0=0.5, Ny= 4
X0[24]=0.0 # NX0=1.0, Ny= 1
X0[25]=0.0 # NX0=1.0, Ny= 2
X0[26]=0.0 # NX0=1.0, Ny= 3
X0[27]=0.0 # NX0=1.0, Ny= 4

#T variables
X0[28]=0.0 # NX0=0.5, Ny= 1
X0[29]=0.0 # NX0=0.5, Ny= 2
X0[30]=0.0 # NX0=0.5, Ny= 3
X0[31]=0.0 # NX0=0.5, Ny= 4
X0[32]=0.0 # NX0=1.0, Ny= 1
X0[33]=0.0 # NX0=1.0, Ny= 2
X0[34]=0.0 # NX0=1.0, Ny= 3
X0[35]=0.0 # NX0=1.0, Ny= 4
