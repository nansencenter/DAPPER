"""
    DAPPER's parameters module
    ======================

    This module defines the parameters for the data assimilation experiment

    The Code is available here : https://github.com/nansencenter/DAPPER/tree/max1/mods/MAOOAM

    :Example:

    >>>from mods.MAOOMA.maooam16 import setup

    Global variable
    -------------------

    ic.X1,ic.X0
    dt,dtObs,T,BurnIn
    f with m,model,noise

    Dependencies
    -------------------

    import numpy as np
    import mods.MAOOAM.params2
    import mods.MAOOAM.aotensor
    import mods.MAOOAM.integrator
    import mods.MAOOAM.ic_def as ic_def

"""

from common import *
import numpy as np
import params2
import aotensor
import integrator
import ic_def as ic_def

#initial condition already after the transient time (10 years here)
mu0= ic.X1

print (" Model initialized")
m = ndim
p = m

#time definition
dt=0.1
dtObs=8.9
T=3244.4
BurnIn=0

t = Chronology(dt,dtObs=dtObs,T=T,BurnIn=BurnIn)

f = {
    'm': m,
    'model': lambda x,t,dt: step(x,t,dt),
    'noise': 0
    }

#variance computed on 200 effective years after a transient time of 1000 years 
#computed with Fortran with a step time of 0.1
var=array([5.68701073021621e-06 , 0.00106977154793093  , 0.00107912057634068  ,
            0.000682220051684437 , 0.000370724677648213 , 0.000380012960718116 ,
            0.000233496123084933 , 0.000234614816221311 , 0.000135658331225872 ,
            0.000135213434886638 , 3.71044080159669e-05 , 0.000155746111510384 ,
            0.000167476737510925 , 6.19206298578686e-05 , 5.27414901902687e-05 ,
            5.15973081700329e-05 , 2.30981682553207e-05 , 2.29988040685897e-05 ,
            1.30342345626815e-05 , 1.29693535370592e-05 , 2.41487424967582e-09 ,
            4.88395185297478e-10 , 2.80830246430725e-10 , 5.06040839449456e-14 ,
            1.52894187894308e-09 , 3.76034851073869e-10 , 3.53078503721912e-10 ,
            5.45737350125537e-14 , 0.00105494674038039  , 4.55069564363275e-05 ,
            0.00108048436423909  , 0.000680107299411221 , 0.00179691436291966  ,
            0.000185073455267781 , 8.45204002944011e-05 , 2.01981237603906e-05  ])

#ensemble VARIANCE is 1% of var 200 years effective dt=0.01
C0 = 0.01*0.01*diag(var)
X0 = GaussRV(C=C0,mu=mu0)

#observation noise variance is 1% of the var on 200 years effective dt=0.01
# R= 0.01*0.01*diag(var2[[20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]])
R= 0.01*0.01*diag(var)
hnoise = GaussRV(C=CovMat(R),mu=0)

h = {
    'm': p,
    'model': lambda x,t: x,
    'TLM'  : lambda x,t: eye(p),
    'noise': hnoise,
    }
other = {'name': os.path.basename(__file__)}

setup = OSSE(f,h,t,X0,**other)
