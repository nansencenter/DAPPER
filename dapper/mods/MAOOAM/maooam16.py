# DAPPER's configuration for MAOOAM

from common import *
import numpy as np
import mods.MAOOAM.params2
import mods.MAOOAM.aotensor
import mods.MAOOAM.integrator
import mods.MAOOAM.ic_def as ic_def

#initial condition already after the transient time (10 years here)
ic_def.load_IC()
import mods.MAOOAM.ic as ic
mu0= ic.X1

print (" Model initialized")
m = mods.MAOOAM.params2.ndim


#3 years simulation - time step 0.1
#obs time step 1 day
#model transient time 10 years
#DA transient time 1

#T=32444
#BurnIn=32444
T=3244.4
BurnIn=0
t = Chronology(0.1,dtObs=8.9,T=T,BurnIn=BurnIn)

f = {
    'm': m,
    'model': lambda x,t,dt: mods.MAOOAM.integrator.step(x,t,dt),
    'TLM'  : 0,
    'noise': 0
    }

#the var on 100 years effective dt=0.1
var = array([   2.25705441e-05,   2.70829072e-04,   3.00434304e-04,
         2.24907615e-04,   1.48750056e-04,   2.30601477e-04,
         6.24018781e-05,   6.07697222e-05,   3.35717671e-05,
         3.32832185e-05,   2.80775530e-05,   5.68928169e-05,
         7.48198724e-05,   2.98037439e-05,   2.41227820e-05,
         8.89876165e-05,   7.04450249e-06,   6.81838195e-06,
         2.59002057e-06,   2.57755616e-06,   1.23963736e-09,
         4.46789964e-08,   1.76000387e-09,   5.35801312e-12,
         1.43288311e-09,   6.22013831e-08,   1.27243907e-09,
         5.07635211e-12,   1.12155637e-04,   2.71467456e-03,
         1.57806939e-04,   2.76227153e-03,   3.06848875e-04,
         3.23858331e-03,   4.04752311e-05,   1.61287593e-06])
#GRL var after 1000 years - 200 years - 0.1 - fortran
# var2=array([  5.50178812e-06,   1.16978285e-03,   1.17177680e-03,
#          7.21962886e-04,   3.83099335e-04,   3.90241255e-04,
#          2.28479504e-04,   2.29940349e-04,   1.34958968e-04,
#          1.36127019e-04,   3.68486787e-05,   1.65499476e-04,
#          1.74648794e-04,   6.60638117e-05,   5.55218680e-05,
#          5.36613113e-05,   2.34540876e-05,   2.35861194e-05,
#          1.35298128e-05,   1.35433822e-05,   6.82109881e-10,
#          9.61053046e-11,   1.87162920e-10,   8.67905811e-15,
#          5.22157944e-10,   7.27624452e-11,   1.82442890e-10,
#          1.20562714e-14,   2.96399161e-04,   3.60052729e-05,
#          8.05671226e-04,   1.79436977e-04,   4.95553796e-04,
#          3.45518933e-05,   5.79866355e-06,   5.70396595e-06])

var2=array([5.68701073021621e-06,0.00106977154793093,0.00107912057634068,
0.000682220051684437,0.000370724677648213,0.000380012960718116,
0.000233496123084933,0.000234614816221311,0.000135658331225872,
0.000135213434886638,3.71044080159669e-05,0.000155746111510384,
0.000167476737510925,6.19206298578686e-05,5.27414901902687e-05,
5.15973081700329e-05,2.30981682553207e-05,2.29988040685897e-05,
1.30342345626815e-05,1.29693535370592e-05,2.41487424967582e-09,
4.88395185297478e-10,2.80830246430725e-10,5.06040839449456e-14,
1.52894187894308e-09,3.76034851073869e-10,3.53078503721912e-10,
5.45737350125537e-14,0.00105494674038039,4.55069564363275e-05,
0.00108048436423909,0.000680107299411221,0.00179691436291966,
0.000185073455267781,8.45204002944011e-05,2.01981237603906e-05])
#ensemble VARIANCE is 1% of var 200 years effective dt=0.01
C0 = 0.01*0.01*diag(var2)
X0 = GaussRV(C=C0,mu=mu0)

############################################################################
#STRONGLY - FULL OBSERVED
############################################################################
#observation noise variance is 1% of the var on 200 years effective dt=0.01
R= 0.01*0.01*diag(var2)
hnoise = GaussRV(C=CovMat(R),mu=0)
@atmost_2d
def hmod(E,t):
  return E[:,:]

h = {
    'm': 36,
    'model' : hmod,
    'noise': hnoise,
    }

other = {'name': os.path.basename(__file__)}

setup = OSSE(f,h,h,t,X0,**other)
############################################################################
#STRONGLY - FULL OBSERVED - DIFFERENT OBS FREQUENCY
############################################################################
#observation noise variance is 1% of the var on 200 years effective dt=0.01
R= 0.01*0.01*diag(var2)
hnoise = GaussRV(C=CovMat(R),mu=0)
@atmost_2d
def hmod(E,t):
  return E[:,:]

h = {
    'm': 36,
    'model' : hmod,
    'noise': hnoise,
    }

Ratm= 0.01*0.01*diag(var2[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]])
hnoiseatm = GaussRV(C=CovMat(Ratm),mu=0)
@atmost_2d
def hmodatm(E,t):
  return E[:,[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]]

hatm = {
    'm': 20,
    'model' : hmodatm,
    'noise': hnoiseatm,
    }

other = {'name': os.path.basename(__file__)}

setupmix = OSSE(f,hatm,h,t,X0,**other)
############################################################################
#STRONGLY - ONLY ATMOSPHERE
############################################################################
#observation noise variance is 1% of the var on 200 years effective dt=0.01
Ratm= 0.01*0.01*diag(var2[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]])
hnoiseatm = GaussRV(C=CovMat(Ratm),mu=0)
@atmost_2d
def hmodatm(E,t):
  return E[:,[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]]

hatm = {
    'm': 20,
    'model' : hmodatm,
    'noise': hnoiseatm,
    }

setupatm = OSSE(f,hatm,hatm,t,X0,**other)
############################################################################
#STRONGLY - ONLY OCEAN
############################################################################
#observation noise variance is 1% of the var on 200 years effective dt=0.01
Roc= 0.01*0.01*diag(var2[[20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]])
hnoiseoc = GaussRV(C=CovMat(Roc),mu=0)
@atmost_2d
def hmodoc(E,t):
  return E[:,[[20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]]]

hoc = {
    'm': 16,
    'model' : hmodoc,
    'noise': hnoiseoc,
    }

setupoc = OSSE(f,hoc,hoc,t,X0,**other)
############################################################################
#WEAKLY
############################################################################
#observation noise variance is 1% of the var on 200 years effective dt=0.01
Ratm= 0.01*0.01*diag(var2[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]])
hnoiseatm = GaussRV(C=CovMat(Ratm),mu=0)


hatmw = {
    'm': 20,
    'model' : lambda x,t: x,
    'noise': hnoiseatm,
    }

#observation noise variance is 1% of the var on 200 years effective dt=0.01
Roc= 0.01*0.01*diag(var2[[20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]])
hnoiseoc = GaussRV(C=CovMat(Roc),mu=0)

hocw = {
    'm': 16,
    'model' : lambda x,t: x,
    'noise': hnoiseoc,
    }

setupw = OSSE(f,hatmw,hocw,t,X0,**other)
############################################################################