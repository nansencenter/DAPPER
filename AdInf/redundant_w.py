# Check if E^a generated from non-redundant parameterization (w^hat)
# coincides with E^a generated from the symmetric ETKF one.

from common import *
seed(5)

minz = lambda J,x0: sp.optimize.fmin_bfgs(J,x0,disp=0)
def print_mm(s, diff):
  print("Mismatch " + s.ljust(15) + ": ", round2(sla.norm(diff),1e-4))


m = 6 # state/obs dim
N = 5 # ens size

E = randn((m,N))
A = anom(E,axis=1)[0]


U,  Sig,  V_T  = svd(A)      # N-th value is 0
Uh, Sigh, Vh_T = tsvd(A,N-1) # Truncate after N-1
#
V  = V_T.T
Vh = Vh_T.T
#Ah = A @ Vh
#Yh = Y @ Vh

R = CovMat(randcov(m))
Rm12 = R.sym_sqrt_inv
H = eye(m)
Y = H@A
dy= randn(m)
d = Y.T @ R.inv @ dy

####################
# ETKF
####################
# Redundant version
Pw = inv((N-1)*eye(N) + Y.T @ R.inv @ Y)
wa = Pw @ d # should have mean 0
xa = A @ wa
#
Ta = sqrtm(Pw) * sqrt(N-1)
Ea = A @ Ta

# Truncated version
Ph = inv((N-1)*eye(N-1) + Vh.T@Y.T @ R.inv @ Y@Vh)
wh = Ph @ Vh.T @ d
xh = A @ Vh @ wh
#
Th = sqrtm(Ph) * sqrt(N-1)
Eh = A @ Vh @ Th @ Vh.T

print_mm("mean",xa - xh)
print_mm("covr",Ea - Eh)
#wa == Vh@wh

# The fact that these match confirms the validity of 
# both approaches (redundant/truncated), 
# something which is perhaps necessary
# for the unfamiliar truncated parameterization.

####################
# With optimization
####################
L    = lambda w:  .5*np.sum( ( Rm12@(dy-Y@w) )**2 )
J    = lambda w:  .5*(N-1)*w@w + L(w)
#Lp  = lambda w:  -Y.T@R.inv@(dy-Y@w)
#Jp  = lambda w:  (N-1)*w + Lp(w)
w_o  = minz(J,zeros(N))

Lh   = lambda wh: .5*np.sum( ( Rm12@(dy-Y@Vh@wh) )**2 )
Jh   = lambda wh: .5*(N-1)*wh@wh + Lh(wh)
wh_o = minz(Jh,zeros(N-1))

print_mm("mean optmz",A@w_o - A@Vh@wh_o)

# Optimization also seems to work well

####################
# Finite-size (-N) prior
####################
g = 0
eN = 1 + 1/N

J_N  = lambda w:  .5*(N+g)*log(eN + w@w) + L(w)
w_N  = minz(J_N,zeros(N))
#Hw   = Y@R.inv@Y.T/(N-1) + eye(N) - 2*np.outer(w,w)/(eN/mc + w@w)

Jh_N = lambda wh: .5* N   *log(eN + wh@wh) + Lh(wh)
wh_N = minz(Jh_N,zeros(N-1))

# Is zero only when g==0
print_mm("mean -N",A@w_N - A@Vh@wh_N)
