##

# Test if expected
#  - projection matrix is identity  (answer: only if N>=M)
#  - sensitivity matrix is unbiased (answer: only if N>=M -- not P)

## Preamble
from common import *

K = 10**4   # Num of experiments
M = 3       # State length
P = 4       # Obs length
N = 6       # Ens size

cntr = lambda E: anom(E,1)[0]

seed(6)
#B = randcov(M)             # yields non-diag Pi_infty. Why? 
B = np.diag(1+np.arange(M)) # yields diagonal Pi_infty.
H = np.round(10*rand((P,M)))
h = lambda x: (H@x)**2 + 3*(H@x) + 4 # 2nd-D polynomial
seed()

# "infty" is exaggerated. There's still noticeable sampling error here.
E_infty = sqrtm(B)@randn((M,K))
h_infty = cntr(h(E_infty)) @ tinv(cntr(E_infty))


P_av = zeros((M,M))
H_av = zeros((P,M))
h_av = zeros((P,M))
for k in range(K):
  E  = sqrtm(B)@randn((M,N))
  iE = tinv(cntr(E), threshold=1-1e-3)

  P_av +=   E  @ iE / K
  H_av += H@E  @ iE / K
  h_av += h(E) @ iE / K

##
print("****trace(P_av)\n", trace(P_av))
print("****P_av\n", round2(P_av  , 0.01))

print("****H\n"    , H)
print("****H_av\n" , round2(H_av , 0.01))

print("****h_infty\n" , round2(h_infty , 0.01))
print("****h_av\n"    , round2(h_av    , 0.01))

##


