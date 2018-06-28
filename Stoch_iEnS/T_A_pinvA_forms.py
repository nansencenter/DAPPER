
# Test if                 T1 == T2                       (1)
# where T1 = tinv(Ak) @ A0
#       T2 = tinv( tinv(A0) @ Ak )
# Conclusion: Yes, provided Ak := A0@T, where
#             T does not reduce the rank of A            (S)
#             Examples -- this gets technical -- better to rely on (S)
#              - null(T') \subseteq null(A)
#              - null(T') NOT\subseteq range(A')

##
from common import *

#sd0 = seed_init(2)

N = 4
M = 5

# Projection mat's
def Proj(u):  return u[:,None]@u[None,:] / (u@u)
def ProjC(u): return eye(len(u)) - Proj(u)
def ProjQ(E):
  [Q,R] = sla.qr(E)
  rk    = nla.matrix_rank(R)
  return Q[:,:rk] @ Q[:,:rk].T

# A0
E0 = randn((M,N))
x0 = mean(E0,axis=1,keepdims=True)
A0 = E0-x0

# Ak -- random, not from A0         -- makes (1) False
# --------------------------
#Ek = randn((M,N))
#xk = mean(Ek,axis=1,keepdims=True)
#Ak = Ek-xk

# Ak = A0@T, where T is...
# --------------------------
#
# ... rectanglular:                 -- makes (1) True if rank = N
#T = randn((N,N+2))
#
# ... PC @ symmetric @ PC:          -- makes (1) True if rank >= min(M,N-1)
rk = 3
Pi = ProjQ(randn((N,rk)))
T  = Pi @ randn((N,N)) @ Pi
# -----------------------
Ak = A0 @ T

# Some Y (that also has the mean subtracted) to multiply with
# P  = 4
# hE = randn((P,N))
# yb = mean(hE,axis=1,keepdims=True)
# Y  = hE-yb

## Test
T1 = tinv(Ak) @ A0
T2 = tinv( tinv(A0) @ Ak )
print(abs(T1 - T2).max())

##


