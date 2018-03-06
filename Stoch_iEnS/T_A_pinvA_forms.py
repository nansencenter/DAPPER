##

# Test if:    tinv(Ak) @ A0 == tinv( tinv(A0) @ Ak )   (1)
# Conclusion: Yes, provided Ak := A0@Tk,
#             where Tk is any of the different cases tested.

##
from common import *

sd0 = seed_init(2)

N = 3
M = 4
P = 4

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
#Ek = randn((M,N))
#xk = mean(Ek,axis=1,keepdims=True)
#Ak = Ek-xk

# Ak = A0@Tk, where Tk is
# -----------------------
# --- PC @ symmetric @ PC           -- makes (1) True
#PC = ProjC(ones(N))
#Tk = randcorr(N)
#Tk = PC @ Tk @ PC
# --- PC @ symmetric @ PC           -- makes (1) True if rank >= min(M,N-1)
rk = M
Pi = ProjQ(randn((N,rk)))
Tk = Pi @ randcorr(N) @ Pi
# --- random square                 -- makes (1) True
#Tk = randn((N,N))
# --- rectangle                     -- makes (1) True
#Tk = randn((N,N-2))
# --- rank deficient                -- makes (1) False
#Tk = zeros((N,N))
#Tk[0,1] = 1
# -----------------------
Ak = A0 @ Tk

# Some Y (that also has the mean subtracted) to multiply with
hE = randn((P,N))
yb = mean(hE,axis=1,keepdims=True)
Y  = hE-yb

##
T1 = tinv(Ak) @ A0
T2 = tinv( tinv(A0) @ Ak )
T1 - T2

##

##
