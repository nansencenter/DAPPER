##

# Consider the univariate problem from Chen, Yan and Oliver, Dean S.
# "Levenberg--Marquardt forms of the iES for efficient history matching and UQ"
# This is here generalized to M dimensions
# by repeating the problem independently for each dim.
# Note: sampling error will couple the dimensions somewhat.
# Setting M = P = 1 can be used to reproduce the results of the paper.

## Preamble
from common import *
from scipy.stats import norm

#plt.xkcd()

sd0 = seed_init(2)

def mean1(E):
  "Enables correct broadcasting for MxN shaped ensemble matrices"
  return mean(E,axis=1,keepdims=True)

M  = 5       # State length
P  = 2       # Obs length
N  = 400     # Ens size
N1 = N-1     #
nbins = 300   # Histogram bins

## Prior
b  = -2*ones((M,1))
B  = 1*eye(M)
E0 = b + sqrtm(B)@randn((M,N))

## Obs
jj = arange(P)
y  = 48*ones((P,1))
R  = 16*eye(P)
#R  = 1*eye(P)
def h1(x): return 7/12*x*x*x - 7/2*x*x + 8*x
def  h(x): return h1(x) [jj,:]
def hp(x):
  H = zeros((P,M))
  for i,j in enumerate(jj):
    H[i,j] = 7/4*x[j]**2 - 7*x[j] + 8
  return H

## PDFs
xx = linspace(-5,10,2001)
dx = xx[1]-xx[0]
def normlz(pp):
  return pp / sum(pp) / dx

prior_xx = norm.pdf(xx,b[0],sqrt(B[0,0]))
lklhd_xx = normlz( norm.pdf(y[0],h1(xx),sqrt(R[0,0])) )
postr_xx = normlz( prior_xx * lklhd_xx )

## Plotting
fig = plt.figure(1)
fig.clear()
fig, (ax1, ax2) = plt.subplots(2,1,sharex=True,num=1, gridspec_kw = {'height_ratios':[5, 1]})

ax1.hist(E0[0,:],bins=linspace(*xx[[0,-1]],nbins),
    normed=True, label='$E_0$',alpha=0.6)

ax1.plot(xx,prior_xx, 'b-' , label='$p(x)$')
ax1.plot(xx,lklhd_xx, 'g-' , label='$p(y='+str(y[0,0])+'|x)$')
ax1.plot(xx,postr_xx, 'r--', label='$p(x|y)$')

with np.errstate(divide='ignore'):
  ax2.plot(xx,-log( prior_xx ), 'b-' )
  ax2.plot(xx,-log( lklhd_xx ), 'g-' )
  ax2.plot(xx,-log( postr_xx ), 'r--')
  ax2.set_ylim(-2,70)

ax1.set_ylabel('p')
ax2.set_ylabel('-log p')
ax1.legend()
ax2.set_xlabel('x')

## Assimilation

# Obs perturbations
D  = sqrtm(R)@randn((P,N))
D -= mean1(D)
D *= sqrt(N/N1)

# Sqrt ininitalization matrix
Tinv   = eye(N)
T      = eye(N)
w      = zeros((N,1))
W      = eye(N)

# Initialize ensemble
E  = E0
x0 = mean1(E0)
A0 = E0 - x0
B0 = A0@A0.T/N1


#FORM='MDA'                   # Ensemble Multiple Data Assimilation
#FORM='RML-GN'                # RML with exact, local gradients. Gauss-Newton.
#FORM='EnRML-GN-obs'          # EnRML, Gauss-Newton
#FORM='EnRML-GN-state'        # Idem. Formulated in state space
#FORM='EnRML-GN-ens'          # Idem. Formulated in ensemble space
#FORM='EnRML-LM-ORIG-state'   # Use Lambda to adjust step lengths. 
#FORM='EnRML-LM-state'        # Use Lambda, modify Hessian  Formulated in state space
#FORM='EnRML-LM-obs'          # Use Lambda, modify Hessian. Formulated in obs space
#FORM='EnRML-LM-APPROX-state' # Drop prior term
#FORM='iEnS-Det-GN'           # Sqrt, determin, iter EnS
FORM='iEnS-Stoch-GN'          # Sqrt, stoch, iter EnS. Equals EnRML-GN ? 

# Only applies for LM methods
Lambda = 1 # Increase in Lambda => Decrease in step length
Gamma  = 4 # Lambda geometric decrease factor

##
nIter = 16
for k in range(nIter):
  Lambda /= Gamma

  A  = E - mean1(E)
  hE = h(E)
  Y  = hE - mean1(hE)
  H  = Y@tinv(A)

  if FORM=='RML':
    dLkl = zeros((M,N))
    dPri = zeros((M,N))
    for n in range(N): 
      Hn        = hp(tp(E[:,n]))
      Pn        = inv( inv(B0) + Hn.T@inv(R)@Hn )
      dLkl[:,n] = Pn@Hn.T@inv(R)@(y.ravel()-D[:,n]-hE[:,n])
      dPri[:,n] = Pn@inv(B0)@(E0[:,n]-E[:,n])
  elif FORM=='MDA':
    D    = sqrtm(R)@randn((P,N))
    D   -= mean1(D)
    D   *= sqrt(N/N1)
    K    = A@Y.T@inv(Y@Y.T + nIter*N1*R)
    dLkl = K@(y-sqrt(nIter)*D-hE)
    dPri = 0
  elif FORM=='EnRML-GN-obs':
    Z    = H@A0
    K    = A0@Z.T@inv(Z@Z.T + N1*R)
    dLkl = K@(y-D-hE)
    dPri = (eye(M) - K@H)@(E0-E)
  elif FORM=='EnRML-GN-ens':
    Z    = H@A0
    K    = A0@inv(Z.T@inv(R)@Z + N1*eye(N))@Z.T@inv(R)
    dLkl = K@(y-D-hE)
    dPri = (eye(M) - K@H)@(E0-E)
  elif FORM=='EnRML-GN-state':
    P    = inv( inv(B0) + H.T@inv(R)@H )
    dLkl = P@H.T@inv(R)@(y-D-hE)
    dPri = P@inv(B0)@(E0-E)
  elif FORM=='EnRML-LM-ORIG-state':
    P    = inv( (1+Lambda)*inv(B0) + H.T@inv(R)@H )
    dLkl = P@H.T@inv(R)@(y-D-hE)
    dPri = P@inv(B0)@(E0-E)
  elif FORM=='EnRML-LM-state':
    Bk   = A@A.T/N1
    P    = inv( (1+Lambda)*inv(Bk) + H.T@inv(R)@H )
    dLkl = P@H.T@inv(R)@(y-D-hE)
    dPri = P@inv(B0)@(E0-E)
  elif FORM=='EnRML-LM-obs':
    Bk   = A@A.T/N1
    P    = inv( (1+Lambda)*inv(Bk) + H.T@inv(R)@H )
    K    = Bk@H.T@inv(H@Bk@H.T + (1+Lambda)*R)
    dLkl = K@(y-D-hE)
    dPri = P@inv(B0)@(E0-E)
  elif FORM=='EnRML-LM-APPROX-state':
    P    = inv( (1+Lambda)*inv(A@A.T/N1) + H.T@inv(R)@H )
    dLkl = P@H.T@inv(R)@(y-D-hE)
    dPri = 0
  elif FORM=='iEnS-Det-GN':
    #Z    = H@A0
    Z    = Y @ Tinv
    Hw   = Z.T@inv(R)@Z + N1*eye(N)
    Pw   = inv(Hw)
    Tinv = funm_psd(Hw, sqrt)/sqrt(N1)
    T    = funm_psd(Pw, sqrt)*sqrt(N1)
    dLkl = Pw@Z.T@inv(R)@(y-mean1(hE))
    dPri = Pw@-w*N1
    w   += dLkl + dPri
    # CVar to ensemble space
    dLkl = A0@dLkl
    dPri = A0@dPri
    # Anomalies update (add to dPri)
    dPri = dPri + A0@T - A
  elif FORM=='iEnS-Stoch-GN':
    #Z    = H@A0
    #Tinv = tinv(tinv(A0)@A)
    Tinv = tinv(T)
    Z    = Y @ Tinv
    Hw   = Z.T@inv(R)@Z + N1*eye(N)
    Pw   = inv(Hw)
    dLkl = Pw@Z.T@inv(R)@(y-D-hE)
    dPri = Pw@(eye(N)-W)*N1
    W   += dLkl + dPri
    T    = Pw@(N1*eye(N) - Z.T@inv(R)@D)
    # CVar to ensemble space
    dLkl = A0@dLkl
    dPri = A0@dPri


  E  = E + dLkl + dPri

  # Animation
  if not k%1:
    ax1.set_title(FORM+', k = '+str(k)+'. Plotting for state index [0]')
    if 'hist' in locals():
      for patch in hist:
        patch.remove()
    _,_,hist = ax1.hist(E[0,:],bins=linspace(*xx[[0,-1]],nbins),
        normed=True, label='$E_0$',color='r',alpha=0.6)
    plt.pause(0.5)




##

##

