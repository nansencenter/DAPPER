##

# Do 1-parameter experiment from paper by Chen, Yan and Oliver, Dean S.
# "Levenberg--Marquardt forms of the iES
# for efficient history matching and UQ"

##
from common import *

sd0 = seed_init(2)

def NormPDF(xx,b=0,B=1):
  return 1/sqrt(2*pi*B)*exp(-(xx-b)**2/2/B)

xx = linspace(-5,10,2001)
dx = xx[1]-xx[0]
def normlz(pp):
  return pp / sum(pp) / dx

## Prior
b = -2
B = 1


N  = 100
N1 = N-1
E0 = b + sqrt(B)*randn((1,N))
B0 = np.cov(E0,ddof=1)

## Likelihood
y  = 48
R  = 16
#R  = 1
def  h(x): return 7/12*x*x*x - 7/2*x*x + 8*x
def hp(x): return 7/4*x*x - 7*x + 8
#def  h(x): return 4*x
#def hp(x): return 4

def prior(xx): return NormPDF(xx,b,B)
def lklhd(xx): return normlz( NormPDF(y,h(xx),R) )
def postr(xx): return normlz( prior(xx) * lklhd(xx) )

## Plotting
fig = plt.figure(1)
fig.clear()
fig, (ax1, ax2) = plt.subplots(2,1,sharex=True,num=1, gridspec_kw = {'height_ratios':[5, 1]})

_,_,hist = ax1.hist(E0.T,bins=linspace(*xx[[0,-1]],100),
    normed=True, label='$E_0$',alpha=0.6)

ax1.plot(xx,prior(xx), 'b-' , label='$p(x)$')
ax1.plot(xx,lklhd(xx), 'g-' , label='$p(y='+str(y)+'|x)$')
ax1.plot(xx,postr(xx), 'r--', label='$p(x|y)$')

with np.errstate(divide='ignore'):
  ax2.plot(xx,-log( prior(xx) ), 'b-' )
  ax2.plot(xx,-log( lklhd(xx) ), 'g-' )
  ax2.plot(xx,-log( postr(xx) ), 'r--')
  ax2.set_ylim(-2,70)

ax1.set_ylabel('p')
ax2.set_ylabel('-log p')
ax1.legend()
ax2.set_xlabel('x')


#fig.subplots_adjust(right=0.7)
#ax3 = plt.axes([0.60,0.6,0.3,0.3])
#ax3.yaxis.tick_right()
#ax3.set_xlim(xx[0],xx[-1])
#ax3.set_ylim(h(xx[0]),h(xx[-1]))
#scat = ax3.scatter(E0.T, h(E0).T)


## Assimilation
D  = sqrt(R)*randn((1,N))
D -= mean(D)
D *= sqrt(N/N1)

E  = E0
x0 = mean(E0)
A0 = E0 - x0

#FORM='MDA'
FORM='RML'             # RML with exact, local gradients. Gauss-Newton.
#FORM='GN-obs'          # Gauss-Newton, FIXED step length (no Lambda)
#FORM='GN-state'        # Idem. Formulated in state space
#FORM='GN-ens'          # Idem. Formulated in ensemble space
#FORM='LM-orig-state'   # Use Lambda
#FORM='LM-state'        # Use Lambda, modify Hessian  Formulated in state space
#FORM='LM-obs'          # Use Lambda, modify Hessian. Formulated in obs space
#FORM='LM-approx-state' # Drop prior term

Lambda = 1 # Increase in Lambda => Decrease in step length
Gamma  = 4 # Lambda geometric decrease factor

##
nIter = 5
for k in range(nIter):
  Lambda /= Gamma

  A  = E - mean(E)
  hE = h(E)
  Y  = hE - mean(hE)
  H  = Y@tinv(A)

  if FORM=='RML':
    dR = zeros((1,N))
    dB = zeros((1,N))
    for n in range(N): 
      Hn = np.atleast_2d(hp(E[:,n]))
      P  = 1/( 1/B0 + Hn.T/R@Hn )
      dR[:,n] = P@Hn.T/R @(y-D[:,n]-hE[:,n])
      dB[:,n] = P     /B0*(E0[:,n]-E[:,n])
  elif FORM=='MDA':
    D  = sqrt(R)*randn((1,N))
    D -= mean(D)
    D *= sqrt(N/N1)
    K  = A@Y.T/(Y@Y.T + nIter*N1*R)
    dR = K@(y-sqrt(nIter)*D-hE)
    dB = 0
  elif FORM=='GN-obs':
    Z  = H@A0
    K  = A0@Z.T/(Z@Z.T + N1*R)
    dR = K@(y-D-hE)
    dB = (eye(1) - K@H)@(E0-E)
  elif FORM=='GN-state':
    P  = 1/( 1/B0 + H.T/R@H )
    dR = P@H.T/R @(y-D-hE)
    dB = P    /B0*(E0-E)
  elif FORM=='GN-ens':
    Z  = H@A0
    K  = A0@tinv(Z.T@Z/R + N1*eye(N))@Z.T/R
    dR = K@(y-D-hE)
    dB = (eye(1) - K@H)@(E0-E)
  elif FORM=='LM-orig-state':
    P  = 1/( (1+Lambda)/B0 + H.T/R@H )
    dR = P@H.T/R @(y-D-hE)
    dB = P    /B0*(E0-E)
  elif FORM=='LM-state':
    Bk = A@A.T/N1
    P  = 1/( (1+Lambda)/Bk + H.T/R@H )
    dR = P@H.T/R @(y-D-hE)
    dB = P    /B0*(E0-E)
  elif FORM=='LM-obs':
    Bk = A@A.T/N1
    P  = 1/( (1+Lambda)/Bk + H.T/R@H )
    K  = Bk@H.T/(H@Bk@H.T + (1+Lambda)*R)
    dR = K@(y-D-hE)
    dB = P    /B0*(E0-E)
  elif FORM=='LM-approx-state':
    P  = 1/( (1+Lambda)/(A@A.T/N1) + H.T/R@H )
    dR = P@H.T/R @(y-D-hE)
    dB = 0

  E  = E + dR + dB

  # Animation
  if not k%1:
    ax1.set_title(FORM+', k = '+str(k))
    if 'scat' in locals():
      scat.remove()
      scat = ax3.scatter(E.T, hE.T)
    if 'hist' in locals():
      for patch in hist:
        patch.remove()
      _,_,hist = ax1.hist(E.T,bins=linspace(*xx[[0,-1]],100),
          normed=True, label='$E_0$',color='r',alpha=0.6)
    plt.pause(0.5)




##

##

