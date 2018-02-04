# Hurrily converted from Matlab.

##

from common import *

from scipy.integrate import quad
from scipy.optimize import minimize_scalar as minz

plt.style.use('AdInf/paper.mplstyle')

seed(90) 

nu  = 4
std = 1 # don't change


## T distribution
t_nnz = lambda x: (1 + (x/std)**2/nu)**(-(nu+1)/2)
t_c   = quad(t_nnz, -np.inf, +np.inf)[0] # Complains if approx nu<1.5
t     = lambda x: t_nnz(x) / t_c
# t   = lambda x: tpdf(x,nu) # alternatively

## Gaussian
g = lambda x,s1: sp.stats.norm.pdf(x,loc=0,scale=s1*std)


## Chi
# Know that: s2 ~ InvChi2(nu).
# With u2 := 1/s2, then u2  ~ Chi2(nu).
cdf_to_s2 = lambda cdf: nu/sp.stats.chi2.ppf(cdf,df=nu)
# See verify_scale_mix.m
# and understanding_innovation_rescaling.m


## Likelihood
y = 5
R = 1
lklhd = lambda x: sp.stats.norm.pdf(x,loc=y,scale=sqrt(R))

## Posterior and plot
xm = 8
xx = linspace(-4,xm,401)
ym = 0.8
fig = plt.figure(1,figsize=(10,3.0))
fig.clear()
ax = plt.gca()

## Posteiror t
pt_nnz = lambda x: t(x)*lklhd(x)
pt_c   = quad(pt_nnz, -np.inf, +np.inf)[0]
pt     = lambda x: pt_nnz(x) / pt_c

## Laplace (Gaussian) approximation
J = lambda x: -log(pt(x))
mode = minz(J, bounds=(-10,10), method='bounded').x
dx = 1e-5
Hess = np.diff(np.diff(J(mode+array([-dx,0,+dx]))))/dx**2
Lap = lambda x: sp.stats.norm.pdf(x,loc=mode,scale=Hess**(-1/2))

## For deterministic "Monte-Carlo" quadrature scale mix.
# I want to "sample" the Gaussians,
# with equidistantly (in CDF terms) scalings.
K  = 17; dK = 1/K
cc = linspace(dK/2,1-dK/2,K)
ss = cdf_to_s2(cc)

## Test locations and the Pdfs at test locations (scale)
x0 = 0; gx0 = 0
x1 = 1; gx1 = 0

h = Bunch()
h.p1 = [0]*len(ss)
h.p2 = [0]*len(ss)

for i,s2 in enumerate(ss):
    s1 = sqrt(s2)
    
    # Prior g
    h.p1[i], = ax.plot(xx,g(xx,s1), c=blend_rgb('mlo',0.6),label='dontstyle',lw=1.0)
    
    gx0 = gx0 + g(x0,s1)
    gx1 = gx1 + g(x1,s1)
    
    # Posterior g
    pg2_nnz = lambda x: g(x,s1)*lklhd(x)
    h.p2[i], = ax.plot(xx,pg2_nnz(xx)*55.7,c=blend_rgb('mlb',0.6),label='dontstyle',lw=1.0)

## Verify that gaussians sum to t distribution:
# Should be equal if K is large.
gratio = gx0/t(x0)
print('t(x1): ',t(x1),'\ng(x1): ,',gx1/gratio,'\n\n')
## Thick lines
h.td,  = ax.plot(xx,t(xx)    ,lw=3,c='mlo')
h.lk,  = ax.plot(xx,lklhd(xx),lw=3,c='mlg')
h.po,  = ax.plot(xx,pt(xx)   ,lw=3,c='mlb')
# h.Lap = plot(xx,Lap(xx)  ,'r:',lw=1)
# structfun(@(h) set(h,label='dontstyle'), h)




## Ens

N = nu+1
# N = 20 # For viz effect
N = 6 # For viz effect
ens = sqrt(cdf_to_s2(0.5))*std*randn(N)
ens = (ens-mean(ens))*sqrt(N/(N-1)) # For viz effect
##
##
#h.g1, = ax.plot(xx,g(xx,1),c='k',lw=2)
#en2  = mode + Hess**(-1/2)*(ens-0)/(sqrt(cdf_to_s2(0.5))*std)
#h.e2 = ax.scatter(en2,zeros(N),40,'b')
#h.g2, = ax.plot(xx,Lap(xx),c='k',lw=2)
h.e1 = ax.scatter(ens,zeros(N),40,'mlo',zorder=3)

ax.set_ylim(-0.02,ym)
ax.set_xlim(-4,xm)
ax.legend([h.e1, h.lk, h.p1[0], h.p2[0], h.td, h.po],
    ['Prior ensemble', 'Likelihood', 'Candidate priors', 'Candidate posteriors', 'Effective prior', 'Effective posterior'],
    loc='upper left',framealpha=0,
    prop={'size':10})
ax.set_xticks([]); ax.set_yticks([]); # See Note1 below.
ax.axis('off')
ax.arrow(-4,0,4+8,0,color='k')
ax.arrow(-4,0,0,0.8,color='k')
ax.text(-4.3,0.4,'pdf',rotation='vertical',fontsize=11)



ax.text(2.2,-0.06,'$x$',fontsize=16)



## 

sys.exit(0)

###

# Note1: must remove ticks before ax.axis('off') to get right bounding box when saving

#fname='illust_scale_mix_py'
#plt.savefig('data/AdInf/figs/'+fname+'.eps',pad_inches=0.01)
#plt.savefig('data/AdInf/figs/'+fname+'.pdf',pad_inches=0.01)


##
