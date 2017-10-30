# Visualize equi-probably iChi2 sample

from common import *
from scipy.stats import chi2

fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(111)

N  = 5
nu = N-1

def iChi2_pdf(x2):
  c  = nu**(nu/2) / 2**(nu/2) / sp.special.gamma(nu/2)
  return c * x2**(-nu/2-1) * np.exp(-nu/2/x2)

def iChi2_gen(n):
  smpl = randn((nu, n))
  chi2 = np.sum(smpl**2,axis=0)/nu
  return 1/chi2

# Equi-space sample InvChi2.
def cdf_to_s2(cc):
  return nu/chi2(df=nu).ppf(cc)

cc = np.linspace(0,1,17+2)[1:-1][::-1]
ss = cdf_to_s2(cc)

# Test cdf_to_s2: test if all areas are equal
from scipy.integrate import quad
ss2 = np.hstack([0,ss,99])
for i in range(len(ss)-1):
  print(quad(iChi2_pdf,ss2[i],ss2[i+1])[0])

xx = np.linspace(10**-5,6,301)
h0 = ax1.plot(xx,iChi2_pdf(xx),'k',lw=3)[0]

hh = []
for s in ss:
  #hh += [ax1.plot([s,s],[0, iChi2_pdf(s)],c=[0.85, 0.325, 0.098],lw=1)[0]]
  hh += [ax1.plot([s,s],[0, iChi2_pdf(s)],c='k',lw=1)[0]]




