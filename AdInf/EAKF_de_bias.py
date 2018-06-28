# EAKF has series (=> scalar) obs.
# This allows analytically deriving
# the sampling distrbution of the posterior's var,
# when the state is also scalar.
#
# In the scalar case (state-length = 1),
# H does not add anything interesting,
# so it's not included.

import numpy as np
from scipy.special import gamma
from scipy.stats   import chi2
import matplotlib.pyplot as plt 
plt.ion()
plt.style.use(['seaborn'])

#np.random.seed(2)

## True
B  = 1
R  = 2



## Formulae
def  P(B): return 1/(1/B+1/R)
def KG(B): return 1/(1 + R/B)

## Ensemble
N = 5     # Ens size
K = 40000 # Num of experiments
E = np.sqrt(B)*np.random.randn(K,N)
B_bar  = (E*E).sum(axis=1) / N


## Analytical

B_bar_pdf = chi2(df=N,scale=B/N).pdf

def P_bar_pdf(x):
  B_bar = 1/(1/x - 1/R) # CVar
  # Version: just adding the jacobian to chi2
  #pp = chi2(df=N,scale=B/N).pdf(B_bar) * (B_bar/x)**2

  # Version: full, explicit formulae
  c  = N**(N/2) / 2**(N/2) / gamma(N/2)
  with np.errstate(divide='ignore',invalid='ignore'):
    pp = c * B**(-N/2) * x**(-2) * B_bar**(N/2+1) * np.exp(-N*B_bar/2/B)

  return pp

def KG_bar_pdf(x):
  B_bar = R/(1/x - 1) # CVar
  return chi2(df=N,scale=B/N).pdf(B_bar) * (B_bar/x)**2 / R


## Plotting
plt.figure(1).clear()
fig, ax = plt.subplots(num=1)

xx   = np.linspace(0,4*B,int(K/200)) + 0.01
opts = {'alpha':0.45, 'bins':xx,'normed':True}

ax.hist(   B_bar  ,**opts , label='Prior'  ,color='C6')
ax.hist( P(B_bar) ,**opts , label='Post'   ,color='C2')
ax.hist(KG(B_bar) ,**opts , label='K.Gain' ,color='C1')

ax.plot(xx,  B_bar_pdf(xx),                 color='C6')
ax.plot(xx,  P_bar_pdf(xx),                 color='C2')
ax.plot(xx, KG_bar_pdf(xx),                 color='C1')

ax.set_title("Sampling distribution of ens-estimated objects's, where \n"+
    'N=%d, B=%g, R=%g, P(B)=%g, K(B)=%g'%(N,B,R,P(B),KG(B)))
ax.legend()

print("Numbers averaged over the %d experiments:"%K)
print("%-6s: %-5g"%('B_bar ',   B_bar .mean()))
print("%-6s: %-5g"%('P_bar ', P(B_bar).mean()))
print("%-6s: %-5g"%('KG_bar',KG(B_bar).mean()))



