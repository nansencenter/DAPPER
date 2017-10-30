# weighted (and correlated) sum of chi2 vars with different scales is not a chi2 var.
# -- but it's pretty similar.

from common import *
from scipy.stats import chi2

K = 10**5
m = 5

A = diag(1+arange(m)**2) # randcov(m)
B = diag(1+arange(m)**2) # randcov(m)
C = A+B

C12 = sqrtm(C)

Am12 = funm_psd(A, lambda x: x**-.5)
Bm12 = funm_psd(B, lambda x: x**-.5)
Cm12 = funm_psd(C, lambda x: x**-.5)


fig, ax = plt.subplots()
xx = linspace(0,4*trace(C))

dd = C12 @ randn((m,K))
nn = sum(dd**2,0)

ax.hist(nn,normed=1,bins=100)

C2 = chi2(df=m,scale=trace(C)/m)
ax.plot(xx,C2.pdf(xx))


