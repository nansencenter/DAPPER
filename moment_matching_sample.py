from common import *

N     = 3
p_max = 2*N
dp    = 1/p_max/2
x_min,x_max = sqrt(ss.chi2.ppf([dp,1-dp],df=1))
xx0   = linspace(x_min,x_max,N)

def fac2(n):
  if n <= 0:
    return 1
  else:
    return n * fac2(n-2)

# Pair moment indices
pp = arange(2,p_max+1,2)

# Pair exact moments (assuming sig=1)
moments = array([fac2(p-1) for p in pp])

# Pair empirical moments
def empirs(xx):
  return array([2*sum(xx**p)/(2*len(xx)+1-p) for p in pp])

#def fuse(old,new):
  #return np.insert(old,[0,len(old)],new)

#old = array([-1.0,1.0])
#for p in pp[1:]:
  #ip   = p//2
  #moms = moments[:ip]
  #G    = (N-p)//2
  #x0   = fuse(old, xx0[[G,-G-1]])
  #emps = empirs(x0)
  #print(x0,moms,emps)
  #old = x0 


diff = lambda xx: sum(np.diff(xx)**2)
ordr = lambda xx: 10*sum((np.diff(np.argsort(xx)) - 1)**2)
mism = lambda xx: sum(((moments-empirs(xx))/moments)**2)

def J(xx):
  return mism(xx)

xx = opt.minimize(J,xx0)
#xx = opt.basinhopping(J,xx0)

#xx1 = linspace(0.001,1,101)
#xx2 = linspace(0.1  ,2,101)
#xx3 = linspace(0.5  ,9,101)

#X,Y,Z = np.meshgrid(xx1,xx2,xx3)
#X = X.ravel()
#Y = Y.ravel()
#Z = Z.ravel()

#JJ = array([J(array([X[i],Y[i],Z[i]])) for i in range(len(X))])

#ind = np.argmin(JJ)
#xyz = [X[ind], Y[ind], Z[ind]]








# QUESTION:
#  I want a "sample" $\{x_n\}^N_{n=1}$ of a univariate Gaussian $\mathcal{N}(0,1)$.
#  
#  I want to choose this sample deterministically, so as to be optimal in some way.
#  "Sigma points"
#  "Quasi-random"
#  "Moment matching"
#  "Gauss-Hermite roots" without weights
#  
#  Not possible if using population estimators.
#  
#  https://mathoverflow.net/questions/214400/matching-moments-in-even-dimensions?noredirect=1&lq=1
#  
#  https://mathoverflow.net/questions/229646/gaussian-and-the-convex-hull-of-moment-curves?rq=1
#  
#  https://mathoverflow.net/questions/229600/moment-matching-construction-of-a-mixture-of-gaussian-distribution-with-lower-m
#  
#  https://mathoverflow.net/questions/20789/approximate-a-probability-distribution-by-moment-matching?noredirect=1&lq=1
