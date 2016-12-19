# Experiments with a non-linear transformation (NLT)
# as the dynamical model that has the property that
# Y = NLT(X) is Gaussian if X is Gaussian.
# 
# Shows that, if using a Sqrt EnKF:
# - Linearity abates sampling error:
#    - barB CV to B after initial transitory regime,
#    - and the ensemble itself also CVes.
#    - Both of these are analytical results of Bocquet/Carrassi/Grudzien!
# - Conversely, non-linearity creates sampling error,
#   as observed by the everpresent jitteriness in barB and barP,
#   even if Gaussianity is maintained.
# - Thus, sampling errors cause problems on their own,
#   and can be regarded as a separate issue from non-Gaussianity.
# - Sampling error causes jitteriness (in B itself) and P and xa.
#   Adapt.Infl can remedy jitteriness, since y also contains info on B.
# - Sampling error also fuels the negative bias (in P).
#   Adapt.Infl should remedy this too.
# - The "sampling error" due to the deterministic NLT behaves just like random noise:
#   for example, barP is negatively biased.
#   This may be ascertained by observing that NLT(sqrt(eN)*normalized(Ens)) has
#   expected sample var of 2 exactly (Ex(barB) = 2) and yet Ex(barP)<1,
#   however, the normalization is turned off for design simplicity.
# - More applications of model render the sample more noisy.
#   Thus, a longer DAW (e.g.) will cause a greater need for infl.
#
# Todo: m-dim case: NLT can be made for radius of multivariate Gaussian,
#       which is chi2(ddof=m).


###########################
# Preamble
###########################

from common import *

#seed(37) # 37 is a good one

# The Non-Lin-Transf is a composite of familiar func's
NLT  = lambda x: sqrt(2)*ss.norm.ppf(ss.chi2.cdf(x**2,1))
Lin  = lambda x: sqrt(2)*x
aprx = lambda x: sqrt(2)*(-0.40 + 0.88*np.abs(zz) + 0.23*np.log(zz**2))
# aprx found by fitting a + b|x| + c log x²

def normalize(xx):
  N  = len(xx)
  mu = np.sum(xx)/N
  xx = xx - mu
  s2 = sum(xx**2)/(N-1)
  xx = xx/sqrt(s2)
  return xx

nBins = 30
nHist = int(1e5)
xlim  = 3

xx = randn(nHist)
yy = NLT(xx)


###########################
# Histograms
###########################
# Just to get familiar with the NLT model.

plt.figure(1)
plt.clf()

axX = plt.subplot(221)
xxp = plt.hist(xx,bins=nBins,range=(-xlim,xlim))[2]
axX.tick_params(labelbottom='off')

axY = plt.subplot(224)
yyp = plt.hist(yy,bins=nBins,range=(-xlim,xlim),orientation="horizontal")[2]
axY.tick_params(labelleft='off')

# Transformation
axT = plt.subplot(223)
zz  = linspace(-xlim,xlim,201)
zz  = zz[zz!=0]
plt.plot(zz,NLT(zz))
plt.plot(zz,aprx(zz))
plt.ylim(-xlim,xlim)


# Bin mapping
xx_bins = [p.get_x() for p in xxp]
yy_bins = [NLT(x) for x in xx_bins]
inds    = sorted(range(nBins), key=lambda k: yy_bins[k])

cm = plt.cm.get_cmap('viridis')

# Bin colouring
for i,p in enumerate(xxp):
  p.set_color(cm(i/(nBins-1)))
for i,p in enumerate(yyp):
  p.set_color(cm(inds[i]/(nBins-1)))


###########################
# EnKF experiment
###########################
R  = 2
P  = 1
y  = 0

N  = 40
eN = 1 + 1/N
E0 = NLT(randn(N))
nT = 100
#nT = 10000

# NLT
#######
E  = E0.copy()
PP = zeros(nT)
BB = zeros(nT)
bb = zeros(nT)
for k in arange(nT):
  b = np.mean(E)
  A = E-b
  B = sum(A**2)/(N-1)
  K = B/(B+R)
  T = sqrt(1 - K)
  xa= b + K*(y - b)
  E = xa + T*A
  bb[k] = xa
  BB[k] = B
  PP[k] = T**2*B
  #E = sqrt(eN) * normalize(E) # Yields Ex(barB) = 2 (exactly)
  E = NLT(E)
  
plt.figure(2)
plt.clf()
plt.ylim((-0.3,3))
plt.yticks(arange(0,4),('0.0','1.0','2.0','3.0'))
plt.xlabel('DA cycle (i.e. time) index')
plt.text(0.93*nT,0.1,'$\\bar{\mathbf{x}}^a$'  ,size=25,color='k')
plt.text(0.93*nT,1.1,'$\\bar{\mathrm{P}}^a$'  ,size=25,color='k')
plt.text(0.93*nT,2.1,'$\\bar{\mathrm{B}}$'    ,size=25,color='k')

plt.plot(BB,'b')
plt.plot(PP,'g')
plt.plot(bb,'r')

print('mean B:', mean(BB))
print('mean P:', mean(PP))


# Lin
#######
plt.figure(3)
plt.clf()

E  = E0.copy()
PP = zeros(nT)
BB = zeros(nT)
bb = zeros(nT)
for k in arange(nT):
  b = np.mean(E)
  A = E-b
  B = sum(A**2)/(N-1)
  K = B/(B+R)
  T = sqrt(1 - K) # Yields the same as left-multiplying sym sqrt
  Aa= T*A
  xa= b + K*(y - b)
  E = xa + Aa
  if k < 20:
    plt.plot(E ,0.45 + k*ones(N),'r|',MarkerSize=15)
  bb[k] = xa
  BB[k] = B
  PP[k] = (1-K)*B
  E = Lin(E)

plt.figure(2)
plt.plot(BB,'b',lw=2)
plt.plot(PP,'g',lw=2)
plt.plot(bb,'r',lw=2)


###########################
# QUIT
###########################
sys.exit(0)


###########################
# Mini experiment 1
###########################
# Show that NLT(normalize(E))
# is slightly biased unless compensated for by sqrt(eN).

N  = 10
eN = 1 + 1/N
B  = 0
P  = 0
nT = 10000
for k in range(nT):
  E = randn(N)
  E = sqrt(eN) * normalize(E)
  b = sum(E)/N
  A = E-b
  B += sum(A**2)/(N-1) / nT

  E = NLT(E)
  b = sum(E)/N
  A = E-b
  P += sum(A**2)/(N-1) / nT
print()
print('mean var before NLT:', B)
print('mean var after  NLT:', P)


###########################
# Mini experiment 2
###########################
# How much "sampling error" does NLT cause?
# I.e. How inexact (as measured by its variance)
#      is NLT(E) if E is an exact sqrt.?

def NLT_recursive(x,level=1):
  if level == 0:
    return x
  x = NLT_recursive(x,level-1)
  x = ss.norm.ppf(ss.chi2.cdf(x**2,1)) # don't use sqrt(2)
  return x


PP = zeros(nT)
for k in range(nT):
  E = randn(N)
  E = sqrt(eN) * normalize(E)    # Remove "sampling error"
  E = NLT_recursive(E,2)
  b = sum(E)/N
  A = E-b
  PP[k] = sum(A**2)/(N-1)

print()
print('mean varance',mean(PP)) # Should be approx 1.
print('var of var  ',np.var(PP)) 
# This should increase with the num of NLT applied.
# Tops out at 0.22 (for recursion level=0 or >5). Min=0.15 (for level=1).





