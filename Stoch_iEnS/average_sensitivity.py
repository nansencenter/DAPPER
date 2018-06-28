# Test if the *average* sensitivity is *exact* for
# - Gaussian distributions                   # Yes
# - Non-Gaussian, with the nonlinear h being a polynomial of degree:
#   - 2: Only if skew=0. But the sensitivity *at* the mean is the same anyways.
#   - 3: No -- even with skew = 0

#


from common import *
import seaborn as sns

#sd0 = seed_init(2)

xx = linspace(-5,10,2001)
dx = xx[1]-xx[0]

## Prior
b = 0
B = 2


N  = 10**7
N1 = N-1

## h
R  = 0
#def  h(x): return -0.1*x + x**2
#def hp(x): return -0.1 + 2*x
def  h(x): return -0.1*x + x**2 - 0.2 *x**3
def hp(x): return -0.1 + 2*x    - 0.6 *x**2
#def  h(x): return 4*x
#def hp(x): return 4

## Ens
xx = b + sqrt(B)*randn((1,N)) # Gaussian
TriMod = False                # Non-Gaussian (but just to check
if TriMod:
  wN = [int(w*N) for w in [0.5, 0.1876]]; wN.append(N-sum(w))
  xx = ccat(    xx[:,:wN[0]],
     3 + 0.5*randn((1,wN[1])),
    -4 + 0.5*randn((1,wN[2])),
    axis=1)
yy = h(xx) + sqrt(R)*randn((1,N))

prnt = lambda s,v: print('%13.13s: %.5f'%(s,v))
prnt("mean(xx)", mean(xx) )
prnt(" var(xx)", np.var(xx,ddof=1) )
prnt("skew(xx)", ss.skew(xx.T,bias=False) )


## Stats
Cyx = np.cov(yy,xx,ddof=1)[0,1]
Cx  = np.var(xx,ddof=1)
H   = Cyx/Cx
print("")
prnt("H",H)
prnt("mean(hp(xx))", mean(hp(xx)) )
prnt("hp(mean(xx))", hp(mean(xx)) )

hp_mu = hp(mean(xx))
mu_hp = mean(hp(xx))

## Plotting
df = pd.DataFrame(ccat(xx,yy).T[np.random.choice(N,10**4)], columns=['x','y'])
sns.jointplot(x="x", y="y", data=df)

