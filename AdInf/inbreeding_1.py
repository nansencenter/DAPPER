# "Filter inbreeding" is a vague term.
# The KG depends on the ensemble,
# and thereafter interacts with it, which 
# yields a dependence that causes a negative bias in P^a.
# This means that the EnKF update is not actually linear,
# so that the posterior is not actually Gaussian?
# But does it mean something more than non-Gaussianity and P^a bias?

# For visual clarity, consider omitting observation perturbations!
# (thus focusing on the "intermediate" pdf before including perturbs)

# Conclusions:
#  - Test with N=2, y=0 for a really cool posterior distribution.
#  - The distribution of the posterior ensemble is non-Gaussian.
#  - The density of the posterior ensemble appears (unsure if striclty true)
#    to have a finite support when N=2 (before the obs perturbations).
#  - Biases are clearly visible (for the bias in the mean (KG), use y≠0)
#  - The posterior ensemble members are dependent (if y≠0).
#  - If y is large, then the estimated posterior ens has a larger marginal spread
#    as compared to the case using the true KG...
#  - ...But the average posterior variances are smaller than for the true KG.
#    Why? Dependence...
#  - ...But this dependence seems to just be the introduction of a linear relationship 
#    between the ensemble members... Thus (being linear), it is already taken into account
#    when deriving the bias inequality (by Janson etc...).
#    I.e. it is not something that "in the next cycle" we need to worry about
#    as if it would lead to other problems than the bias.
#  - By contrast, the non-Gaussianity is disconcerting,
#    but largely fixed by the obs perturbations. What happens with a Sqrt update?
     
# - Posterior (true KG) winds up 

# Q: how related are the distributions to the Beta distribution?
# en.wikipedia.org/wiki/Beta_distribution#Generating_beta-distributed_random_variates


from common import *
from statsmodels.nonparametric.kernel_density import KDEMultivariate as kde

#seed(3)

#################################
# Experiment params
#################################
N = 2     # Ens size
K = 10**4 # Num of experiments
B = 2**2  # Prior cov
R = B     # Obs cov
y = 4     # Obs

T = 9 # Num of cycles
# Note: stats seem to CV for T=5 => no point increasing beyond 10.
# Note: The below analysis (plotting, etc) all happens for
#       the ensembles from the last cycle.


#################################
# Ensemble
#################################

Ea   = sqrt(0.5*B)*randn((N,K)) # Prior ensemble (size N), K times
Ea_N = Ea.copy()

for t in range(T):
  # Forecast -- this is very artificial.
  # It's just designed to double the var by 2.
  #E   = sqrt(2)*(Ea   - Ea  .mean(axis=0))
  #E_N = sqrt(2)*(Ea_N - Ea_N.mean(axis=0))
  E   = sqrt(2)*Ea
  E_N = sqrt(2)*Ea_N

  # Analysis
  if False: # Perturbed-obs update
    D = sqrt(R)*randn((N,K))
    #D = 0 # set to zero to omit & focus on "intermediate" pdf.

    # True KG update
    KG   = B/(B + R)                # True Gain -- yields Gaussian posterior
    Ea   = E + KG*(y - E - D)       # Analysis ensemble

    # Ens-estimated update
    B_N  = E_N.var(ddof=1,axis=0)   # Prior sample cov
    #B_N  = (E_N**2).mean(axis=0)    # Prior sample cov
    KG_N = B_N/(B_N + R)            # Kalman gain
    Ea_N = E_N + KG_N*(y - E_N - D) # Analysis ensemble
  else: # Sqrt (ETKF) update
    b_N  = E_N.mean(axis=0)         # Prior sample cov
    B_N  = E_N.var(ddof=1,axis=0)   # Prior sample cov
    A_N  = E_N - b_N

    KG   = B/(B + R)                # True Gain -- yields Gaussian posterior
    b    = E.mean(axis=0)
    A    = E - b
    Ea   = b + KG*(y - b) + A*sqrt(eye(1)-KG)  # Analysis ensemble

    KG_N = B_N/(B_N + R)            # Kalman gain
    Ea_N = b_N + KG_N*(y - b_N) + A_N*sqrt(eye(1)-KG_N)  # Analysis ensemble


#################################
# Plotting
#################################
plt.figure(1).clear()
fig, (ax1,ax2) = plt.subplots(2,num=1,sharex=True,gridspec_kw={'height_ratios':[5, 1]})

EE = array([E, Ea, Ea_N])
ll = ['Prior ($t-1$)','Posterior','Post $N='+str(N)+'$']
xx = np.linspace(*np.percentile(EE,[1,99]), 101)

bb = ax2.boxplot([Ea_N.ravel(), Ea.ravel(), E.ravel()],
    notch=0,sym='',vert=False,
    whis=[5, 95],medianprops={'linewidth':0},
    showmeans=True,meanline=True,meanprops={'color':'k','linestyle':'-'},
    patch_artist=True,labels=ll[::-1],widths=0.7)

for i, (Ei, lbl, patch) in enumerate(zip(EE,ll,bb['boxes'][::-1])):
  # Joint comps
  print("%14.14s var::: Individual: %5.3g, Total: %5.3g "%(lbl,
    mean(Ei.var(ddof=1,axis=0)), Ei.var(ddof=1)))
  Ei  = Ei.ravel()
  dd  = kde(Ei,'c',bw=[0.5*(xx[1]-xx[0])]).pdf(xx)
  lh, = ax1.plot(xx,dd,label=lbl)
  c   = lh.get_color()
  hh  = ax1.hist(Ei, color=c, alpha=0.2, bins=xx,normed=True)
  lh2 = ax2.plot(Ei.mean() + Ei.std(ddof=1)*array([-1,1]), len(EE)-i*ones(2), c=c)
  patch.set(facecolor=c,alpha=0.8)
ax1.legend()
ax1.set_title('Histogram (from K=%d repeats) of'%K +
    '\nthe (N=%d) ensemble members'%N)


#################################
# Plotting
#################################
plt.figure(2,figsize=(5,5)).clear()
fig2, ax3 = plt.subplots(num=2)
ax3.scatter(*E   [:2],alpha=0.05,color='C0')
ax3.scatter(*Ea  [:2],alpha=0.30,color='C1')
ax3.scatter(*Ea_N[:2],alpha=0.10,color='C2')
ax3.set_xlim(xx[0],xx[-1])
ax3.set_ylim(xx[0],xx[-1])
ax3.axis('equal')



#################################
# Moment comparison
#################################
# mm    = []          # empirical moments of individual members
# mm_N  = []          # empirical moments of individual members
# Em    = 1/Ea        # holds ensemble, power m
# Em_N  = 1/Ea_N      # holds ensemble, power m
# im    = arange(9)   # indices of moments
# for m in im:
#   Em   *= Ea
#   Em_N *= Ea_N
#   mm  .append(Em  .mean(axis=1)) # compute moment -- average across K, not ensemble.
#   mm_N.append(Em_N.mean(axis=1)) # compute moment -- average across K, not ensemble.
# mm = array(mm)
# 
# # Actually, this is the same as the ensemble's moments. TODO: understand
# print("Moments, individual members:")
# print_together( im,  np.mean( mm /( mm[2,:]**(im[:,None]/2) ), axis=1) )






