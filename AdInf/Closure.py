# Check if an N=3 ensemble can maintain perfect moments
# through propagation through a d=2 polynomial.
# Conclusion: NO


from common import *

poly1 = lambda x:        3*x + 4
poly2 = lambda x: x**2 + 3*x + 4

#fun = poly1 # Bug check: with this (linear) model, moments do match.
fun = poly2 # Does not seem to work.

# Init.
# I believe the central member must be zero if the skewness is to be zero.
# Thus, moment-matching a Gauss(mu=0,sig=1) with N=3 up to 3rd moment
# does not even require solving a system of eqns.
E  = array([-1,0,1])
N  = 10**7
MC = randn(N)

# Propagation
E2  = fun(E)
MC2 = fun(MC)

# Assessment
m1 = mean(E2)
A  = E2 - m1
print('m1: ', m1)
print('m2: ', sum(A**2,0)/2)
print('m3: ', sum(A**3,0)/2)

print()

m1 = mean(MC2)
A  = MC2 - m1
print('m1: ', m1)
print('m2: ', sum(A**2,0)/(N-1))
print('m3: ', sum(A**3,0)/(N-1))





###############
# Experiment 2
###############
# What if we increase N and num of constraints?

#  def F(E):
#    #mean(    E)  = 0
#    #std (    E)  = 1
#    #skew(    E)  = 0
#    #mean(fun(E)) = 5  # std (fun(MC)) 
#    #std (fun(E)) = 11 # std (fun(MC))
#    #skew(fun(E)) = 62 # skew(fun(MC))
#    N  = len(E)
#    m1 = mean(E)
#    A  = E - m1
#    m2 = sum(A**2,0)/(N-1)
#    m3 = sum(A**3,0)/(N-1)
#  
#    Ef  = fun(E)
#    mf1 = mean(Ef)
#    #Af  = Ef - mf1
#    #mf2 = sum(Af**2,0)/(N-1)
#    #mf3 = sum(Af**3,0)/(N-1)
#  
#    # Just including a single nonlin constraints => fail to converge
#    return array([m1, m2-1.0, m3, mf1-5.0])
#    #return array([m1, m2-1.0, m3, mf1-5.0, mf2-11.0, mf3-62.0])
#  
#  
#  import scipy.optimize
#  E0 = randn(4)
#  #E6 = scipy.optimize.broyden1(F, E0)
#  #E6 = scipy.optimize.excitingmixing(F, E0)
#  #E6 = scipy.optimize.linearmixing(F, E0)
#  #E6 = scipy.optimize.newton_krylov(F, E0)




