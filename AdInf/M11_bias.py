# Test if there's a bias involved
# in using the inflation estimate
# s = (d^2 - R)/P,
# where B = s_true*P.
# The answer it no, as can be shown by
# conditional expectation (wrt s_true).

from common import *

K = 10**5

# B = s*P
# d^2 = B+R

TRUE_FACTOR = 0.5
R = 1
 
dd    = zeros(K)
estim = zeros(K)

for k in range(K):
  P = 2 + rand()
  B = TRUE_FACTOR*P
  dd[k]    = sqrt(B+R)*randn()
  estim[k] = (dd[k]**2 - R)/P

print(mean(estim))

