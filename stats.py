from common import *


class Stats:
  """Contains and computes peformance stats."""
  # TODO: Include skew/kurt?
  def __init__(self,params,cfg):
    self.params = params
    m    = params.f.m
    K    = params.t.K
    KObs = params.t.KObs
    #
    self.mu    = zeros((K+1,m))
    self.var   = zeros((K+1,m))
    self.mad   = zeros((K+1,m))
    self.umisf = zeros((K+1,m))
    self.smisf = zeros(K+1)
    self.ldet  = zeros(K+1)
    self.logp  = zeros(K+1)
    self.logp_r= zeros(K+1)
    self.logp_m= zeros(K+1)
    self.skew  = zeros(K+1)
    self.kurt  = zeros(K+1)
    self.err   = zeros((K+1,m))
    self.rmv   = zeros(K+1)
    self.rmse  = zeros(K+1)
    self.rh    = zeros((K+1,m))
    self.trHK  = zeros(KObs+1)
    if hasattr(cfg,'N'):
      N    = cfg.N
      m_Nm = np.minimum(m,N)
      self.svals = zeros((K+1,m_Nm))

  def assess(self,E,x,k):
    assert(type(E) is np.ndarray)
    N,m             = E.shape
    self.mu[k]    = mean(E,0)
    A               = E - self.mu[k]
    self.var[k]   = sum(A**2  ,0) / (N-1)
    self.mad[k]   = sum(abs(A),0) / (N-1)
    self.skew[k]    = mean( sum(A**3,0)/N / self.var[k]**(3/2) )
    self.kurt[k]    = mean( sum(A**4,0)/N / self.var[k]**2 - 3 )
    self.err[k]   = x[k] - self.mu[k]
    self.rmv[k]    = sqrt(mean(self.var[k]))
    self.rmse[k]    = sqrt(mean(self.err[k]**2))

    # Marginal log score
    ldet            = sum(log(self.var[k]))
    nmisf           = self.var[k]**(-1/2) * self.err[k]
    logp_m          = sum(nmisf**2) + ldet
    self.logp_m[k]  = logp_m/m

    # Preparation for log score
    V,s,UT          = svd(A)
    s              /= sqrt(N-1)
    self.svals[k]   = s
    s               = s[s>1e-4]
    r               = np.minimum(len(s),5)
    s               = s[:r]

    # Full-joint Gaussian log score
    #alpha           = 1/20*mean(s)
    alpha           = 1e-2*sum(s)
    # Truncating s by alpha doesnt work:
    #s               = s[s>alpha]
    #r               = len(s)
    s2_full         = array(list(s**2) + [alpha]*(m-r))
    ldet            = sum(log(s2_full)) / m
    umisf           = UT @ self.err[k]
    nmisf           = (s2_full)**(-1/2) * umisf
    logp            = ldet + sum(nmisf**2)
    self.umisf[k] = umisf
    self.smisf[k]   = sum(nmisf**2)/m
    self.ldet[k]    = ldet/m
    self.logp[k]    = logp/m

    # Reduced-Joint Gaussian log score
    ldet            = sum(log(s**2))
    nmisf           = s**(-1) * (UT[:r] @ self.err[k])
    logp_r          = sum(nmisf**2) + ldet
    self.logp_r[k]  = logp_r/r

    # Rank histogram
    Ex_sorted       = np.sort(np.vstack((E,x[k])),axis=0,kind='heapsort')
    self.rh[k]    = [np.where(Ex_sorted[:,i] == x[k,i])[0][0] for i in range(m)]

  def assess_w(self,E,x,k,w):
    assert(type(E) is np.ndarray)
    assert(abs(sum(w)-1) < 1e-5)
    N,m           = E.shape
    self.mu[k]  = w @ E
    A             = E - self.mu[k]
    self.var[k] = w @ A**2
    self.err[k] = self.mu[k] - x[k]
    self.rmv[k]  = sqrt(mean(self.var[k]))
    self.rmse[k]  = sqrt(mean(self.err[k]**2))

  def assess_ext(self,mu,ss,x,k):
    m             = len(mu)
    self.mu[k]  = mu
    self.var[k] = ss**2
    self.err[k] = self.mu[k] - x[k]
    self.rmv[k]  = sqrt(mean(self.var[k]))
    self.rmse[k]  = sqrt(mean(self.err[k]**2))

  def copy_paste(self,s,kObs):
    """
    Load s into stats object at kObs.
    Avoids having to pass kObs into enkf_analysis (e.g.).
    """
    for key,val in s.items():
      getattr(self,key)[kObs] = val

  def average_after_burn(self):
    t    = self.params.t
    kk_a = t.kkObsBI                   # analysis time > BurnIn
    kk_f = t.kkObsBI-1                 # forecast      > BurnIn
    kk_u = t.kk                        # all times     > BurnIn
    kk_O = arange(t.kObsBI, t.KObs+1)  # all obs times > BurnIn
    avrg = dict()
    for key,val in vars(self).items():
      if type(val) is np.ndarray:
        if is1d(val):
          if len(val) == t.K+1:
            avrg[key + '_a'] = series_mean_with_conf(val[kk_a])
            avrg[key + '_f'] = series_mean_with_conf(val[kk_f])
            avrg[key + '_u'] = series_mean_with_conf(val[kk_u])
          elif len(val) == t.KObs+1:
            avrg[key] = series_mean_with_conf(val[kk_O])
    return avrg

      


def average_each_field(ss,axis=None):
  assert ss.ndim == 2
  if axis == 0:
    ss = np.transpose(ss)
  m,N = ss.shape
  avrg = np.empty(m,dict)
  keys = ss[0][0].keys()
  for i,row in enumerate(ss):
    avrg[i] = dict()
    for key in keys:
      avrg[i][key] = val_with_conf(
          val  = np.mean([s_ij[key][0] for s_ij in row]),
          conf = np.mean([s_ij[key][1] for s_ij in row])/sqrt(N))
      # NB: This is a rudimentary averaging of confidence intervals
      # Should be checked against var avrg[i][key].val
  return avrg


def print_averages(DAMs,avrgs,*statnames):
  headr = ' '*17
  for sname in statnames:
    headr += '{0: >8} ±'.format(sname) + ' '*7
  print(headr)
  for k,meth in enumerate(DAMs):
    line = '{0: <16}'.format(meth.da_method.__name__)
    for sname in statnames:
      val = avrgs[k][sname]
      if type(val) is val_with_conf:
        line += '{0: >9.4f} {1: <6g} '.format(val.val,round2sigfig(val.conf))
      else:
        line += '{0: >9.4f} {1: <6g} '.format(val[0],val[1])
    print(line)



