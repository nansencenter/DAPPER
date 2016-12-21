from common import *


class Stats:
  """
  Contains and computes peformance stats.
  """

  # Skip heavy computations (of level n)
  # if m > MAX_m_LEVEL_n
  MAX_m_LEVEL_3 = 10**2

  def __init__(self,setup,config):
    self.setup = setup
    m    = setup.f.m
    K    = setup.t.K
    KObs = setup.t.KObs
    #
    self.mu    = zeros((K+1,m))
    self.var   = zeros((K+1,m))
    self.mad   = zeros((K+1,m))
    self.logp_m= zeros(K+1)
    self.skew  = zeros(K+1)
    self.kurt  = zeros(K+1)
    self.err   = zeros((K+1,m))
    self.rmv   = zeros(K+1)
    self.rmse  = zeros(K+1)
    self.rh    = zeros((K+1,m))

    # Ensemble-only init
    if hasattr(config,'N'):
      N    = config.N
      m_Nm = np.minimum(m,N)
      self.w     = zeros((K+1,N))
      self.svals = zeros((K+1,m_Nm))
      self.umisf = zeros((K+1,m_Nm))

    # Analysis-only init
    self.trHK  = zeros(KObs+1)
    # Note that non-default analysis stats
    # may also be initialized throug at().
    

  def assess(self,E,x,k):
    """Ensemble assessment method."""
    return self.assess_w(E,x,k)

  def assess_w(self,E,x,k,w=None):
    """Particle filter (weighted/importance) assessment."""
    N,m          = E.shape
    w            = 1/N*ones(N) if w==None else w
    assert np.all(np.isfinite(E))
    assert(abs(sum(w)-1) < 1e-5)

    
    self.w[k]    = w
    self.mu[k]   = w @ E
    A            = E - self.mu[k]
    self.var[k]  = w @ A**2
    var_unbiased = 1/(1 - w@w) # Scaling "sample variance" into "unbiased estimator"
    self.var[k] *= var_unbiased

    if max(m,N) <= Stats.MAX_m_LEVEL_3:
      V,s,UT         = svd( (sqrt(w)*A.T).T, full_matrices=False)
      s             *= sqrt(var_unbiased) # Makes s^2 unbiased
      #s             /= sqrt(N-1)         # For A un-weighted
      self.svals[k]  = s
      umisf          = UT @ self.err[k]

      # For each state dim [i], compute rank of truth (x) among the ensemble (E)
      Ex_sorted     = np.sort(np.vstack((E,x[k])),axis=0,kind='heapsort')
      self.rh[k]    = [np.where(Ex_sorted[:,i] == x[k,i])[0][0] for i in range(m)]
      # TODO: Mean/Median skewness

    # Use naive, "empirical measure" formulae. Also see doc/unbiased_skew_kurt.jpg
    self.mad[k]  = w @ abs(A)                            # Mean abs deviations
    self.skew[k] = mean( w @ A**3 / self.var[k]**(3/2) ) # Skewness (normalized)
    self.kurt[k] = mean( w @ A**4 / self.var[k]**2 - 3 ) # "Excess" kurtosis

    self.derivatives(k,x)

    return self

  def assess_ext(self,mu,ss,x,k):
    """Kalman filter (Gaussian) assessment."""
    m            = len(mu)
    self.mu[k]   = mu
    self.var[k]  = ss**2
    self.derivatives(k,x)
    return self

  def derivatives(self,k,x):
    """Stats that apply for both _w and _ext paradigms and derive from the other stats."""
    self.err[k]  = self.mu[k] - x[k]
    self.rmv[k]  = sqrt(mean(self.var[k]))
    self.rmse[k] = sqrt(mean(self.err[k]**2))
    self.MGLS(k)
    
  def MGLS(self,k):
    # Marginal Gaussian Log Score.
    m              = len(self.err[k])
    ldet           = sum(log(self.var[k]))
    nmisf          = self.var[k]**(-1/2) * self.err[k]
    logp_m         = sum(nmisf**2) + ldet
    self.logp_m[k] = logp_m/m


  def at(self,kObs):
    """Provide a write-access method for the analysis frame of index kObs"""
    def write_at_kObs(**kwargs):
      for key,val in kwargs.items():
        if not hasattr(self,key):
          shape = (self.setup.t.KObs+1,)
          if isinstance(val,np.ndarray):
            shape += val.shape
          setattr(self,key,zeros(shape))
        getattr(self,key)[kObs] = val # writes by reference ([kObs])
    return write_at_kObs

  def average_in_time(self):
    t    = self.setup.t
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


def get_vc(val_conf):
  if isinstance(val_conf, val_with_conf):
    v,c = val_conf.val, val_conf.conf
  else:
    # TODO: Remove this format (i.e. [0], [1])
    v,c = val_conf[0], val_conf[1]
  return v,c
  

def print_averages(cfgs,Avrgs,attrkeys=(),statkeys=()):
  """
  For i in range(len(cfgs)):
    Print cfgs[i][attrkeys], Avrgs[i][statkeys]
  """
  if isinstance(cfgs,DAC):
    cfgs  = DAC_list(cfgs)
    Avrgs = [Avrgs]

  # Defaults averages
  if not statkeys:
    statkeys = ['rmse_a','rmv_a','logp_m_a']

  # Defaults attributes
  if   attrkeys == 0: headr = ['da_driver']
  elif attrkeys ==(): headr = list(cfgs.distinct_attrs)
  else:               headr = list(attrkeys)
  mattr = [cfgs.distinct_attrs[key] for key in headr]

  # Add separator
  headr += ['|']
  mattr += [['|']*len(cfgs)]

  # Format stats_with_conf. Use #'s to avoid auto-cropping by tabulate().
  for key in statkeys:
    col = ['{0:#>9} ±'.format(key)]
    for i in range(len(cfgs)):
      val,conf = get_vc(Avrgs[i][key])
      col.append('{0:#>9.4f} {1: <6g} '.format(val,round2sigfig(conf)))
    crop= min([s.count('#') for s in col])
    col = [s[crop:]         for s in col]
    headr.append(col[0])
    mattr.append(col[1:])
  table = tabulate(mattr, headr).replace('#',' ')
  print(table)



