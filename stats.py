from common import *

class Stats:
  """
  Contains and computes peformance stats.
  """

  # Adjust this to omit heavy computations
  comp_threshold_3 = 51

  def __init__(self,setup,config,xx,yy):
    self.setup  = setup
    self.config = config
    self.xx     = xx
    self.yy     = yy
    m    = setup.f.m    ; assert m   ==xx.shape[1]
    K    = setup.t.K    ; assert K   ==xx.shape[0]-1
    p    = setup.h.m    ; assert p   ==yy.shape[1]
    KObs = setup.t.KObs ; assert KObs==yy.shape[0]-1

    # Convenience Fseries constructor
    def fs(m,**kwargs):
      store_u = getattr(config,'store_u',True)
      return Fseries(setup.t, m, store_u=store_u, **kwargs)

    #
    self.mu     = fs(m) # Mean
    self.var    = fs(m) # Variances
    self.mad    = fs(m) # Mean abs deviations
    self.err    = fs(m) # Error (mu-truth)
    self.logp_m = fs(1) # Marginal, Gaussian Log score
    self.skew   = fs(1) # Skewness
    self.kurt   = fs(1) # Kurtosis
    self.rmv    = fs(1) # Root-mean variance
    self.rmse   = fs(1) # Root-mean square error

    if hasattr(config,'N'):
      # Ensemble-only init
      N    = config.N
      # NB: necessary coz isinstance(np.int64,int) fails.
      # TODO: remove np.min from common.
      m_Nm = int(np.minimum(m,N))
      self.w  = fs(N)     # Likelihood weights
      self.rh = fs(m,dtype=int) # Rank histogram
      #self.N  = N        # Use w.shape[1] instead
      self.is_ens = True
    else:
      self.is_ens = False
      m_Nm = m
    self.svals = fs(m_Nm) # Principal component (SVD) scores
    self.umisf = fs(m_Nm) # Error in component directions

    # Analysis-only init
    self.trHK  = fs(1)
    # Note that non-default analysis stats
    # may also be initialized throug at().
    

  def assess(self,k,kObs=None,f_or_a='u',
      E=None,w=None,mu=None,Cov=None):
    """Wrapper for assess_ens/ext and liveplotting."""

    LP      = getattr(self.config,'liveplotting',False)
    store_u = getattr(self.config,'store_u'     ,False)

    if LP or store_u or kObs!=None:

      if E is not None:
        # Ensemble assessment
        comp  = self.assess_ens
        parm  = {'E':E,'w':w}
      else:
        # Linear-Gaussian assessment
        assert mu is not None
        comp = self.assess_ext
        parm = {'mu':mu,'P':Cov}

      key = (k,kObs,f_or_a)
      comp(key,**parm)

      if LP:
        if k==0:
          self.lplot = LivePlot(self,**parm)
        elif f_or_a=='u':
          self.lplot.update(k,kObs,**parm)

    # Return self for daisy-chaining
    return self 


  def assess_ens(self,k,E,w=None):
    """Ensemble and Particle filter (weighted/importance) assessment."""
    N,m          = E.shape
    if w is None:
      self.has_w = False
      w          = 1/N
    else:
      self.has_w = True
    if np.isscalar(w):
      assert w  != 0
      w          = w*ones(N)
    assert(abs(sum(w)-1) < 1e-5)
    assert np.all(np.isfinite(E))
    assert np.all(np.isreal(E))

    x = self.xx[k[0]]

    self.w[k]    = w
    self.mu[k]   = w @ E
    A            = E - self.mu[k]
    self.var[k]  = w @ A**2
    self.mad[k]  = w @ abs(A)  # Mean abs deviations

    unbias_var   = 1/(1 - w@w) # =N/(N-1) if w==ones(N)/N.
    if (1-w.max()) < 1e-10:
      # Don't do in case of weights collapse
      unbias_var = 1
    self.var[k] *= unbias_var
    

    # For simplicity, use naive (and biased) formulae, derived from "empirical measure".
    # See doc/unbiased_skew_kurt.jpg.
    # Normalize by var. Compute "excess" kurt, which is 0 for Gaussians.
    self.skew[k] = mean( w @ A**3 / self.var[k]**(3/2) )
    self.kurt[k] = mean( w @ A**4 / self.var[k]**2 - 3 )

    self.derivative_stats(k,x)

    if sqrt(m*N) <= Stats.comp_threshold_3:
      V,s,UT         = svd( (sqrt(w)*A.T).T, full_matrices=False)
      s             *= sqrt(unbias_var) # Makes s^2 unbiased
      self.svals[k]  = s
      self.umisf[k]  = UT @ self.err[k]

      # For each state dim [i], compute rank of truth (x) among the ensemble (E)
      Ex_sorted     = np.sort(np.vstack((E,x)),axis=0,kind='heapsort')
      self.rh[k]    = [np.where(Ex_sorted[:,i] == x[i])[0][0] for i in range(m)]


  def assess_ext(self,k,mu,P):
    """Kalman filter (Gaussian) assessment."""
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(P))
    assert np.all(np.isreal(mu))   and np.all(np.isreal(P))
    m           = len(mu)

    x = self.xx[k[0]]

    self.mu[k]  = mu
    self.var[k] = diag(P)
    self.mad[k] = sqrt(self.var[k])*sqrt(2/pi)
    # ... because sqrt(2/pi) = ratio MAD/STD for Gaussians

    self.derivative_stats(k,x)

    if m <= Stats.comp_threshold_3:
      s2,U                = nla.eigh(P)
      self.svals[k][::-1] = sqrt(np.maximum(s2,0.0))
      self.umisf[k][::-1] = U.T @ self.err[k]


  def derivative_stats(self,k,x):
    """Stats that apply for both _w and _ext paradigms and derive from the other stats."""
    self.err[k]  = self.mu[k] - x
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


  # TODO: Implement with __getitem__?
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
    """
    Avarage all univariate (scalar) time series.
    """
    avrg = dict()
    for key,series in vars(self).items():
      try:
        # Compute
        f_a_u = series.average()
        # Add the sub-fields as sub-scripted fields
        for sub in f_a_u:
          avrg[key+'_'+sub] = f_a_u[sub]
      except (AttributeError,NotImplementedError):
        pass
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
          val  = mean([s_ij[key].val  for s_ij in row]),
          conf = mean([s_ij[key].conf for s_ij in row])/sqrt(N))
      # NB: This is a rudimentary averaging of confidence intervals
      # Should be checked against variance of avrg[i][key].val
  return avrg


def print_averages(cfgs,Avrgs,attrkeys=(),statkeys=()):
  """
  For i in range(len(cfgs)):
    Print cfgs[i][attrkeys], Avrgs[i][statkeys]
  - attrkeys: list of attributes to include.
      - if -1: only print da_driver.
      - if  0: print distinct_attrs
  - statkeys: list of statistics to include.
  """
  if isinstance(cfgs,DAC):
    cfgs  = DAC_list(cfgs)
    Avrgs = [Avrgs]

  # Defaults averages
  if not statkeys:
    statkeys = ['rmse_a','rmv_a','logp_m_a']

  # Defaults attributes
  if not attrkeys:       headr = list(cfgs.distinct_attrs)
  elif   attrkeys == -1: headr = ['da_driver']
  else:                  headr = list(attrkeys)

  # Filter excld
  excld = ['liveplotting']
  headr = [x for x in headr if x not in excld]
  
  # Get attribute values
  mattr = [cfgs.distinct_attrs[key] for key in headr]

  # Add separator
  headr += ['|']
  mattr += [['|']*len(cfgs)]

  # Get stats.
  # Format stats_with_conf. Use #'s to avoid auto-cropping by tabulate().
  for key in statkeys:
    col = ['{0:#>9} ±'.format(key)]
    for i in range(len(cfgs)):
      try:
        val  = Avrgs[i][key].val
        conf = Avrgs[i][key].conf
        col.append('{0:#>9.4g} {1: <6g} '.format(val,round2sigfig(conf)))
      except KeyError:
        col.append(' '*16)
    crop= min([s.count('#') for s in col])
    col = [s[crop:]         for s in col]
    headr.append(col[0])
    mattr.append(col[1:])
  table = tabulate(mattr, headr).replace('#',' ')
  print(table)



