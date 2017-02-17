from common import *

class Stats:
  """
  Contains and computes statistics of the DA methods.
  """

  # Adjust this to omit heavy computations
  comp_threshold_3 = 51

  def __init__(self,setup,config,xx,yy):
    """
    Init a bunch of stats.
    Note: the individual stats may very well be allocated & computed
          elsewhere, and simply added as an attribute to the stats
          instance. But the most common ones are gathered here.
    """

    self.setup  = setup
    self.config = config
    self.xx     = xx
    self.yy     = yy
    m    = setup.f.m    ; assert m   ==xx.shape[1]
    K    = setup.t.K    ; assert K   ==xx.shape[0]-1
    p    = setup.h.m    ; assert p   ==yy.shape[1]
    KObs = setup.t.KObs ; assert KObs==yy.shape[0]-1

    fs          = self.new_Fseries
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
      m_Nm = min(m,N)
      self.w  = fs(N)           # Likelihood weights
      self.rh = fs(m,dtype=int) # Rank histogram
      #self.N  = N              # Use w.shape[1] instead
      self.is_ens = True
    else:
      self.is_ens = False
      m_Nm = m
    self.svals = fs(m_Nm) # Principal component (SVD) scores
    self.umisf = fs(m_Nm) # Error in component directions

    # Other. 
    self.trHK = np.full(KObs+1, nan)
    self.infl = np.full(KObs+1, nan)

  def assess(self,k,kObs=None,f_a_u=None,
      E=None,w=None,mu=None,Cov=None):
    """
    Common interface for both assess_ens and _ext.
    f_a_u: One or more of ['f',' a', 'u'], indicating
           that the result should be stored in (respectively)
           the forecast/analysis/universal attribute.
           Defaults: see source code.
    If 'u' in f_a_u: call/update LivePlot.
    """

    if k==0: assert kObs is None,\
        "Should not have any obs at initial time."+\
        "This very easily yields bugs, and not 'DA convention'."

    # Defaults for f_a_u
    if f_a_u is None:
      if kObs is None:
        f_a_u = 'u'
      else:
        f_a_u = 'au'
    elif f_a_u == 'fau':
      if kObs is None:
        f_a_u = 'u'

    LP      = getattr(self.config,'liveplotting',False)
    store_u = getattr(self.config,'store_u'     ,False)

    if not (LP or store_u) and kObs==None:
      pass # Skip assessment
    else:
      if E is not None:
        # Ensemble assessment
        ens_or_ext = self.assess_ens
        state_prms = {'E':E,'w':w}
      else:
        # Linear-Gaussian assessment
        assert mu is not None
        ens_or_ext = self.assess_ext
        state_prms = {'mu':mu,'P':Cov}

      # Compute
      key = (k,kObs,f_a_u)
      ens_or_ext(key,**state_prms)

      # LivePlot
      if LP:
        if k==0:
          self.lplot = LivePlot(self,**state_prms,only=LP)
        elif 'u' in f_a_u:
          self.lplot.update(k,kObs,**state_prms)

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
    assert(abs(w.sum()-1) < 1e-5)
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

    m = len(mu)
    x = self.xx[k[0]]

    self.mu[k]  = mu
    self.var[k] = diag(P)
    self.mad[k] = sqrt(self.var[k])*sqrt(2/pi)
    # ... because sqrt(2/pi) = ratio MAD/STD for Gaussians

    self.derivative_stats(k,x)

    if m <= Stats.comp_threshold_3:
      s2,U          = nla.eigh(P)
      self.svals[k] = sqrt(np.maximum(s2,0.0))[::-1]
      self.umisf[k] = (U.T @ self.err[k])[::-1]


  def derivative_stats(self,k,x):
    """Stats that apply for both _w and _ext paradigms and derive from the other stats."""
    self.err[k]  = self.mu[k] - x
    self.rmv[k]  = sqrt(mean(self.var[k]))
    self.rmse[k] = sqrt(mean(self.err[k]**2))
    self.MGLS(k)
    
  def MGLS(self,k):
    # Marginal Gaussian Log Score.
    m              = len(self.err[k])
    ldet           = log(self.var[k]).sum()
    nmisf          = self.var[k]**(-1/2) * self.err[k]
    logp_m         = (nmisf**2).sum() + ldet
    self.logp_m[k] = logp_m/m


  def average_in_time(self):
    """
    Avarage all univariate (scalar) time series.
    """
    avrg = dict()
    for key,series in vars(self).items():
      try:
        # Fseries
        if isinstance(series,Fseries):
          # Compute
          f_a_u = series.average()
          # Add the sub-fields as sub-scripted fields
          for sub in f_a_u: avrg[key+'_'+sub] = f_a_u[sub]
        # Array
        elif isinstance(series,np.ndarray):
          if series.ndim > 1:
            raise NotImplementedError
          t = self.setup.t
          if len(series) == len(t.kkObs):
            inds = t.maskObs_BI
          elif len(series) == len(t.kk):
            inds = t.kk_BI
          else:
            raise ValueError
          # Compute
          avrg[key] = series_mean_with_conf(series[inds])
        # Scalars
        elif np.isscalar(series):
          avrg[key] = series
        else:
          raise NotImplementedError
      except NotImplementedError:
        pass
    return avrg

  def new_Fseries(self,m,**kwargs):
    "Convenience Fseries constructor."
    store_u = getattr(self.config,'store_u',True)
    return Fseries(self.setup.t, m, store_u=store_u, **kwargs)

  # Better to initialize manually (np.full...)
  # def new_array(self,f_a_u,m,**kwargs):
  #   "Convenience array constructor."
  #   t = self.setup.t
  #   # Convert int-len to shape-tuple
  #   if is_int(m):
  #     if m==1: m = ()
  #     else:    m = (m,)
  #   # Set length
  #   if f_a_u=='a':
  #     K = t.KObs
  #   elif f_a_u=='u':
  #     K = t.K
  #   #
  #   return np.full((K+1,)+m,**kwargs)



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
    #statkeys = ['rmse_a','rmv_a','logp_m_a']
    statkeys = ['rmse_a','rmv_a','rmse_f']

  # Defaults attributes
  if not attrkeys:       headr = list(cfgs.distinct_attrs)
  elif   attrkeys == -1: headr = ['da_driver']
  else:                  headr = list(attrkeys)

  # Filter excld
  excld = ['liveplotting','store_u']
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



