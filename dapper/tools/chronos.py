# Time sequence management

from dapper import *

class Chronology():
  """Time schedules with consistency checks.

  - Uses int records => tt[k] == k*dt.
  - Uses generators  => time series may be arbitrarily long.

  Illustration:

  .. code-block:: none

                         <----dtObs---->
                  <--dt-->
    tt:    0.0    0.2    0.4    0.6    0.8    1.0    T
    kk:    0      1      2      3      4      5      K
           |------|------|------|------|------|------|
    kObs:  None   None   0      None   1      None   KObs
    kkObs:               2             4             6
                         <----dkObs---->

  .. warning:: By convention, there is no obs at 0.
               This is hardcorded in DAPPER,
               whose cycling starts by the forecast.

  Identities (subject to precision)::

    len(kk)    == len(tt)    == K   +1
    len(kkObs) == len(ttObs) == KObs+1

    kkObs[0]   == dkObs      == dtObs/dt == K/(KObs+1)
    kkObs[-1]  == K          == T/dt
    KObs       == T/dtObs-1

  These attributes may be set (altered) after init: dt, dkObs, K, T.
  Other attributes may not, due to ambiguity
  (e.g. should dtObs*=2 yield a doubling of T too?)
  """

  def __init__(self,dt=None,dtObs=None,T=None,BurnIn=-1, \
      dkObs=None,KObs=None,K=None,Tplot=None):

    assert 3 == [dt,dtObs,T,dkObs,KObs,K].count(None) , \
        'Chronology is specified using exactly 3 parameters.'

    # Reduce all to "state vars" dt,dkObs,K
    if not dt:
      if T and K:
        dt = T/K
      elif dkObs and dtObs:
        dt = dtObs/dkObs
      elif T and dkObs and KObs:
        dt = T/(KObs+1)/dkObs
      else: raise TypeError('Unable to interpret time setup')
    if not dkObs:
      if dtObs:
        dkObs = round(dtObs/dt)
        assert abs(dtObs - dkObs*dt) < dt*1e-9
      else: raise TypeError('Unable to interpret time setup')
    assert is_int(dkObs)
    if not K:
      if T:
        K = round(T/dt)
        assert abs(T - K*dt) < dt*1e-9
      elif KObs:
        K = dkObs*(KObs+1)
      else: raise TypeError('Unable to interpret time setup')
    K = int(ceil(K/dkObs)*dkObs)

    # "State vars"
    self._dt      = dt
    self._dkObs   = dkObs
    self._K       = K

    # BurnIn, Tplot
    assert self.T >= BurnIn, "Experiment duration < BurnIn time"
    self.BurnIn = BurnIn
    if Tplot is None:
      Tplot = BurnIn
    self.Tplot = Tplot # don't enforce <T here

    assert len(self.kkObs) == self.KObs+1



  ######################################
  # "State vars". Can be set (changed).
  ######################################
  @property
  def dt(self):
    return self._dt
  @dt.setter
  def dt(self,value):
    dkObs_new = self.dkObs * self.dt/value
    if not np.isclose(int(dkObs_new), dkObs_new):
      raise ValueError('New value is amgiguous with respect to dkObs')
    dkObs_new = int(dkObs_new)
    self.__init__(dt=value,dkObs=dkObs_new,T=self.T,BurnIn=self.BurnIn,Tplot=self.Tplot)
  @property
  def dkObs(self):
    return self._dkObs
  @dkObs.setter
  def dkObs(self,value):
    ratio = value/self.dkObs
    self.__init__(dt=self.dt,dkObs=value,T=ratio*self.T,BurnIn=self.BurnIn,Tplot=self.Tplot)
  @property
  def K(self):
    return self._K
  @K.setter
  def K(self,value):
    self.__init__(dt=self.dt,dkObs=self.dkObs,K=value,BurnIn=self.BurnIn,Tplot=self.Tplot)

  ######################################
  # Read/write (though not state var)
  ######################################
  @property
  def T(self):
    return self.dt*self.K
  @T.setter
  def T(self,value):
    self.__init__(dt=self.dt,dkObs=self.dkObs,T=value,BurnIn=self.BurnIn,Tplot=self.Tplot)

  @property
  def KObs(self):
    return int(self.K/self.dkObs)-1
  @KObs.setter
  def KObs(self,value):
    self.__init__(dt=self.dt,dkObs=self.dkObs,KObs=value,BurnIn=self.BurnIn,Tplot=self.Tplot)

  ######################################
  # Read-only
  ######################################
  @property
  def dtObs(self):
    return self.dkObs*self.dt

  @property
  def kk(self):
    return arange(self.K+1)
  @property
  def kkObs(self):
    return self.kk[self.dkObs::self.dkObs]

  @property
  def tt(self):
    return self.kk * self.dt
  @property
  def ttObs(self):
    return self.kkObs * self.dt

  # Burn In. NB: uses > (strict inequality)
  @property
  def mask_BI(self):
    "Example use: kk_BI = kk[mask_BI]"
    return self.tt > self.BurnIn
  @property
  def maskObs_BI(self):
    "Example use: kkObs_BI = kkObs[maskObs_BI]"
    return self.ttObs > self.BurnIn

  ######################################
  # Other
  ######################################
  @property
  def ticker(self):
    """"Fancy version of range(1,K+1).

    Also yields t, dt, and kObs.
    """
    tckr = Ticker(self.tt,self.kkObs)
    next(tckr)
    return tckr

  def cycle(self,kObs):
    """The range (in kk) between observation kObs-1 and kObs.

    Also yields t and dt.
    """
    for k in kObs * self.dkObs + arange(1,self.dkObs+1):
      t  = self.tt[k]
      dt = t - self.tt[k-1]
      yield k, t, dt

  def __str__(self):
    printable = ['K','KObs','T','BurnIn','dtObs','dt']
    return str(AlignedDict([(k, getattr(self,k)) for k in printable]))
  def __repr__(self):
    return repr_type_and_name(self) + "\n" + str(self)

  ######################################
  # Utilities
  ######################################
  def copy(self):
    "Copy via state vars."
    return Chronology(dt=self.dt,dkObs=self.dkObs,K=self.K,BurnIn=self.BurnIn)
  def __eq__(self, other):
    if isinstance(other, self.__class__):
      return self.__dict__ == other.__dict__
    return False
  def __ne__(self, other):
    return not self.__eq__(other)


class Ticker:
  """Iterator over kk and kkObs, yielding (k,kObs,t,dt).

  Includes __len__ for progressbar usage.

  kObs = kkObs.index(k), or None otherwise,
  but computed without this repeated look-up operation.
  """
  def __init__(self, tt, kkObs):
    self.tt  = tt
    self.kkO = kkObs
    self.reset()
  def reset(self):
    self.k   = 0
    self._kO = 0
    self.kO  = None
  def __len__(self):
    return len(self.tt) - self.k
  def __iter__(self): return self
  def __next__(self):
    if self.k >= len(self.tt):
      raise StopIteration
    t    = self.tt[self.k]
    dt   = t - self.tt[self.k-1] if self.k > 0 else np.NaN
    tple = (self.k,self.kO,t,dt)
    self.k += 1
    if self._kO < len(self.kkO) and self.k == self.kkO[self._kO]:
      self.kO = self._kO
      self._kO += 1
    else:
      self.kO = None
    return tple


def format_time(k,kObs,t):
  if k==None:
    k    = "init"
    t    = "init"
    kObs = "N/A"
  else:
    t    = "   t=%g"%t
    k    = "   k=%d"%k
    kObs = "kObs=%s"%kObs
  s = "\n".join([t,k,kObs])
  return s


