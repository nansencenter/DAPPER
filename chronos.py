# Time sequence management

from common import *

class Chronology:
  """Time ledgers with consistency checks.

  Uses integer records => tt[k] == k*dt.
  Uses generators => time series may be arbitrarily long.

  Example:
                       <-----dtObs----->
                <---dt--->
  tt:    0.0    0.2    0.4    0.6    0.8    1.0    T
  kk:    0      1      2      3      4      5      K
         |------|------|------|------|------|------|
  kObs:  No     No     0      No     1      No     KObs
  kkObs:               2             4             6
                       <----dkObs----->

  I.e. no obs at 0 coz it doesn't fit with convention,
  which is hardcorded in DA code,
  whose cycling conventionally starts by forecasting.

  Identities:
    len(kk)    = len(tt)    = K   +1
    len(kkObs) = len(ttObs) = KObs+1
  and
    kkObs[0]  = dkObs = dtObs/dt = T/(KObs+1)
    kkObs[-1] = K     = T/dt
    KObs      = T/dtObs-1

  The variables are not protected, but should not be changed,
  nor are they implemented as properties.
  => changes requires reinitialization the constructor.
  Why: the ambiguity involved in e.g. >>> tseq.dtObs = 2*tseq.dtObs
  is too big. Should T also be doubled?
  """
  def __init__(self,dt=None,dtObs=None,T=None,BurnIn=-1, \
      dkObs=None,KObs=None,K=None):

    assert 3 == [dt,dtObs,T,dkObs,KObs,K].count(None) , \
        'Chronology is specified using exactly 3 parameters.'

    # Reduce all to dt,dkObs,K
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
    assert(isinstance(dkObs,int))
    if not K:
      if T:
        K = round(T/dt)
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

    # TODO: Prevent these from being written
    self.kk      = arange(K+1)
    self.kkObs   = self.kk[dkObs::dkObs]

    # TODO: Prevent these from being written
    self.dtObs    = dkObs*dt
    #self.T        = dt*K
    self.KObs     = int(K/dkObs)-1
    assert self.KObs == len(self.kkObs)-1

    # Burn In
    if self.T <= BurnIn: raise ValueError('BurnIn > T')
    self.BurnIn  = BurnIn
    self.kBI     = find_1st_ind(self.tt    > self.BurnIn)
    self.kObsBI  = find_1st_ind(self.ttObs > self.BurnIn)
    self.ttBI    = self.tt[self.kBI:]
    self.kkBI    = self.kk[self.kBI:]
    self.kkObsBI = self.kkObs[self.kObsBI:]
    self.ttObsBI = self.ttObs[self.kObsBI:]

  # "State vars"
  @property
  def dt(self):
    return self._dt
  @dt.setter
  def dt(self,value):
    self.__init__(dt=value,dkObs=self.dkObs,K=self.K,BurnIn=self.BurnIn)
  @property
  def dkObs(self):
    return self._dkObs
  @dkObs.setter
  def dkObs(self,value):
    self.__init__(dt=self.dt,dkObs=value,K=self.K,BurnIn=self.BurnIn)
  @property
  def K(self):
    return self._K
  @K.setter
  def K(self,value):
    self.__init__(dt=self.dt,dkObs=self.dkObs,K=value,BurnIn=self.BurnIn)

  # Read-only
  @property
  def tt(self):
    return self.kk * self.dt
  @property
  def ttObs(self):
    return self.kkObs * self.dt

  # Read/write (but non-state var)
  @property
  def T(self):
    return self.dt*self.K
  @T.setter
  def T(self,value):
    self.__init__(dt=self.dt,dkObs=self.dkObs,T=value,BurnIn=self.BurnIn)

  @property
  def forecast_range(self):
    """"Fancy version of range(1,K+1),
    which also provides t, dt, and kObs"""
    tckr = Ticker(self.tt,self.kkObs)
    next(tckr)
    return tckr

  def DAW_range(self,kObs):
    """The range of the kObs-th data assimilation window (DAW).
    Also yields t and dt."""
    for k in kObs * self.dkObs + arange(1,self.dkObs+1):
      t  = self.tt[k]
      dt = t - self.tt[k-1]
      yield k, t, dt

  def __str__(self):
    s = []
    printable = ['K','KObs','dtObs','dt','T','BurnIn']
    w = 4 + max(len(s) for s in printable)
    for k in printable:
      s.append('{0:>{1}}: '.format(k,w) + str(getattr(self,k)))
    return '\n'.join(s)

  def __repr__(self):
      return self.__str__()


class Ticker:
  """ Iterator over kk, AND kkObs,
  the latter being handled without repeated look-ups.
  Includes __len__ for progressbar usage.
  Provides additional params: k,dt."""
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



