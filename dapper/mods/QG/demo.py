# Script to illustrate the QG (quasi-geostrophic) model.

from dapper import *
from dapper.mods.QG.core import sample_filename, nx, square, default_prms


def show(x0,psi=True,ax=None):
  #
  def psi_or_q(x):
    return x if psi else compute_q(x)
  #
  if ax==None:
    fig, ax = plt.subplots()

  im = ax.imshow(psi_or_q(square(x0)))

  if psi: im.set_clim(-30,30)
  else:   im.set_clim(-28e4,25e4)

  def update(x):
    im.set_data(psi_or_q(square(x)))
    plt.pause(0.01)
  return update


# Although psi is the state variable, q looks cooler.
# q = Nabla^2(psi) - F*psi.
import scipy.ndimage.filters as filters
dx = 1/(nx-1)
def compute_q(psi):
  Lapl = filters.laplace(psi,mode='constant')/dx**2
  # mode='constant' coz BCs are: psi = nabla psi = nabla^2 psi = 0
  return Lapl - default_prms['F']*psi


###########
# Main
###########
fig, (ax1,ax2) = plt.subplots(ncols=2,sharex=True,sharey=True,figsize=(8,4))
for ax in (ax1,ax2): ax.set_aspect('equal',adjustable_box_or_forced())
ax1.set_title(r'$\psi$')
ax2.set_title('$q$')

xx = np.load(sample_filename)['sample'] 
setter1 = show(xx[0],psi=True ,ax=ax1)
setter2 = show(xx[0],psi=False,ax=ax2)

for k, x in progbar(list(enumerate(xx)),"Animating"):
  if k%2==0:
    setter1(x)
    setter2(x)
    fig.suptitle("k: "+str(k))
    plt.pause(0.01)


