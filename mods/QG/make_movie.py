# NB For saving animation, this should be
# run without having done: plt.ion().
# Therefore pwd should probably not be DAPPER

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import scipy.io
from mods.QG.f90 import QG

mat = scipy.io.loadmat('mods/QG/f90/QG_samples-12.mat')
S = mat['S']
m, n_samples = S.shape
nx = int(np.sqrt(m))

def square(x):
  psi = x.copy()
  psi = psi.reshape((nx,nx),order='F')
  return psi

def flatten(psi):
  x = psi.ravel(order='F')
  return x

psi0 = square(S[:,-1])

K = 20

pp    = np.zeros((K,m))
pp[0] = flatten(psi0)

f,ax = plt.subplots()
ims  = []

for k in range(1,K):
  psi = square(pp[k-1])
  t   = np.array([0.0])
  QG.interface_mod.step(t,psi,'mods/QG/f90/pat.txt')
  pp[k] = flatten(psi)

  im = plt.imshow(psi, animated=True)
  ims.append([im])


ani = animation.ArtistAnimation(f, ims, interval=20, blit=False, repeat_delay=1000)

#ani.save('dynamic_images.mp4')

#plt.show()

