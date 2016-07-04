# Utilities (non-math)

from common import *

def myprogbar(itrble, desc='Prog.'):
  L  = len(itrble)
  print('{}: {: >2d}'.format(desc,0),end='')
  for k,i in enumerate(itrble):
    yield i
    p = (k+1)/L
    e = '' if k<(L-1) else '\n'
    print('\b\b\b\b {: >2d}%'.format( \
        int(100*p)),end=e)
    sys.stdout.flush()

try:
  import tqdm
  progbar = lambda inds: tqdm.tqdm(inds, desc="Assim.", leave=1)
except ImportError:
  progbar = lambda inds: myprogbar(inds, desc="Assim.")

from os import popen
def set_np_linewidth():
  rows, columns = popen('stty size', 'r').read().split()
  np.set_printoptions(linewidth=int(columns))



#stackoverflow.com/q/292095
#stackoverflow.com/a/25442391/38281
import select
def heardEnter():
  i,o,e = select.select([sys.stdin],[],[],0.0001)
  for s in i:
    if s == sys.stdin:
      input = sys.stdin.readline()
      return True
  return False







