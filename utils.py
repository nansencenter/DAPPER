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
  #progbar = lambda inds: tqdm.tqdm(inds, desc="Assim.", leave=1)
  def progbar(inds, desc="Assim.",leave=1):
    return tqdm.tqdm(inds,desc=desc,leave=leave)
except ImportError:
  progbar = lambda inds: myprogbar(inds, desc="Assim.")
  #def progbar(inds, desc="Assim.",leave=1):
    #return myprogbar(inds,desc)

from os import popen
def set_np_linewidth():
  rows, columns = popen('stty size', 'r').read().split()
  np.set_printoptions(linewidth=int(columns)-1)



#stackoverflow.com/q/292095
import select
def poll_input():
  i,o,e = select.select([sys.stdin],[],[],0.0001)
  for s in i: # Only happens if <Enter> has been pressed
    if s == sys.stdin:
      return sys.stdin.readline()
  return None

# Can't get thread solution working (in combination with getch()):
# stackoverflow.com/a/25442391/38281

# stackoverflow.com/a/21659588/38281
# (Wait for) any key:
def _find_getch():
    try:
        import termios
    except ImportError:
        # Non-POSIX. Return msvcrt's (Windows') getch.
        import msvcrt
        return msvcrt.getch
    # POSIX system. Create and return a getch that manipulates the tty.
    import sys, tty
    def _getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    return _getch
getch = _find_getch()

