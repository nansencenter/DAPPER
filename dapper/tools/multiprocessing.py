from dapper import *

import textwrap

# Multiprocessing is kept as an option in DAPPER,
# since some of these libraries are experimental.
# TODO: make it mandatory?
try:
  # Multiprocessing requries pickling. The package 'dill' is able to
  # pickle much more than basic pickle (e.g. nested functions),
  # and is being used by 'multiprocessing_on_dill'.
  # Alternatively, the package pathos also enables multiprocessing with dill.
  import multiprocessing_on_dill as mpd
  
  # Deciding on core numbers
  from psutil import cpu_percent, cpu_count

  # Enforcing individual core usage.
  # Issue: numpy uses multiple cores (github.com/numpy/numpy/issues/11826).
  #   This may yield some performance gain, but typically not much
  #   compared to manual parallelization over independent experiments,
  #   ensemble forecasts simulations, or local analyses.
  # Therefore: force numpy to only use a single core.
  #
  # Ref stackoverflow.com/a/53224849
  # for envar in [
  #     "OMP_NUM_THREADS",        # openmp
  #     "OPENBLAS_NUM_THREADS",   # openblas
  #     "MKL_NUM_THREADS",        # mkl
  #     "VECLIB_MAXIMUM_THREADS", # accelerate
  #     "NUMEXPR_NUM_THREADS"]:   # numexpr
  #     os.environ[envar] = "1"
  # The above may be the safest way to limit thread use on all systems,
  # but requires importing before np. => Instead, use threadpoolctl!
  #
  # >>> import numpy as np
  # >>> from threadpoolctl import threadpool_limits
  # >>> N  = 4*10**3
  # >>> a  = np.random.randn(N, N)
  # >>> # Now start monitoring CPU usage (with e.g. htop).
  # >>> with threadpool_limits(limits=1, user_api='blas'):
  # >>>   a2 = a @ a
  import threadpoolctl
  threadpoolctl.threadpool_limits(1)

  no_MP = False
except ImportError:
  no_MP = True

  def MP_warn():
    if not MP_warn.have_printed: # only print once!
      print(textwrap.dedent("""
        Warning: Multiprocessing (MP) was requsted during execution,
        but has not been properly installed.
        Try re-installing DAPPER with `pip install -e <path-to-DAPPER>[MP]`.
        """))
      MP_warn.have_printed = True
  MP_warn.have_printed = False


#########################################
# Multiprocessing
#########################################

if no_MP:
  def multiproc_map(func,xx,**kwargs):
    MP_warn()
    return [func(x, **kwargs) for x in xx]

else:
    def multiproc_map(func,xx,**kwargs):
      """A parallelized version of map.

      Similar to::

        result = [func(x, **kwargs) for x in xx]

      Note: unlike reading, writing "in-place" does not work with multiprocessing
      (unless "shared" arrays are used, but this has not been tried out here).

      See example use in mods/QG/core.py.

      Technicalities dealt with:
       - passing kwargs
       - join(), close()

      However, the main achievement of this helper function is to make
      "Ctrl+C", i.e. KeyboardInterruption,
      stop the execution of the program, and do so "gracefully",
      something which is quite tricky to achieve with multiprocessing.
      """

      # The Ctrl-C issue is mainly cosmetic, but an annoying issue. E.g.
      #  - Pressing ctrl-c should terminate execution.
      #  - It should only be necessary to press Ctrl-C once.
      #  - The traceback does not extend beyond the multiprocessing management
      #    (not into the worker codes), and should therefore be cropped before then.
      #
      # NB: Here be fuckin dragons!
      # This solution is mostly based on stackoverflow.com/a/35134329.
      # I don't (fully) understand why the issues arise,
      # nor why my patchwork solution somewhat works.
      #
      # I urge great caution in modifying this code because
      # issue reproduction is difficult (coz behaviour depends on
      # where the execution is currently at when Ctrl-C is pressed)
      # => testing is difficult.
      #
      # Alternative to try: 
      # - Use concurrent.futures, as the bug seems to have been patched there:
      #   https://bugs.python.org/issue9205. However, does this work with dill?
      # - Multithreading: has the advantage of sharing memory,
      #   but was significantly slower than using processes,
      #   testing on DAPPER-relevant.


      # Ignore Ctrl-C.
      # Alternative: Pool(initializer=[ignore sig]).
      # But the following way seems to work better.
      import signal
      orig = signal.signal(signal.SIGINT, signal.SIG_IGN)

      # Setup multiprocessing pool (pool workers should ignore Ctrl-C)
      NPROC = None # None => multiprocessing.cpu_count()
      pool = mpd.Pool(NPROC)

      # Restore Ctrl-C action
      signal.signal(signal.SIGINT, orig)

      try:
        f = functools.partial(func,**kwargs) # Fix kwargs

        # map vs imap: stackoverflow.com/a/26521507
        result = pool.map(f,xx) 

        # Relating to Ctrl-C issue, map_async was preferred: stackoverflow.com/a/1408476
        # However, this does not appear to be necessary anymore...
        # result = pool.map_async(f, xx)
        # timeout = 60 # Required for get() to not ignore signals.
        # result = result.get(timeout)

      except KeyboardInterrupt as e:
        try:
          pool.terminate()
          # Attempts to propagate "Ctrl-C" with reasonable traceback print:
          # ------------------------------------------------------------------
          # ALTERNATIVE 1: ------- shit coz: only includes multiprocessing trace.
          # traceback.print_tb(e.__traceback__,limit=1)
          # sys.exit(0)
          # ALTERNATIVE 2: ------- shit coz: includes multiprocessing trace.
          # raise e
          # ALTERNATIVE 3: ------- shit coz: includes multiprocessing trace.
          # raise KeyboardInterrupt
          # ALTERNATIVE 4:
          was_interrupted = True
        except KeyboardInterrupt as e2:
          # Sometimes the KeyboardInterrupt caught above just causes things to hang,
          # and another "Ctrl-C" is required, which is then caught by this 2nd try-catch.
          pool.terminate()
          was_interrupted = True
      else:
        # Resume normal execution
        was_interrupted = False
        pool.close() # => Processes will terminate once their jobs are done.

      try:
        # Helps with debugging,
        # according to stackoverflow.com/a/38271957
        pool.join() 
      except KeyboardInterrupt as e:
        # Also need to handle Ctrl-C in join()...
        # This might necessitate pressing Ctrl-C again, but it's
        # better than getting spammed by traceback full of garbage.
        pool.terminate()
        was_interrupted = True

      # Start the KeyboardInterrupt trace here.
      if was_interrupted:
        raise KeyboardInterrupt

      return result


#########################################
# Writing / Loading Independent experiments
#########################################
# Multiprocessing, but for scripts,
# and launching the processes (scripts) outside of python.

import subprocess
def distribute(script,sysargs,xticks,prefix='',nCore=0.99,xCost=None):
  """Parallelization.

  Runs 'script' either as master, worker, or stand-alone,
  depending on 'sysargs[2]'.

  Return corresponding
   - portion of 'xticks'
   - portion of 'rep_inds' (setting repeat indices)
   - save_path.

  xCost: The computational assumed: [(1-xCost) + xCost*S for S in xticks].
  This controls how the xticks array gets distributed to nodes.

   - Set xCost to 0 for uniform distribution of xticks array.
   - Set xCost to 1 if the costs scale linearly with the setting.
  """

  import glob
  def get_numbering(glb):
    ls = glob.glob(glb+'*')
    return [int(re.search(glb+'([0-9]*).*',f).group(1)) for f in ls]

  def prep_run():
    "Call save_dir, and create (and reserve) path (with its prefix and RUN)"
    path  = save_dir(script)
    # Add RUN part
    path += prefix+'_' if prefix else ''
    path += 'run'
    RUN   = str(1 + max(get_numbering(path),default=0))
    path += RUN
    print("Will save to",path+"...")
    # Reserve filename
    subprocess.run(['touch',path]) 
    return path, RUN

  # Make running count (rep_inds) of repeated xticks.
  # This is typically used to modify the experiment seeds.
  rep_inds = [ list(xticks[:i]).count(x) for i,x in enumerate(xticks) ]

  # Set run_type
  if len(sysargs)<=2:
      run_type = 'SIMPLE'
  else:
      run_type = sysargs[2].upper()
      if run_type in ['TMP', 'WORKER']:
        pass # i.e. this is a valid run_type
      elif 'PARA' in run_type or 'DIST' in run_type:
        if no_MP:
          MP_warn()
          run_type = 'SIMPLE'
        else:
          run_type = 'MASTER'
      else:
        raise ValueError('%s is not a valid run type.'%sysargs[2])


  # Switch(run_type)
  if   run_type == 'SIMPLE':
      save_path, _ = prep_run()

  elif run_type == 'TMP':
      save_path = os.path.join(dirs['data'],'tmp_data')

  elif run_type == 'MASTER':
      save_path, RUN = prep_run()

      # THIS SECTION CAN BE MODIFIED TO YOUR OWN QUEUE SYSTEM, etc.
      # ------------------------------------------------------------
      # The implemention here-below does not use any queing system,
      # but simply launches a bunch of processes.
      # It uses (gnu's) screen for coolness, coz then individual
      # worker progress and printouts can be accessed using 'screen -r'.

      # screenrc path. This config is the "master".
      rcdir = os.path.join(dirs['data'],'screenrc')
      os.makedirs(rcdir, exist_ok=True)
      screenrc  = os.path.join(rcdir,'tmp_screenrc_')
      screenrc += os.path.split(script)[1].split('.')[0] + '_run'+RUN

      HEADER = """
      # Auto-generated screenrc file for experiment parallelization.
      source $HOME/.screenrc
      screen -t bash bash # make one empty bash session
      """.replace('/',os.path.sep)
      HEADER = textwrap.dedent(HEADER)
      # Other useful screens to launch
      #screen -t IPython ipython --no-banner # empty python session
      #screen -t TEST bash -c 'echo nThread $MKL_NUM_THREADS; exec bash'

      # Decide number of batches (i.e. processes) to run.
      if isinstance(nCore,float): # interpret as ratio of total available CPU
        nBatch = round( nCore * (1 - cpu_percent()/100) * cpu_count() )
      else:       # interpret as number of cores
        nBatch = min(nCore, cpu_count())

      # Write workers to screenrc
      with open(screenrc,'w') as f:
        f.write(HEADER)
        for i in range(nBatch):
          iWorker = i + 1 # start indexing from 1
          f.write('screen -t W'+str(iWorker)+' ipython -i --no-banner '+
              ' '.join([script,sysargs[1],'WORKER',str(iWorker),str(nBatch),save_path])+'\n')
          # sysargs:      0        1         2            3        4            5
        f.write("")
      sleep(0.2)
      # Launch
      subprocess.run(['screen', '-dmS', 'run'+RUN,'-c', screenrc])
      print("Experiments launched. Use 'screen -r' to view their progress.")
      sys.exit(0)

  elif run_type == 'WORKER':
      # Parse worker's sysargs
      iWorker = int(sysargs[3])
      nBatch  = int(sysargs[4])
      nBatch  = min(nBatch, len(xticks))
      if iWorker-1 >= nBatch:
        sys.exit(0)

      # xCost defaults
      if xCost==None:
        if prefix=='N':
          xCost = 0.02
        elif prefix=='F':
          xCost = 0

      # Split xticks array to this "worker":
      if xCost==None:
        # Split uniformly
        xticks   = np.array_split(xticks  ,nBatch)[iWorker-1]
        rep_inds = np.array_split(rep_inds,nBatch)[iWorker-1]
      else:
        # Weigh xticks by costs, before splitting uniformly
        eps = 1e-6                       # small number
        cc  = (1-xCost) + xCost*xticks   # computational cost,...
        cc  = np.cumsum(cc)              # ...cumulatively
        cc /= cc[-1]                     # ...normalized
        # Find index dividors between cc such that cumsum deltas are approx const:
        divs     = array([find_1st_ind(cc>c+1e-6) for c in linspace(0,1,nBatch+1)])
        divs[-1] = len(xticks)
        # The above partition may be "unlucky": fix by some ad-hoc post-processing.
        divs[[i+1 for i, d in enumerate(diff(diff(divs))) if d>0]] += 1
        # Select iWorker's part.
        xticks   = array(xticks)  [divs[iWorker-1]:divs[iWorker]]
        rep_inds = array(rep_inds)[divs[iWorker-1]:divs[iWorker]]

      print("xticks partition index:",iWorker)
      print("=> xticks array:",xticks)

      # Append worker index to save_path
      save_path = sysargs[5] + '_W' + str(iWorker)
      print("Will save to",save_path+"...")
      
  return xticks, save_path, rep_inds




