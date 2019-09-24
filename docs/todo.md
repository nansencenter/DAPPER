Todo
================================================
* Simplify time management?
* Use pandas for stats time series?

* Bugs:
		* Window focus.
    * __name__ for HMM via inspect fails when running a 2nd, ≠ script.
		* Is this why ctrl-c fails so often (from https://docs.python.org/3.5/library/select.html ):
				"Changed in version 3.5:
				The function is now retried with a recomputed timeout when interrupted by a signal,
				except if the signal handler raises an exception (see PEP 475 for the rationale),
				instead of raising InterruptedError."

## DAC changes:
Try making something like the following
to deal with:
  * Use inspect somehow instead of C.update_setting
  * EITHER: Rm *args, **kwargs from da_methods? (less spelling errors)
  *         Replace with opts argument (could be anything).
  *     OR: implement warning if config argument was not used
                        (warns against misspelling, etc)

  * Use AlignedDict for DA_Config's repr?
  * Make function DA_Config() a member called 'method' of DAC. Rename DAC to DA_Config.
      => Yields (???) decorator syntax @DA_Config.method  (which would look nice) 
  * Rename DA_Config to DA_method or something
  * pass name from DAC_list to tqdm in assimilator?
  * Get rid of Tuple assignment of twin/setup items

class DA_Header:
   __init__(self,*args, **kwargs):
       write args to self

@wraps(EnKF)
EnKF_config(*args, **kwargs):
    super().__init__()
    assimilator = EnKF(*args,**kwargs)
   
def EnKF(stats,upd_a, N, infl):
    assimilate and assess
    return stats





