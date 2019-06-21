Todo
================================================
* Make changes to L63.core and sak12 in the rest
* rename viz.py:span to "xtrma". Or rm?
* Rename partial_direct_Obs -> partial_Id_Obs
* Rename dfdx -> dstep_dx
* Rename TLM -> d2x_dtdx
* Rename jacob -> linearization
* Ensure plot_pause is used for all liveplotting
* Is this why ctrl-c fails so often (from https://docs.python.org/3.5/library/select.html ):
    "Changed in version 3.5: The function is now retried with a recomputed timeout when interrupted by a signal, except if the signal handler raises an exception (see PEP 475 for the rationale), instead of raising InterruptedError."
* avoid tqdm multiline (https://stackoverflow.com/a/38345993/38281) ???
  No, INSTEAD: capture key press and don't send to stdout

* Bugs:
    * __name__ for HMM via inspect fails when running a 2nd, ≠ script.

* EITHER: Rm *args, **kwargs from da_methods? (less spelling errors)
*         Replace with opts argument (could be anything).
*     OR: implement warning if config argument was not used
		      (warns against misspelling, etc)
* Use inspect somehow instead of C.update_setting
* pass name from DAC_list to tqdm in assimilator?
* Use AlignedDict for DA_Config's repr?
* Make function DA_Config() a member called 'method' of DAC. Rename DAC to DA_Config.
    => Yields (???) decorator syntax @DA_Config.method  (which would look nice) 
* Rename DA_Config to DA_method or something
* Get rid of Tuple assignment of twin/setup items

* Reorg file structure: Turn into package?
* Rename common to DAPPER_workspace.
* Welcome message (setting mkl.set_num_threads, setting style, importing from DAPPER "setpaths") etc
* Defaults file (fail_gently, liveplotting, store_u mkl.set_num_threads, print options in NestedPrint, computational_threshold)
* Post version on norce, nersc and link from enkf.nersc

* Darcy, LotkaVolterra, 2pendulum, Kuramoto-Sivashinsky , Nikolaevskiy Equation
* Simplify time management?
* Use pandas for stats time series?



