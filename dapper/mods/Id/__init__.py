"""The identity model (that does nothing, i.e. sets `output = input`).

This means that the state dynamics are just Brownian motion.

Next to setting the state to a constant, this is the simplest model you can think of.
"""
import dapper.mods as modelling

tseq = modelling.Chronology(1, dko=1, Ko=2000, Tplot=10, BurnIn=0)
M = 4
Obs = {'noise': 2, 'M': M}
Dyn = {'noise': 1, 'M': M}
X0 = modelling.GaussRV(C=1, M=M)

HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)

#########################
#  Benchmarking script  #
#########################
# import dapper as dpr
# dpr.rc.field_summaries.append("ms")

# # We do not include Climatology and OptInterp because their variance and accuracy
# # are less interesting since they grow with the duration of the experiment.
# import dapper.da_methods as da
# xps = dpr.xpList()
# xps += da.Var3D("eye", xB=2)
# xps += da.ExtKF()
# xps += da.EnKF('Sqrt', N=100)

# save_as = xps.launch(HMM, save_as=False)

# # The theoretic (expected) analysis (resp. forecast) mean-square error is 1 (resp. 2),
# # as follows from the KF/Ricatti equations with DynMod=ObsMod=Id, R=2, Q=1. Verify:
# print(xps.tabulate_avrgs(['err.ms.a', 'spread.ms.a', 'err.ms.f', 'spread.ms.f']))

# # The RMS errors do not have the `sqrt` of the above as their expected values,
# # since this is a nonlinear function, which does not interchange with the averaging
# # (still, you can make the discrepancy negligible by setting Nx to be large).
# # By contrast, the spread (expected variances) is not subject to this effect,
# # since it is always equal/homogeneous/isotropic.
# print(xps.tabulate_avrgs(['rmse.a', 'rmv.a', 'rmse.f', 'rmv.f']))
