# Nonlinearity vs divergence:
# - growth of covar perturbations in (superlinearly) divergent flow.
# - palatella argued that close-to-neutral modes are affected by mixing.
#   \citetalias{bocquet2015expanding} showed that these are the ones that
#   that receive the most fixing by inflation.
# Here we check if nonlinearity can "refresh the directions" of the ensemble,
# such that they may give some coverage to subspace outside of the highest
# (N-1) BLVs.
# If so, then the precipice of the RMSE vs N graph should be
# less vertical for the nonlinear case than for the linear case.


##############################
# Plotting 
##############################
# Note: only uses saved data => could be run in separate session.

#ResultsTable(save_path).plot_mean_field('rmse_a')

R = ResultsTable('data/remote/AdInf/Can_NonLin_avoid_Divergence/run3')
R.rm('LETKF')
R.load('data/remote/AdInf/Can_NonLin_avoid_Divergence/run4')
R.plot_mean_field('rmse_a')
plt.yscale('log')


