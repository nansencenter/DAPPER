This branch of DAPPER reproduces the benchmark results plotted in Figure 4 (a,b,c,d) from the article
    
    
    "Adaptive covariance inflation in the ensemble Kalman filter by Gaussian scale mixtures"
    by Patrick N. Raanes, Marc Bocquet, and Alberto Carrassi.


DAPPER may be installed as described
[for in the master branch (accessed July 2019)](https://github.com/nansencenter/DAPPER#installation). 

To reproduce the benchmark data points:
1. Do: `git checkout paper_AdInf` to view this branch.
2. Do: `cd AdInf`
2. Run the desired script, doing one of:
   * Figure 4a: `python bench_LUV.py F`
   * Figure 4b: `python bench_LUV.py c`
   * Figure 4c: `python bench_L95.py F`
   * Figure 4d: `python bench_L95.py Q`
   All scripts are based on `example_3.py`; refer to this for more detailed commenting.
