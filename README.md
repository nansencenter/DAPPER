This branch of DAPPER provides reproduces the benchmark results plotted in `Figure 1` of the article


    "Revising the stochastic iterative ensemble smoother"
    by Patrick N. Raanes, Geir Evensen, and Andreas S. Stordal


DAPPER may be installed as described
[for in the master branch (accessed July 2019)](https://github.com/nansencenter/DAPPER#installation). 

To reproduce the benchmark data points:
1. Execute: `git checkout paper_StochIEnS` to view this branch.
2. Run the script: `python bench_L95.py N`  
   (based on `example_3.py`, which has more comments).
