from addons import *
from tools.viz import plot_benchmark_analysis

#then run the whole.
if __name__=='__main__':

	b=Benchmark(config=EnKF('Sqrt', N=50, infl=1.01, rot=True, liveplotting=False),tunning=False)
	p=b.setup.h.m

	#####################################
	### Build the bench #################
	#####################################
	"""
	b+=MARKOV(size=p,thin=0.8)

	b+=MARKOV(size=p,thin=0.6)

	b+=MARKOV(size=p,thin=0.4)

	b+=MARKOV(size=p,thin=0.2)
	"""

	for i in range(1,19):
		b+=MultiDiag(size=p,Rinfl=i/10)

	"""

	b+=SOAR(size=p,thin=0.1)

	b+=SOAR(size=p,thin=0.05)

	b+=SOAR(size=p,thin=0.01)

	b+=SOAR(size=p,thin=0.005)

	b+=SOAR(size=p,thin=0.001)

	#b+=MultiDiag(size=p,Rinfl=1.3)

	b+=MultiDiag(size=p,diags=3)

	b+=MultiDiag(size=p,diags=5)

	b+=BlockDiag(size=p,submat=MARKOV(size=5))

	b+=BlockDiag(SOAR(size=p//2), MARKOV(size=p//2))

	b+=MultiDiag(size=p,diags=5,decay=2)

	b+=Custom(size=p,f=lambda x,y: 1*(abs(x-y)<2) )

	b+=Sparse(size=p)
	
	b+=SOAR(size=p)

	b+=SOAR(size=p,thin=0.8)

	b+=SOAR(size=p,thin=0.6)

	b+=SOAR(size=p,thin=0.4)

	b+=SOAR(size=p,thin=0.2)

	b+=Sparse(size=p)
	"""

	####################################
	############ Run ###################
	####################################

	b.run()

	####################################
	############ Save ##################
	####################################	

	b.save()

	####################################
	############ Plot ##################
	####################################
	plot_benchmark_analysis(b)




