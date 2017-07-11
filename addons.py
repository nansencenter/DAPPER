from common import *
# Lorenz95.sak08 is fully observated setup: see script -> h=id, forecast noise is zero, 
from mods.Lorenz95.sak08 import setup
#from mods.QG.sak08 import setup
import pickle



"""
The thinning methods could (and should) be built upon the transorm_by method of the CovMat class:
This would imply a big refactorization of this class and this approach: Indeed, the class ObsErrMat is 
redundant as the class CovMat already offers a lot of desired functionalities:
the class obserrmat should only build arrays, used as data for the CovMat class, and the covmat class should store the arguments
used to build the matrix (deltax lr threshold decay whatever they are).Cov mat class should be modified as well so that
the trunc argument could be accessed	not hidden behind self_trunc

Eg of the thinning method:
#define the mat:
mat=ObsErrMat('MultiDiagonal',diags=5)
C=CovMat(mat)
#The data can then be accessed through C.full (numpy.ndarray)
#modify experiment function so that
C.experiment(xx,yy,setup)
#eventually modify the list_of_matrices class. Benchmark class could remain the same (apparently, eventually add the __repr__ builtin method)

#If thinning is desired:
C.trunc=thinning_value
C.transform_by()
#This would as well require small modification of transform by in order to set the argument function as identity (f: x --> x)
"""


#Classes of usual matrices:
"""
class ObsErrMat():

	#Gathers the different kinds of correlation observation error matrices known so far
	#Default type is the identity matrix
	#weight argument is just the inflation factor

	def __init__(
		self
		,kind='MultiDiagonal'
		,size=1
		,deltax=1
		,Lr=1
		,threshold=1.0e-4
		,weight=1
		,diags=1
		,decay=2
		,f=lambda x,y:(x==y)*1
		,submat=ObsErrMat()
		,circulantvector=[1*(i==0) for i in range(1)]
		):

		#Just pick up the arguments to build the log
		frame=currentframe()
		args, _, _, values = getargvalues(frame)

		try :
			assert kind in ['MARKOV','SOAR','MultiDiagonal','BlockDiagonal','Custom','Sparse']
			#Messy definition of the block diagonal matrix, this bloc is necessary to avoid the missing arument exception
			#if any([type(a)==numpy.ndarray for a in args]):
			#	kind='BlockDiag'
			#Markov matrix as define in Stewart 2013, the threshold allows to set to zero the very small components of the matrix
			
			if kind=='MARKOV':
				#use deltax=lr
				A=zeros((size,size))
				desc='MARKOV( deltax='+str(deltax)+', Lr='+str(Lr)+', threshold='+str(threshold)+')'
				#this double loop could be optimized by a vector multiplication probably...
				for (i,j) in product(range(size),repeat=2):
					A[i,j]=exp(-abs(i-j)*deltax/Lr)*weight
					#Eventually truncate the matrix by removing the elements smaller than a fixed threshold
					if threshold>0:
						A[i,j]=A[i,j]*(A[i,j]>threshold)

			#Soar matrix as define in Stewart 2013, the threshold allows to set to zero the very small components of the matrix
			elif kind=='SOAR':
				A=zeros((size,size))
				desc='SOAR( deltax='+str(deltax)+', Lr='+str(Lr)+', threshold='+str(threshold)+')'
				#this double loop could be optimized by a vector multiplication probably...
				for (i,j) in product(range(size),repeat=2):
					A[i,j]=exp(-abs(i-j)*deltax/Lr)*(1+abs(i-j)*deltax/Lr)*weight
				#Eventually truncate the matrix by removing the elements smaller than a fixed threshold
					if threshold>0:
						A[i,j]=A[i,j]*(A[i,j]>threshold)

			#Multidiagonal matrix, can contain as much overdiagonals as desired
			elif kind=='MultiDiagonal':
				try:			
					assert diags%2==1
					assert (diags//2)*2<=size
					A=eye(size)
					if diags>1 and diags%2==1:
						for k in range(1,diags//2+1):
							A+=(eye(size,k=-k)+eye(size,k=k))/pow(decay,k)
					A=A*weight
					desc=str(diags)+'-diagonal matrix, inflated by '+str(weight)+' decay of '+str(decay)
				except AssertionError:
					print('uncorrect number of diagonals')

			#Block-diagonally defined matrix
			elif kind=='BlockDiagonal':
				B=submat.matrix
				q=B.shape[0]
				#t=submatkind
				n=size/q
				assert (n%1==0), 'bad definition'
				#S=ObsErrMat(kind=t,size=q).matrix
				A=zeros((size,size))
				#Build the matrix
				for k in range(int(n)):
					for (i,j) in product(range(q),repeat=2):
						A[k*q+i,k*q+j]=B[i,j]
				desc='Blockdiagonal matrix, of subsizes='+str(q)+', subtype '+str(submat.name)		

			#User defined function f:(i,j) -> A[i,j]
			elif kind=='Custom':
				A=zeros((size,size))
				desc='Custom defined matrix'
				for (i,j) in product(range(size),repeat=2):
					A[i,j]=f(i,j)

			#'sparse' matrix: 1 on the diagonal and in the corners and smoothed decrease in the centers of each sub and under matrix
			#Same behavior whatever the size is odd or even

			elif kind=='Sparse':
				A=zeros((size,size))
				desc='\'Sparse\' matrix (deltax='+str(deltax)+', Lr='+str(Lr)+' )'
				for (i,j) in product(range(size),repeat=2):
					A[i,j]=exp((abs(abs(i-j)-(size-1)/2)-(size-1)/2)*deltax/Lr)
					if threshold>0:
						A[i,j]=A[i,j]*(A[i,j]>threshold)

			elif kind=='Circulant':
				#Computation could be highly improved (time sortened) by storing only the circulant vector rather than the whole matrix Use of discrete Fourier transforms in the 1D-Var retrieval problem
					#By S. B. HEALY∗ and A. A. WHITE
				A=array([[circulantvector[-i:]+circulantvector[:-i]] for i in range(size)])


			#Store the name and the content of the matrix
			arguments=[(i,values[i]) for i in args if i not in ['f','self','size','submat']]+[('Subsize',submat.matrix.shape[0])]
			self.name=desc
			self.matrix=A
			self.arguments=arguments
			
		except AssertionError:
			print('Unknown matrix kind')

		

	def experiment(self,X,Y,config,setup):

		#timepoint reference
		start_time=time()
		#assimilate
		setup.h.noise.C=CovMat(self.matrix)
		stats=config.assimilate(setup,X,Y)
		#Close timewindow
		delta_t=time()-start_time

		#get rmse and spread through time, once BurnIn is done
		rmse_s=stats.rmse.a[setup.t.maskObs_BI]
		spread_s=mean(stats.mad.a,axis=1)[setup.t.maskObs_BI]
		timewindow=setup.t.T-setup.t.BurnIn

		#Compute the penalties !!! Need to be chosen more wisely !!!
		#Penalty is doubled if the algo underestimate the error committed
		#Penalty is the average error per step somehow
		penalty=sum(
			(
			(spread_s-rmse_s)*(spread_s>rmse_s)+ \
			2*(rmse_s-spread_s)*(rmse_s>spread_s)
			)*setup.t.dtObs
			)/timewindow

		#Get the average rmse and spread
		rmse=mean(rmse_s)
		spread=mean(spread_s)

		#Add this experiment to the output dataframe
		output=[('RMSE',rmse),('Spread',spread),('CPU_Time',delta_t),('Penalty',penalty)]+self.arguments
		return output

	def __repr__(self):
		a=[i[0] for i in self.arguments]
		b=[str(i[1]) for i in self.arguments]
		for i in range(len(a)):
			a[i]=str(a[i])+' '*(max(len(a[i]),len(b[i]))-len(a[i]))
			b[i]=str(b[i])+' '*(max(len(a[i]
				),len(b[i]))-len(b[i]))
		cols=' | '.join(a)
		inter='-'*len(cols)
		row=' | '.join(b)
		return cols+'\n'+inter+'\n'+row
"""

class List_of_matrices(list):

	def __init__(self,*args):
		for mat in args:
			self.append(mat)


	#Probably here add a decorator so that the added matrix is fit to the size requested by the setup
	def __iadd__(self,val):

		if not hasattr(val,'__iter__'):
			val = [val]
		for item in val:
			try:
				self.append(item)
			except RuntimeError:
				print('\nUndefined matrix')
		return self

	def __repr__(self):
		s=''
		for (i,a) in enumerate(self):
			s+=str(i+1)+'.'*4+'\n\t'+a.__str__().replace('\n','\n\t')
			s+='\n-\n'
		return s


class Benchmark(object):
	#Default inflation chosen on paper from sakov 2008

	def __init__(
		self,
		setup=setup,
		config=EnKF('Sqrt', N=50, infl=1.0, rot=True, liveplotting=False),
		Rt=MARKOV(size=setup.h.m,deltax=1,Lr=1),
		tunning=True,
		assimcycles=10**4
		):

		cols=['Setup','R','DA','RMSE','Spread','CPU_Time']

		#set the Rt: this noise will generate the observation
		setup.prepare_setup(Rt,assimcycles)
		self.Rt=Rt
		self._bench=List_of_matrices(Rt)
		self.setup=setup
		self.config=config
		self._output=DataFrame(columns=cols)
		self._is_paused=False
		self.tunning=tunning

		"""
		This is so far my best attempt to set default size depending on the current Benchmark in use.
		I of course stopped this investigations to avoid global state.

		Works fine at the first glance however.

		global MARKOV
		MARKOV=functools.partial(MARKOV,size=self.setup.h.m)
		"""

	#Be careful when changing the opti criterion. Need to change both *while* condition and the r or s returned and temp+-0.01.
	def assess_matrix(self,m,xx,yy):
		#Function aimed at tunning the inflation factor:
		temp=self.config.infl
		#Ugly as possible, should need rework

		r=dict(m.experiment(xx,yy,setup=self.setup,config=self.config))
		if self.tunning:
			#t=0
			s=r.copy()
			#Allow a more flexible criterion
			while r['RMSE']>=s['RMSE']*0.95: #or s['RMSE']>s['Spread']):
				#t+=1
				r=s.copy()
				temp+=0.01
				s=dict(m.experiment(xx,yy,setup=self.setup,config=self.config.update_settings(infl=temp)))
		return [('DA',self.config.update_settings(infl=temp-0.01))]+list(r.items())


	def run(self):
		#Simulate
		xx,yy=simulate(self.setup)
		#D=diag(diag(self.Rt.matrix)**0.5)

		#Go through the bench
		for (i,m) in enumerate(tqdm.tqdm(self._bench,desc='Benchmark')):
					#self.pause_run()
					if m.rk==m.m:
						r=self.assess_matrix(m,xx,yy)

						#now fill a pandas dataframe with results
						#Create a dictionary of all the useful information for the experiment
						row=dict([('Setup',self.setup.name),('R',m.__class__)]+r)

						if i==0:
							row['kind']+=' (Rt)'
						#Complete the row/dataframe
						for k in set(row.keys())-set(self._output.columns):
							self._output[k]=''
						for l in set(self._output.columns)-set(row.keys()):
							row[l]=''
						self._output.loc[i+1]=row
	
	def save(self):
		
		day=strftime("%m-%d")
		hour=strftime('%H:%M')
		titlecsv='outputcsv-'+hour
		titlepkl='outputpkl-'+hour
		directory='/Users/remubo/Documents/obs_error_invest/benchmarks/'+day

		if not os.path.exists(directory):
			os.makedirs(directory)

		#Get the arrays column names:
		arrs=[c for c in self.output.columns if type(self.output[c].iloc[0])==numpy.ndarray]
		#Get clean data
		clean=[c for c in self.output.columns if c not in arrs]

		#Build the arrays dictionary:
		dico={name+'-'+str(i):self.output[name].iloc[i] for (name,i) in product(arrs,range(len(self.output)))}

		#Get the rest of the output:
		outcut=self.output[clean]

		#Save the arrays containers
		with open(directory+'/'+titlepkl,'wb') as f:
			pickle.dump(dico,f,pickle.HIGHEST_PROTOCOL)

		#save the csv
		outcut.to_csv(directory+'/'+titlecsv)

		plot_benchmark_analysis(self,skip=True)
		plt.savefig(directory+'/plot-'+hour+'.pdf',bbox_inches='tight')
		plt.close()
		plt.close()
		print('\noutput saved')

	@staticmethod
	def load(day,hour,extra=''):
		#build the paths:
		directory='/Users/remubo/Documents/obs_error_invest/benchmarks/'+day
		titlecsv='outputcsv-'+hour+extra
		titlepkl='outputpkl-'+hour+extra

		pathpkl=directory+'/'+titlepkl
		pathcsv=directory+'/'+titlecsv

		#Load the csv-saved dataframe:
		df=read_csv(pathcsv,keep_default_na=False)

		#Read the pkled dictionary:
		with open(pathpkl,'rb') as infile:
			dico=pickle.load(infile)

			#Complete the df
			for c in [k.split('-')[0] for k in dico.keys()]:
				df[c]=''
			#Fill in
			for a in dico.items():
				df[a[0].split('-')[0]].iloc[int(a[0].split('-')[1])]=a[1]

		return df

	def pause_run(self):
		key = poll_input()

		if key == '\n' and self._is_paused==False: 
			print('paused')
			self._is_paused = True # pause run
			return None #Avoid the next (unpausing step)
		if self._is_paused:
			# If paused
			ch = getch()
			if ch==' ':
				print('unpaused')
				#remove upper line of tqdm
				CURSOR_UP_ONE = '\x1b[1A'
				ERASE_LINE = '\x1b[2K'
				print(CURSOR_UP_ONE + ERASE_LINE)
				self._is_paused = False

		

	def __repr__(self):
		s=''
		cols=['Setup','Config','Rt','Bench']
		vs=[self.setup.name,self.config,self.Rt,self._bench]
		for (i,j) in zip(cols,vs):
			s+='.'+i+'\n'
			s+='.\t.'+j.__repr__().replace('\n','\n.\t.')+'\n'
		return s

	def __iadd__(self,val):
		self._bench.__iadd__(val)
		return self


	@property
	def bench(self):
		return self._bench

	@property
	def output(self):
		return self._output

