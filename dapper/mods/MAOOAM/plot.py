#plot and calculate statistics

import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from pylab import *
from mods.MAOOAM.params2 import ndim,tw,t_run,t_trans
def plotC():
	"""" 3-D plot of the attractor.
	Axis X : thetao2 : 2nd coordinates of the ocean temperature
	Axis Y : psio2   : 2nd coordinates of the ocean streamfunction
	Axis Z  :psia1   : 1st coordinates of the atmosphere streamfunction """

	data=np.loadtxt("evol_field_python_"+str(t_trans)+" "+str(t_run)+".dat")
	t=data[:,0]
	thetao2=data[:,30]
	psio2=data[:,22]
	psia1=data[:,1]
	

# 	plot(t,psia1)
# 	show()
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.plot(xs=psio2, ys=thetao2, zs=psia1, zdir='z', label='zs=0, zdir=z')

	plt.show()
	
def plotD():
	""" 2-D plot of the ndim coordinates
	Axis X : 36 coordinates
	Axis Y : their values
	The graphics are superimposed"""
	data=np.loadtxt("evol_field_python_"+str(t_trans)+" "+str(t_run)+".dat")
	X=np.zeros(ndim)
	nb=(t_run/tw)-1
	print (int(nb))
	for j in range(1,int(nb)) :
		for i in range(1,36) :
			X[i-1]=data[j,i]
		plot(X)
	show()

def plotDanim():
	""" 2-D plot of the ndim coordinates
	Axis X : 36 coordinates
	Axis Y : their values
	The graphics are superimposed"""
	data=np.loadtxt("evol_field_python_"+str(t_trans)+" "+str(t_run)+".dat")
	X=np.zeros(ndim)
	nb=(t_run/tw)-1
	
	ion()
	#first curve
	for i in range(1,36) :
		X[i-1]=data[1,i]
	line,=plot(X)
	nb=(t_run/tw)-1
	#animation
	for j in range(2,int(nb)) :
		for i in range(1,36) :
			X[i-1]=data[j,i]
			line.set_ydata(X)
			draw()
	ioff()
	show()


 
def calculate_stat():
	data=np.loadtxt("evol_field_python_"+str(t_trans)+" "+str(t_run)+".dat")
	mean=np.zeros(ndim)
	var=np.zeros(ndim)
	for i in range(1,ndim+1) :
		mean[i-1]=np.mean(data[:,i])
		var[i-1]=np.var(data[:,i])
	fichier = open("mean_field_python_"+str(t_trans)+" "+str(t_run)+".dat", "w")
	
	for i in range(0,ndim) :
		fichier.write(str.format("{0:.12f}", mean[i])+" ")
	fichier.write("\n")
	
	for i in range(0,ndim) :
		fichier.write(str.format("{0:.12f}", var[i])+" ")
	fichier.write("\n")
	fichier.close()

def plot_stat():
	data=np.loadtxt("mean_field_python_"+str(t_trans)+" "+str(t_run)+".dat")
	plot(data[0,:])
	show()


if __name__ == "__main__":
	plotC()
