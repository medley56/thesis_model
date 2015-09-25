"""
Module for analysis of simulated data
Written by Gavin Medley
University of Colorado
"""


# Import the necessary modules
import pylab as pl
from matplotlib.patches import Ellipse
import numpy as np
import os
import json
from natsort import index_natsorted, order_by_index

"""
Function definitions below
Smoothe()
GVis()
MVis()
"""
#%%
def ImportData():
	"""
	Import data from the path below
	"""
	os.chdir('/home/gmedley/Documents/Thesis/TrajectoryModel/RunsToAnalyze/')

	# Load the json file that contains the parameter dictionary for all runs
	parameters = json.load(open('params.txt','r'))
	print('Successfully loaded the parameters governing all runs')
	maxGen = parameters['maxGen'] # This has to be constant throughout all runs

	# Make empty arrays to store the data
	MRuns = np.empty(shape=[0,maxGen,2,2])
	GRuns = np.empty(shape=[0,maxGen,2,2])
	rMuRuns = np.empty(shape=[0,maxGen])
	phenoZRuns = np.empty(shape=[0,maxGen,2])
	dirList = list()
	for d in filter(os.path.isdir, os.listdir()): dirList.append(d)

	"""
	Run through the filesystem and fill all the data arrays.
	They come out in a weird order and dirnames only comes out once.
	Note that '.' implies we need to be IN the directory where the runX
	files have been written.
	"""
	for dirname, dirnames, filenames in os.walk('.'):
		# Capture a list of all directory names to sort later
		if dirname!='.' and dirname!='params.txt':
			# read in the G matrix history file and append it to GRuns
			newG = np.load(dirname+'/gmatrix_history.npy')[np.newaxis,:,:,:]
			GRuns = np.append(GRuns, newG, axis=0)
			print('Successfully appended ',dirname,' G history')

			newM = np.load(dirname+'/mmatrix_history.npy')[np.newaxis,:,:,:]
			MRuns = np.append(MRuns, newM, axis=0)
			print('Successfully appended ',dirname,' M history')

			newrMu = np.load(dirname+'/mean_rmu_history.npy')[np.newaxis,:]
			rMuRuns = np.append(rMuRuns, newrMu, axis=0)
			print('Successfully appended ',dirname,' mean mutational correlation history')

			newPhenoZ = np.load(dirname+'/z_trait_history.npy')[np.newaxis,:,:]
			phenoZRuns = np.append(phenoZRuns, newPhenoZ, axis=0)
			print('Succesfully appended ',dirname,'mean trait values history')

	# Some work to sort everything in a logical way
	dirIndex = index_natsorted(dirList) # store the indices order
	dirList = order_by_index(dirList, dirIndex) # Since dirList is a list, use
	# Sort all the arrays so that data are ordered run0, run1, run2, ...
	GRuns = GRuns[dirIndex,:,:,:]
	rMuRuns = rMuRuns[dirIndex,:]
	phenoZRuns = phenoZRuns[dirIndex,:,:]


	print('Parameters used:\n',parameters)

	return({
		'GRuns':GRuns,
		'rMuRuns':rMuRuns,
		'phenoZRuns':phenoZRuns,
		'MRuns':MRuns,
		'parameters':parameters
		})

#%%
def Smoothe(dat, nPts):
	"""
	Smoothe() smoothes the data using the nPts closest points on either side
	"""
	newDat = np.empty(dat.size)
	for pt in range(dat.size):
		if pt<nPts:
			newDat[pt] = np.average(dat[0:pt+nPts+1])
		elif pt>=(len(dat)-nPts):
			newDat[pt] = np.average(dat[pt-nPts:-1])
		else:
			newDat[pt] = np.average(dat[pt-nPts:pt+nPts+1] )
	return(newDat)

#%%
def GVis(timeseries, smoothing, nPts):
	"""
	GVis() plots out the ellipses in phenotypic space, along with the
	angle of g_max, size of G, and epsilon (inverse eccentricity) over
	generations.
	If smoothing==True, we smoothe the angles, Sigmas, and epsilons
	using nPts on either side
	"""
	GRuns = timeseries['GRuns']
	phenoZRuns = timeseries['phenoZRuns']
	parameters = timeseries['parameters']
	nRuns = GRuns.shape[0]
	nGen = GRuns.shape[1]

	# Declare arrays to hold the info that we plot with
	Gangles = np.empty(shape=[nRuns,nGen])
	GSigmas = np.empty(shape=[nRuns,nGen])
	Gepsilons = np.empty(shape=[nRuns,nGen])


	# Loop through different runs
	for run in range(nRuns):
		print('Run',run)

		# Declare a list to hold the ellipse objects for plotting
		runEllipses = list()

		# Loop through generations within the current run
		for gen in range(nGen):
			# Get eigenvalues and eigenvectors of the current G matrix
			eigs = np.linalg.eigh(GRuns[run,gen,:,:])
			# Which is the dominant eigenvalue/vector
			which = np.argmax(eigs[0])
			# Define the angle as
			Gangles[run,gen] = np.arctan2(eigs[1][1,which],eigs[1][0,which])

			# If Gangles change sign drastically, it probably means they just moved a little but
			# across the -pi to +pi line. Fix that below. Note that if
			# angle changes by about pi/2 it probably means we just changed which
			# eigenvector was dominant.
			if gen!=0 and Gangles[run,gen]-Gangles[run,gen-1] > np.pi:
				print('Difference on gen=',gen,' : ',Gangles[run,gen]-Gangles[run,gen-1])
				print('Angle at gen=',Gangles[run,gen])
				Gangles[run,gen] = Gangles[run,gen] - 2*np.pi
				print('New difference=',Gangles[run,gen]-Gangles[run,gen-1])
				print('New angle',Gangles[run,gen])
			elif gen!=0 and Gangles[run,gen]-Gangles[run,gen-1] < -np.pi:
				print('Difference on gen=',gen,' : ',Gangles[run,gen]-Gangles[run,gen-1])
				print('Angle at gen=',Gangles[run,gen])
				Gangles[run,gen] = Gangles[run,gen] + 2*np.pi
				print('New difference=',Gangles[run,gen]-Gangles[run,gen-1])
				print('New angle',Gangles[run,gen])

			# The magnitude of G as the sum of the diagonals (variances)
			GSigmas[run,gen] = GRuns[run,gen,0,0] + GRuns[run,gen,1,1]

			if np.max(eigs[0])==0:
				Gepsilons[run,gen] = 0
				print('HEY, we got a nilpotent G matrix here (0 eigenvalues).')
				print('generation',gen, 'in run', run)
			else:
				Gepsilons[run,gen] = np.min(eigs[0]) / np.max(eigs[0])

			# This bit of code actually makes the ellipse object that is added to the plot
			runEllipses.append(Ellipse(xy=phenoZRuns[run,gen,:],
				width=GSigmas[run,gen],
				height=GSigmas[run,gen]*Gepsilons[run,gen],
				angle=Gangles[run,gen]*360/(2*np.pi),
				alpha=.3, facecolor=[80/256,200/256,250/256]))

			# We don't want to see every single ellipse. It's too much.
			# So set most of them to be invisible
			if np.mod(gen,10)!=0:
				runEllipses[gen].set_alpha(0)
			# Make the last ellipse colorful
			if gen==nGen-1:
				runEllipses[gen].set_facecolor([140/256,256/256,200/256])
				runEllipses[gen].set_alpha(.5)

		# Create a nice set of plot figures
		ellGTrajectories = pl.figure('G Trajectory Centered at Mean Trait Values')
		GanglesPlot = pl.figure('Angle of Primary Eigenvector of G (g_max)')
		GSigmaPlot = pl.figure('Size of G (Sigma)')
		GepsPlot = pl.figure('Eccentricity of G (epsilon)')

		# Set up axes for the gmax Gangles plot
		GangleAxes = GanglesPlot.add_subplot(1,1,1,#3,2,run+1,
			xlim=[0,nGen], ylim=[np.min(Gangles[run,:])-.5,np.max(Gangles[run,:])+.5],
			title='Angle of Primary Eigenvector of G (Run'+str(run)+')')

		# Set up axes for the Sigma (size) plot
		GSigmaAxes = GSigmaPlot.add_subplot(1,1,1,
			xlim=[0,nGen], ylim=[0,np.max(GSigmas[run,:])],
			title='Magnitude of G (Sigma) (Run'+str(run)+')')

		# Set up axes for the Gepsilons (inversely related to eccentricity) plot
		GepsAxes = GepsPlot.add_subplot(1,1,1,
			xlim=[0,nGen], ylim=[0,1],
			title='Eccentricity of G (epsilon) (Run'+str(run)+')')

		# Set up axes for the ellipse trajectories plot
		ellGAxes = ellGTrajectories.add_subplot(1,1,1,
			xlim=[-40,55], ylim=[-40,55],
			title='G Trajectory Centered at Mean Trait Values (Run'+str(run)+')')

		# Plot the mean phenotype trajectory parametrically through time
		ellGAxes.plot(phenoZRuns[run,:,0],phenoZRuns[run,:,1], "-", c="black")
		phenoStart = phenoZRuns[run,0,:]
		ellGAxes.plot(phenoStart[0],phenoStart[1],"o",c="blue", markersize=8)
		phenoEnd = phenoZRuns[run,-1,:]
		ellGAxes.plot(phenoEnd[0],phenoEnd[1],"s",c="yellow", markersize=8)
		optimum = parameters['theta']
		ellGAxes.plot(optimum[0],optimum[1],"x",c="red", mew=3, markersize=8)

		# Add the ellipse objects to the plot
		for gen in range(nGen):
			ellGAxes.add_artist(runEllipses[gen])

		if smoothing==True:
			print('Smoothing is turned on.')
			Gangles[run,:] = Smoothe(Gangles[run,:], nPts)
			GSigmas[run,:] = Smoothe(GSigmas[run,:], nPts)
			Gepsilons[run,:] = Smoothe(Gepsilons[run,:], nPts)

		# After each set of generations, plot the Gangles
		GangleAxes.plot(range(0,nGen), Gangles[run,:], '-o', markersize=3)

		# After each set of generations, plot the GSigmas (total 'size' of the G matrix)
		GSigmaAxes.plot(range(0,nGen), GSigmas[run,:], '-o', markersize=3)

		# After each set of generations, plot the Gepsilons (inversely related to eccentricity)
		GepsAxes.plot(range(0,nGen), Gepsilons[run,:], '-o', markersize=3)

		# Format the plots with .tight_layout() so they are pretty
		ellGTrajectories.tight_layout()
		GanglesPlot.tight_layout()
		GepsPlot.tight_layout()
		GSigmaPlot.tight_layout()

		pl.show()

		input("Press Enter to continue...")

	return

#%%
def MVis(timeseries, smoothing, nPts):
	"""
	Plotting the mutational correlation over time along with the angle of the
	dominant eigenvector of the M matrix
	"""

	rMuRuns = timeseries['rMuRuns']
	MRuns = timeseries['MRuns']
	nRuns = MRuns.shape[0]
	nGen = MRuns.shape[1]

	# Make empty array to hold the angles over time
	Mangles = np.empty(shape=[nRuns,nGen])
	MSigmas = np.empty(shape=[nRuns,nGen])
	Mepsilons = np.empty(shape=[nRuns,nGen])


	# Loop through different runs
	for run in range(nRuns):
		print('Run',run)

		# Loop through generations within the current run
		for gen in range(0,nGen,1):
			# Get eigenvalues and eigenvectors of the current M matrix
			eigs = np.linalg.eigh(MRuns[run,gen,:,:])
			# Which is the dominant eigenvalue/vector
			which = np.argmax(eigs[0])
			# Define the angle as
			Mangles[run,gen] = np.arctan2(eigs[1][1,which],eigs[1][0,which])

			# If angles change sign drastically, it probably means they just moved a little but
			# across the -pi to +pi line. Fix that below. Note that if
			# angle changes by about pi/2 it probably means we just changed which
			# eigenvector was dominant.
			if Mangles[run,gen]-Mangles[run,gen-1]>np.pi:
				Mangles[run,gen] = Mangles[run,gen] - 2*np.pi
			elif Mangles[run,gen]-Mangles[run,gen-1]<-np.pi:
				Mangles[run,gen] = Mangles[run,gen] + 2*np.pi

			if np.max(eigs[0])==0:
				print('HEY, we got a nilpotent M matrix here (0 eigenvalues).')
				print('generation',gen, 'in run', run)

			# The magnitude of G as the sum of the diagonals (variances)
			MSigmas[run,gen] = MRuns[run,gen,0,0] + MRuns[run,gen,1,1]

			# Calculate the eccentricity of the M matrix.
			if np.max(eigs[0])==0:
				Mepsilons[run,gen] = 0
				print('HEY, we got a nilpotent G matrix here (0 eigenvalues).')
				print('generation',gen, 'in run', run)
			else:
				Mepsilons[run,gen] = np.min(eigs[0]) / np.max(eigs[0])

		# Set up the figures to plot in
		ManglesPlot = pl.figure('Angle of Primary Eigenvector of M (m_max)')
		rMuPlot = pl.figure('Mutational Correlation (r_mu)')
		MSigmaPlot = pl.figure('Magnitude of M Matrix (Sigma)')

		# Set up axes for the m_max angles plot
		MangleAxes = ManglesPlot.add_subplot(1,1,1,
			xlim=[0,rMuRuns.shape[1]], ylim=[np.min(Mangles[run])-.5,np.max(Mangles[run])+.5],
			title='Angle of Primary Eigenvector of M (Run'+str(run)+')')

		rMuAxes = rMuPlot.add_subplot(1,1,1,
			xlim=[0,rMuRuns.shape[1]], ylim=[-1,1],
			title='Mutational Correlation (Run'+str(run)+')')

		MSigmaAxes = MSigmaPlot.add_subplot(1,1,1,
			xlim=[0,rMuRuns.shape[1]], ylim=[np.min(MSigmas[run,:]),np.max(MSigmas[run,:])],
			title='Magnitude of M Matrix (Run'+str(run)+')')

		# After each set of generations, plot the angles
		if smoothing==True:
			print('Smoothing is turned on.')
			Mangles[run,:] = Smoothe(Mangles[run,:], nPts)
			rMuRuns[run,:] = Smoothe(rMuRuns[run,:], nPts)

		# Plot the angle of M over generations
		MangleAxes.plot(range(0,rMuRuns.shape[1]), Mangles[run,:], '-o', markersize=3)

		# Also plot the mutational correlations over generations
		rMuAxes.plot(range(0,rMuRuns.shape[1]), rMuRuns[run,:], '-o', markersize=3)

		# Plot the magnitude of M (MSigmas) over generations
		MSigmaAxes.plot(range(0,rMuRuns.shape[1]), MSigmas[run,:], '-o', markersize=3)

		# Make the plots pretty using .tight_layout()
		ManglesPlot.tight_layout()
		rMuPlot.tight_layout()

		# Show plots
		pl.show()

		input("Press Enter to continue...")

	return


def FitnessVis():
	foo = 'bar'
	return(foo)


def PhenotypeVis():
	foo = 'bar'
	return(foo)

