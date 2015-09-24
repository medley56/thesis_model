"""
Analysis of simulation data generated by trajectory.py
Written by Gavin Medley
University of Colorado
"""

# Import simanalysis module for analysis functions
import simanalysis as sa

# Import necessary module dependencies
import numpy as np
import os
import json
from natsort import index_natsorted, order_by_index

"""
Function definitions below
"""

#def Smoothe(dat, nClose=2):
#	"""
#	Smoothe() smoothes the data using the nClose closest points on either side
#	"""
#	newDat = np.empty(dat.size)
#	for pt in range(dat.size):
#		if pt<nClose:
#			newDat[pt] = np.mean(dat[0:pt+nClose+1])
#		elif pt>=(len(dat)-nClose):
#			newDat[pt] = np.mean(dat[pt-nClose:-1])
#		else:
#			newDat[pt] = np.mean(dat[pt-nClose:pt+nClose+1])
#	return(newDat)
#
#
#def GVis(GRuns):
#	"""
#	GVis() plots out the ellipses in phenotypic space, along with the
#	angle of g_max, size of G, and epsilon (inverse eccentricity) over
#	generations.
#	"""
#
#	# Declare arrays to hold the info that we plot with
#	angles = np.empty(shape=[GRuns.shape[0],GRuns.shape[1]])
#	Sigmas = np.empty(shape=[GRuns.shape[0],GRuns.shape[1]])
#	epsilons = np.empty(shape=[GRuns.shape[0],GRuns.shape[1]])
#
#	# Create a nice set of plot figures
#	ellipseTrajectories = pl.figure('G Trajectories')
#	anglesPlot = pl.figure('Angle of g_max')
#	sizePlot = pl.figure('Size of G')
#	epsPlot = pl.figure('epsilon by Generation')
#
#	# Loop through different runs
#	for run in GRuns.shape[0]:
#		print(run)
#		# Set up axes for the gmax angles plot
#		angleAxes = anglesPlot.add_subplot(3,2,run+1,
#			xlim=[0,GRuns.shape[1]], ylim=[-360,360],
#			title='Run'+str(run))
#
#		# Set up axes for the Sigma (size) plot
#		sizeAxes = sizePlot.add_subplot(3,2,run+1,
#			xlim=[0,GRuns.shape[1]], ylim=[0,15],
#			title='Run'+str(run))
#
#		# Set up axes for the epsilons (inversely related to eccentricity) plot
#		epsAxes = epsPlot.add_subplot(3,2,run+1,
#			xlim=[0,GRuns.shape[1]], ylim=[0,10],
#			title='Run'+str(run))
#
#		# Set up axes for the ellipse trajectories plot
#		ellAxes = ellipseTrajectories.add_subplot(3,2,run+1,
#			xlim=[-40,55], ylim=[-40,55],
#			title='Run'+str(run))
#
#		# Plot the mean phenotype start and end points of the run
#		phenoStart = phenoZRuns[run,0,:]
#		ellAxes.plot(phenoStart[0],phenoStart[1],"o",c="blue")
#		phenoEnd = phenoZRuns[run,-1,:]
#		ellAxes.plot(phenoEnd[0],phenoEnd[1],"s",c="yellow")
#
#		# Loop through generations within the current run
#		for gen in range(0,GRuns.shape[1],1):
#			# Get eigenvalues and eigenvectors of the current G matrix
#			eigs = np.linalg.eigh(GRuns[run,gen,:,:])
#			# Which is the dominant eigenvalue/vector
#			which = np.argmax(eigs[0])
#			# Define the angle as
#			angles[run,gen] = math.atan2(eigs[1][1,which],eigs[1][0,which])*360.0/(2*math.pi)
#			Sigmas[run,gen] = GRuns[run,gen,0,0] + GRuns[run,gen,1,1]
#
#			if np.max(eigs[0])==0:
#				epsilons[run,gen] = 0
#				print('HEY, we got a nilpotent G matrix here (0 eigenvalues).')
#				print('generation',gen, 'in run', run)
#			else:
#				epsilons[run,gen] = np.min(eigs[0]) / np.max(eigs[0])
#
#			# This bit of code actually makes the ellipse object that is added to the plot
#			thisEllipse = Ellipse(xy=phenoZRuns[run,gen,:],
#				width=Sigmas[run,gen],
#				height=Sigmas[run,gen]*epsilons[run,gen],
#				angle=angles[run,gen],
#				alpha=.5, facecolor=[1,1,1])
#
#			# Add the ellipse object to the plot
#			ellAxes.add_artist(thisEllipse)
#			#thisEllipse.set_clip_box(ellAxes.bbox)
#
#
#		# After each set of generations, plot the angles
#		angleAxes.plot(range(0,GRuns.shape[1]), angles[run,:], '-o', markersize=3)
#
#		# After each set of generations, plot the Sigmas (total 'size' of the G matrix)
#		sizeAxes.plot(range(0,GRuns.shape[1]), Sigmas[run,:], '-o', markersize=3)
#
#		# After each set of generations, plot the epsilons (inversely related to eccentricity)
#		epsAxes.plot(range(0,GRuns.shape[1]), Sigmas[run,:], '-o', markersize=3)
#
#	# Format the plots with .tight_layout() so they are pretty
#	ellipseTrajectories.tight_layout()
#	anglesPlot.tight_layout()
#	epsPlot.tight_layout()
#	sizePlot.tight_layout()
#
#	# Show the plots
#	pl.show()
#	return
#
#
#def MVis(rMuRuns, paramRuns):
#	"""
#	Plotting the mutational correlation over time along with the angle of the
#	dominant eigenvector of the M matrix
#	"""
#
#	# Make empty array to hold the angles over time
#	angles = np.empty(shape=[GRuns.shape[0],GRuns.shape[1]])
#
#	# Set up the figure to plot in
#	anglesPlot = pl.figure('m_max Angles')
#	corrPlot = pl.figure('Mutational Correlation r_mu')
#
#	# Loop through different runs
#	for run in range(rMuRuns.shape[0]):
#		print(run)
#		# Set up axes for the m_max angles plot
#		angleAxes = anglesPlot.add_subplot(3,2,run+1,
#			xlim=[0,rMuRuns.shape[1]], ylim=[-360,360],
#			title='Run'+str(run))
#
#		corrAxes = corrPlot.add_subplot(3,2,run+1,
#			xlim=[0,rMuRuns.shape[1]], ylim=[-1,1],
#			title='Run'+str(run))
#
#		# Loop through generations within the current run
#		for gen in range(0,rMuRuns.shape[1],1):
#			M = np.mat([[paramRuns[run]['alphaZ1'], rMuRuns[run,gen]],
#				[rMuRuns[run,gen], paramRuns[run]['alphaZ2']]])
#
#			# Get eigenvalues and eigenvectors of the current M matrix
#			eigs = np.linalg.eigh(M)
#			# Which is the dominant eigenvalue/vector
#			which = np.argmax(eigs[0])
#			# Define the angle as
#			angles[run,gen] = math.atan2(eigs[1][1,which],eigs[1][0,which])*360.0/(2*math.pi)
#
#			if np.max(eigs[0])==0:
#				epsilons[run,gen] = 0
#				print('HEY, we got a nilpotent M matrix here (0 eigenvalues).')
#				print('generation',gen, 'in run', run)
#
#
#		# After each set of generations, plot the angles
#		angleAxes.plot(range(0,rMuRuns.shape[1]), angles[run,:], '-o', markersize=3)
#
#		# Also plot the mutational correlations over generations
#		corrAxes.plot(range(0,rMuRuns.shape[1]), rMuRuns[run,:], '-o', markersize=3)
#
#	# Make the plots pretty using .tight_layout()
#	anglesPlot.tight_layout()
#	corrPlot.tight_layout()
#
#	# Show plots
#	pl.show()
#	return
#
#
#def FitnessVis():
#	foo = 'bar'
#
#
#def PhenotypeVis():
#	foo = 'bar'


"""
Import data from the path below
"""
os.chdir('/home/gmedley/Documents/Thesis/TrajectoryModel/TestRunsToAnalyze/')

maxGen = 500 # This has to be constant throughout all runs

# Make empty arrays to store the data
MRuns = np.empty(shape=[0,maxGen,2,2])
GRuns = np.empty(shape=[0,maxGen,2,2])
rMuRuns = np.empty(shape=[0,maxGen])
phenoZRuns = np.empty(shape=[0,maxGen,2])
paramRuns = []
dirList = os.listdir()

"""
Run through the filesystem and fill all the data arrays.
They come out in a weird order and dirnames only comes out once.
Note that '.' implies we need to be IN the directory where the runX
files have been written.
"""
for dirname, dirnames, filenames in os.walk('.'):
	# Capture a list of all directory names to sort later
	if dirname!='.':
		# read in the G matrix history file and append it to GRuns
		newG = np.load(dirname+'/gmatrix_history.npy')[np.newaxis,:,:,:]
		GRuns = np.append(GRuns, newG, axis=0)
		print('Successfully appended ',dirname,' G history')

		newrMu = np.load(dirname+'/mean_rmu_history.npy')[np.newaxis,:]
		rMuRuns = np.append(rMuRuns, newrMu, axis=0)
		print('Successfully appended ',dirname,' mean mutational correlation history')

		newPhenoZ = np.load(dirname+'/z_trait_history.npy')[np.newaxis,:,:]
		phenoZRuns = np.append(phenoZRuns, newPhenoZ, axis=0)
		print('Succesfully appended ',dirname,'mean trait values history')

		# Load the json file that contains the parameter dictionary for each run
		newParams = json.load(open(dirname+'/params.txt','r'))
		paramRuns.append(newParams)
		print('Successfully loaded the parameters for ',dirname,'\n')


# Some work to sort everything in a logical way
dirIndex = index_natsorted(dirList) # store the indices order
dirList = order_by_index(dirList, dirIndex) # Since dirList is a list, use
# Sort all the arrays so that data are ordered run0, run1, run2, ...
GRuns = GRuns[dirIndex,:,:,:]
rMuRuns = rMuRuns[dirIndex,:]
phenoZRuns = phenoZRuns[dirIndex,:,:]
paramRuns = order_by_index(paramRuns, dirIndex)


sa.GVis(GRuns[0:4,:,:,:], phenoZRuns)

#sa.MVis(rMuRuns, paramRuns, GRuns)

#MVis(rMuRuns[0:6,:], paramRuns)