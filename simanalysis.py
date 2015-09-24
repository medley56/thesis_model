"""
Module for analysis of simulated data
Written by Gavin Medley
University of Colorado
"""


# Import the necessary modules
import pylab as pl
from matplotlib.patches import Ellipse
import numpy as np
import math

"""
Function definitions below
Smoothe()
GVis()
MVis()
"""

def Smoothe(dat, nClose=2):
	"""
	Smoothe() smoothes the data using the nClose closest points on either side
	"""
	newDat = np.empty(dat.size)
	for pt in range(dat.size):
		if pt<nClose:
			newDat[pt] = np.mean(dat[0:pt+nClose+1])
		elif pt>=(len(dat)-nClose):
			newDat[pt] = np.mean(dat[pt-nClose:-1])
		else:
			newDat[pt] = np.mean(dat[pt-nClose:pt+nClose+1])
	return(newDat)


def GVis(GRuns, phenoZRuns):
	"""
	GVis() plots out the ellipses in phenotypic space, along with the
	angle of g_max, size of G, and epsilon (inverse eccentricity) over
	generations.
	"""

	# Declare arrays to hold the info that we plot with
	angles = np.empty(shape=[GRuns.shape[0],GRuns.shape[1]])
	Sigmas = np.empty(shape=[GRuns.shape[0],GRuns.shape[1]])
	epsilons = np.empty(shape=[GRuns.shape[0],GRuns.shape[1]])


	# Loop through different runs
	for run in range(GRuns.shape[0]):
		print(run)

		# Declare a list to hold the ellipse objects for plotting
		runEllipses = list()


		# Loop through generations within the current run
		for gen in range(0,GRuns.shape[1],1):
			# Get eigenvalues and eigenvectors of the current G matrix
			eigs = np.linalg.eigh(GRuns[run,gen,:,:])
			# Which is the dominant eigenvalue/vector
			which = np.argmax(eigs[0])
			# Define the angle as
			angles[run,gen] = np.arctan2(eigs[1][1,which],eigs[1][0,which])#*360.0/(2*np.pi)
			# If angles change sign drastically, it probably means they just moved a little but
			# across the -pi to +pi line. Fix that below. Note that if
			# angle changes by about pi/2 it probably means we just changed which
			# eigenvector was dominant.
			if angles[run,gen]-angles[run,gen-1]>np.pi:
				angles[run,gen] = angles[run,gen] - 2*np.pi
			elif angles[run,gen]-angles[run,gen-1]<-np.pi:
				angles[run,gen] = angles[run,gen] + 2*np.pi

			Sigmas[run,gen] = GRuns[run,gen,0,0] + GRuns[run,gen,1,1]

			if np.max(eigs[0])==0:
				epsilons[run,gen] = 0
				print('HEY, we got a nilpotent G matrix here (0 eigenvalues).')
				print('generation',gen, 'in run', run)
			else:
				epsilons[run,gen] = np.min(eigs[0]) / np.max(eigs[0])

			# This bit of code actually makes the ellipse object that is added to the plot
			runEllipses.append(Ellipse(xy=phenoZRuns[run,gen,:],
				width=Sigmas[run,gen],
				height=Sigmas[run,gen]*epsilons[run,gen],
				angle=angles[run,gen],
				alpha=.5, facecolor=[1,1,1]))


		# Create a nice set of plot figures
		ellipseTrajectories = pl.figure('G Trajectories')
		anglesPlot = pl.figure('Angle of g_max')
		sizePlot = pl.figure('Size of G')
		epsPlot = pl.figure('epsilon by Generation')

		# Set up axes for the gmax angles plot
		angleAxes = anglesPlot.add_subplot(1,1,1,#3,2,run+1,
			xlim=[0,GRuns.shape[1]], ylim=[np.min(angles[run,:])-.5,np.max(angles[run,:])+.5],
			title='Run'+str(run))

		# Set up axes for the Sigma (size) plot
		sizeAxes = sizePlot.add_subplot(1,1,1,
			xlim=[0,GRuns.shape[1]], ylim=[0,np.max(Sigmas[run,:])],
			title='Run'+str(run))

		# Set up axes for the epsilons (inversely related to eccentricity) plot
		epsAxes = epsPlot.add_subplot(1,1,1,
			xlim=[0,GRuns.shape[1]], ylim=[0,1],
			title='Run'+str(run))

		# Set up axes for the ellipse trajectories plot
		ellAxes = ellipseTrajectories.add_subplot(1,1,1,
			xlim=[-40,55], ylim=[-40,55],
			title='Run'+str(run))

		# Plot the mean phenotype start and end points of the run
		phenoStart = phenoZRuns[run,0,:]
		ellAxes.plot(phenoStart[0],phenoStart[1],"o",c="blue")
		phenoEnd = phenoZRuns[run,-1,:]
		ellAxes.plot(phenoEnd[0],phenoEnd[1],"s",c="yellow")


		# Add the ellipse objects to the plot
		for gen in range(GRuns.shape[1]):
			ellAxes.add_artist(runEllipses[gen])
		#thisEllipse.set_clip_box(ellAxes.bbox)

		# After each set of generations, plot the angles
		angleAxes.plot(range(0,GRuns.shape[1]), angles[run,:], '-o', markersize=3)

		# After each set of generations, plot the Sigmas (total 'size' of the G matrix)
		sizeAxes.plot(range(0,GRuns.shape[1]), Sigmas[run,:], '-o', markersize=3)

		# After each set of generations, plot the epsilons (inversely related to eccentricity)
		epsAxes.plot(range(0,GRuns.shape[1]), epsilons[run,:], '-o', markersize=3)

		# Format the plots with .tight_layout() so they are pretty
		ellipseTrajectories.tight_layout()
		anglesPlot.tight_layout()
		epsPlot.tight_layout()
		sizePlot.tight_layout()

		pl.show()

		input("Press Enter to continue...")

	# Show the plots
	#pl.show()
	return


def MVis(rMuRuns, paramRuns, GRuns):
	"""
	Plotting the mutational correlation over time along with the angle of the
	dominant eigenvector of the M matrix
	"""

	# Make empty array to hold the angles over time
	angles = np.empty(shape=[GRuns.shape[0],GRuns.shape[1]])

	# Set up the figure to plot in
	anglesPlot = pl.figure('m_max Angles')
	corrPlot = pl.figure('Mutational Correlation r_mu')

	# Loop through different runs
	for run in range(rMuRuns.shape[0]):
		print(run)
		# Set up axes for the m_max angles plot
		angleAxes = anglesPlot.add_subplot(3,2,run+1,
			xlim=[0,rMuRuns.shape[1]], ylim=[-360,360],
			title='Run'+str(run))

		corrAxes = corrPlot.add_subplot(3,2,run+1,
			xlim=[0,rMuRuns.shape[1]], ylim=[-1,1],
			title='Run'+str(run))

		# Loop through generations within the current run
		for gen in range(0,rMuRuns.shape[1],1):
			M = np.mat([[paramRuns[run]['alphaZ1'], rMuRuns[run,gen]],
				[rMuRuns[run,gen], paramRuns[run]['alphaZ2']]])

			# Get eigenvalues and eigenvectors of the current M matrix
			eigs = np.linalg.eigh(M)
			# Which is the dominant eigenvalue/vector
			which = np.argmax(eigs[0])
			# Define the angle as
			angles[run,gen] = math.atan2(eigs[1][1,which],eigs[1][0,which])*360.0/(2*math.pi)

			if np.max(eigs[0])==0:
				#epsilons[run,gen] = 0
				print('HEY, we got a nilpotent M matrix here (0 eigenvalues).')
				print('generation',gen, 'in run', run)


		# After each set of generations, plot the angles
		angleAxes.plot(range(0,rMuRuns.shape[1]), angles[run,:], '-o', markersize=3)

		# Also plot the mutational correlations over generations
		corrAxes.plot(range(0,rMuRuns.shape[1]), rMuRuns[run,:], '-o', markersize=3)

	# Make the plots pretty using .tight_layout()
	anglesPlot.tight_layout()
	corrPlot.tight_layout()

	# Show plots
	pl.show()
	return


def FitnessVis():
	foo = 'bar'


def PhenotypeVis():
	foo = 'bar'

