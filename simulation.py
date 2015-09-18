"""
Module containing functions used in the course of generating simulation data
Written by Gavin Medley
University of Colorado
"""

# Import necessary module dependencies
import numpy as np

# Declare a few variables to be global
global nZloci, nRloci
nZloci = 10
nRloci = 5

################################
# Defining functions here
################################

# The call to make multivariate normal draws sucks so:
mvn = np.random.multivariate_normal

# Default values are meant for testing the model
def MainModel(maxGen=500, K=100, theta=[1., -1.], omega=[[50., 0.],[0., 50.]],
	mu=0.0002, alphaZ1=0.01, alphaZ2=0.01, alphaR=0.01):
	"""
	MainModel uses all the functions below and actually simulates data.
	In order:
	initializes variables, arrays, etc
	loops through generations
		Mutate() current population
		creates offspring using Recombine()
		calculates phenotypes with zPhenotypes()
		calculates fitnesses with Fitness()
		chooses survivors based on fitnesses
		keep only survivors
		regulate to carrying capacity
	"""

	# First generation of Z genotypes
	genoZ = np.array([np.random.normal(np.repeat(0, nZloci*4*K), 2)])
	#genoZ = np.array(np.random.binomial(np.repeat(1, nZloci*4*K), .2),dtype=float)
	genoZ.shape = [K, 2, 2*nZloci]

	# First generation of R genotypes
	# We want to start them all at zero to see if the simulation generates correlation
	#genoR = np.array([np.random.normal(np.repeat(0, nZloci*4*K), 1)])
	genoR = np.array(np.repeat(0., nRloci*2*K),dtype=float)
	genoR.shape = [K, 2*nRloci]

	# Calculate R phenotypes for initializing generation
	# These phenotype values are NOT stored in the timeseries
	phenoR = rPhenotypes(genoR)

	# Initialize arrays to hold variable values
	mean_r_mu = np.empty(maxGen)
	G = np.empty(shape=[maxGen,2,2])
	zz = np.empty(shape=[maxGen,2])
	mean_fitness = np.empty(maxGen)

	################################
	# GENERATION LOOP
	################################
	for gen in range(maxGen):
		if genoZ.shape[0]<2:
			print('The population grew too small. Exiting...')
			break

		print('Generation ',gen)

		# Gametes mutate prior to mating. Mutate() doesn't return a value
		# because it actually updates genoZ and genoR IN the function
		Mutate(genoZ, genoR, phenoR, mu, alphaZ1, alphaZ2, alphaR)

		if np.isnan(genoZ).any():
			print('nan detected in genoZ after Mutate() (line 68). Exiting...')
			break

		# Mutated gametes recombine into the new population
		# Each mating pair (monogamous) produces 4 offspring according to
		# random assortment and unlinked loci (those are the same, right?)
		offspring = Recombine(genoZ, genoR)
		OSgenoZ = offspring['OSZ']
		OSgenoR = offspring['OSR']

		# Viability selection on the new population
		# Calculate Z phenotypes
		phenoZ = zPhenotypes(OSgenoZ)

		# Calculate fitnesses
		fitnesses = np.empty(phenoZ.shape[0])
		fitnesses = Fitness(phenoZ, omega, theta)

		# Choose survivors interpreting fitnesses as a vector of survival probabilities
		survivors = np.array(np.random.binomial(1,fitnesses),dtype=bool)

		# Redefine the genoZ and genoR vectors to be the survivors
		genoZ = OSgenoZ[survivors,:,:]
		phenoZ = phenoZ[survivors,:]
		genoR = OSgenoR[survivors,:]

		# Regulate population if we have more survivors than we can support
		if genoZ.shape[0]>K:
			sample = np.random.choice(range(genoZ.shape[0]),size=K,replace=False)
			genoZ = genoZ[sample,:,:]
			phenoZ = phenoZ[sample,:]
			genoR = genoR[sample,:]

		# Calculate individual mutational correlation phenotypes (r_mu) for current population
		phenoR = rPhenotypes(genoR)

		print('Population contains ',phenoR.size,' individuals.')

		# UPDATE AVERAGE r_mu
		mean_r_mu[gen] = np.mean(phenoR)

		# UPDATE G MATRIX
		G[gen,:,:] = np.cov(phenoZ,rowvar=0)

		# Save the average trait values
		zz[gen,:] = [np.mean(phenoZ[:,0]),np.mean(phenoZ[:,1])]
		print('Mean trait values (phenotype) are ',zz[gen,:])

		# Save mean fitness value
		mean_fitness[gen] = np.mean(fitnesses)
		print('Mean fitness is ',mean_fitness[gen])

	return({
		'G':G,
		'r_mu':mean_r_mu,
		'ztraits':zz,
		'fitness':mean_fitness
		})


def Mutate(genoZ, genoR, phenoR, mu, alphaZ1, alphaZ2, alphaR):
	"""
	Input: Z genotypes, mutational correlation genotypes, mutational variance
	genotypes.
	Mutate takes the current population and updates to a mutated version
	"""
	# Store the unmutated population of genoZ to calculate the covariance structure of the mutation effects
	unmutatedGenoZ = np.array(genoZ)

	# Choose mutation locations for Z genotypes
	mutZlocs = np.array(np.random.binomial(np.repeat(1,genoZ.shape[0]*2*nZloci), mu),dtype=bool)
	mutZlocs.shape = [genoZ.shape[0],2*nZloci] # number of individuals high, 2*nZloci wide (diploid)
	mutZlocs = np.repeat(mutZlocs, 2, axis=0) # repeat each line twice
	mutZlocs.shape = [genoZ.shape[0],2,2*nZloci] # reshape to [number of individuals deep, 2 high, 2*nZloci wide]

	# Choose mutation locations for R genotypes
	mutRlocs = np.array(np.random.binomial(np.repeat(1,2*nRloci*genoR.shape[0]), mu),dtype=bool)
	mutRlocs.shape = [genoR.shape[0],2*nRloci] # number of individuals high, 2*nRloci wide (diploid)

	# Add a bivariate Gaussian mutation with covariance Sigma to selected Z loci
	# Also add a univariate Gaussian mutation to selected R loci
	for i in range(genoZ.shape[0]): # for each individual in the population
		# Covariance for this individual, recall that phenoR is the mutational correlations and we need covariances
		Sigma = np.mat(
			[[alphaZ1, phenoR[i]*np.sqrt(alphaZ1*alphaZ2)],
			[phenoR[i]*np.sqrt(alphaZ1*alphaZ2), alphaZ2]])

		# Z mutations drawn from a bivariate normal
		# we draw a number of mutations equal to the sum of one row of booleans
		# dimensions are size high by 2 wide (bivariate)
		mutZ = mvn(
			mean=[0,0],
			cov=Sigma,
			size=np.sum(mutZlocs[i,0,:])
			)

		# Transpose and reshape such that we have all the trait1 mutations followed by all the trait2 mutations
		mutZ = np.reshape(np.transpose(mutZ), np.sum(mutZlocs[i,:,:]))

		# Add Z mutations to existing genotype. When we subset genoZ[i,:,:][---] it comes out in one long vector
		# We add the vector mutZ of [trait1mutations|trait2mutations] to the long vector genoZ[i,:,:][mutZlocs[i,:,:]]
		# without reshaping genoZ
		genoZ[i,:,:][mutZlocs[i,:,:]] = genoZ[i,:,:][mutZlocs[i,:,:]] + mutZ

		# R mutations drawn from univariate normal, variance alphaR
		mutR = np.random.normal(0,alphaR,np.sum(mutRlocs[i,:]))

		# Add mutations to existing genotype
		genoR[i,:][mutRlocs[i,:]] = genoR[i,:][mutRlocs[i,:]] + mutR

	return

def Recombine(genoZ, genoR):
	global nZloci, nRloci
	# Input: Z genotypes, R genotypes
	# Recombine() returns OSgenoZ and OSgenoR (offspring genotypes)

	# Declare arrays for offspring 4 times the current number
	OSgenoZ = np.empty(shape=[4*(genoZ.shape[0]),2,2*nZloci])
	OSgenoR = np.empty(shape=[4*(genoZ.shape[0]),2*nRloci])

	# non-assortative mating
	# no selfing
	# individuals may mate multiple times in different mating_events
	for mating_event in range(genoZ.shape[0]):
		# Choose random parents (without replacement)
		# Note that the next time through the loop, we can choose the same parents again
		[p1,p2] = np.random.choice(range(genoZ.shape[0]), size=2, replace=False)

		# Keep track of the parents' genotypes
		p1genoZ = genoZ[p1,:,:]
		p2genoZ = genoZ[p2,:,:]

		p1genoR = genoR[p1,:]
		p2genoR = genoR[p2,:]

		# Create 4 randomly assorted offspring, assuming loci are unlinked
		for o in range(4):
			## Starting with z1 and z2 ##
			# Make a gamete from parent 1
			these = np.array(np.random.binomial(np.repeat(1,nZloci),.5),dtype=bool)
			these = np.hstack((these,np.logical_not(these)))
			gamete1 = p1genoZ[:,these]

			# Make a gamete from parent 2
			these = np.array(np.random.binomial(np.repeat(1,nZloci),.5),dtype=bool)
			these = np.hstack((these,np.logical_not(these)))
			gamete2 = p2genoZ[:,these]

			OSgenoZ[(mating_event*4)+o,:,:] = np.hstack((gamete1,gamete2))

			## Now r_mu ##
			# Make a gamete from parent 1
			these = np.array(np.random.binomial(np.repeat(1,nRloci),.5),dtype=bool)
			these = np.hstack((these,np.logical_not(these)))
			gamete1 = p1genoR[these]

			# Make a gamete from parent 2
			these = np.array(np.random.binomial(np.repeat(1,nRloci),.5),dtype=bool)
			these = np.hstack((these,np.logical_not(these)))
			gamete2 = p2genoR[these]

			OSgenoR[(mating_event*4)+o,:] = np.hstack((gamete1,gamete2))

	return({'OSZ':OSgenoZ,'OSR':OSgenoR})

def rPhenotypes(genoR):
	# Input: R genotypes
	# rPhenotypes() calculates all the mutational correlation trait for each individual
	phenoR = np.sum(genoR, axis=1)

	# Scale correlations to be between -1 and 1
	phenoR = ( ( 2. * np.exp( 2. * phenoR ) ) / (1. + np.exp( 2. * phenoR ) ) ) - 1.
	return(phenoR)

def zPhenotypes(genoZ):
	# Input: Z genotypes
	# zPhenotypes() calculates the two trait phenotypes from the Z loci

	phenoZ = np.sum(genoZ, axis=2)

	# Add random normal environmental effects to the phenotypes with variance 1
	#phenoZ[:,0] = np.random.normal(phenoZ[:,0],1)
	#phenoZ[:,1] = np.random.normal(phenoZ[:,1],1)

	return(phenoZ)

def zEpistaticPhenotypes(genoZ):
	foo = 'bar'
	return

def Fitness(phenoZ, omega, theta):
	# Input: Z phenotypes, selection surface, fitness optimum
	# Fitness() returns the fitness vector for all individuals

	fit = np.empty(phenoZ.shape[0])
	for i in range(phenoZ.shape[0]):
		fit[i] = np.exp( -.5 * (np.mat(phenoZ[i,:]-theta)*np.mat(omega).I*np.mat(phenoZ[i,:]-theta).T) )
	return(fit)



