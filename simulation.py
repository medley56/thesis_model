"""
Module containing functions used in the course of generating simulation data
Written by Gavin Medley
University of Colorado
"""

# Import necessary module dependencies
import numpy as np

# Declare a few variables to be global
global nZloci, nRloci
nZloci = 20
nRloci = 20

################################
# Defining functions here
################################

# The call to make multivariate normal draws sucks so:
mvn = np.random.multivariate_normal

# Default values are meant for testing the model
def MainModel(maxGen=500, K=512, theta=[0., 0.], omega=[[49., 0.],[0., 49.]],
	mu=0.0005, alphaZ1=0.05, alphaZ2=0.05, alphaR=0.05, epiSigma=1.0, initSigma=1.0):
	# alphas are variances, not standard deviations
	# epi sigma is also a variance. Note that np.random.normal requires the sd not the variance as input
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
	# Epistatic coefficient matrices. A draw from random normals.
	# epistaticCoeffMat[0,:,:] holds the coefficients for trait 1
	# epistaticCoeffMat[1,:,:] holds the coefficients for trait 2
	epistaticCoeffs = np.random.normal(0,np.sqrt(epiSigma),size=[2,nZloci,nZloci])
	# Make them strictly upper triangular i.e zero diagonal
	epistaticCoeffs[0,:,:] = np.triu(epistaticCoeffs[0,:,:],+1)
	epistaticCoeffs[1,:,:] = np.triu(epistaticCoeffs[1,:,:],+1)

	# First generation of Z genotypes
	#genoZ = np.array([np.random.normal(np.repeat(0, nZloci*4*K), 2)])
	genoZ = np.random.normal(0,initSigma,size=nZloci*4*5*K)
	genoZ.shape = [5*K, 2, 2*nZloci]

	# First generation of R genotypes
	# We want to start them all at zero to see if the simulation generates correlation
	#genoR = np.array([np.random.normal(np.repeat(0, nZloci*4*K), 1)])
	genoR = np.array(np.repeat(0., nRloci*2*5*K),dtype=float)
	genoR.shape = [5*K, 2*nRloci]

	# Calculate R phenotypes for initializing generation
	# These phenotype values are NOT stored in the timeseries
	phenoR = rPhenotypes(genoR)

	# Initialize arrays to hold variable values
	mean_r_mu = np.empty(maxGen)
	G = np.empty(shape=[maxGen,2,2])
	M = np.empty(shape=[maxGen,2,2])
	zz = np.empty(shape=[maxGen,2])
	mean_fitness = np.empty(maxGen)

	################################
	# GENERATION LOOP
	################################
	for gen in range(maxGen):

		print('Generation ',gen)

		# Store the unmutated population of genoZ to calculate the covariance
		# structure of the effects of mutation
		UMgenoZ = np.array(genoZ)

		# Gametes mutate prior to mating. Mutate() doesn't return a value
		# because it actually updates genoZ and genoR IN the function
		# I think this is ecologically sound because mutations occur
		# prior to fertilization during meiosis, right?
		Mutate(genoZ, genoR, phenoR, mu, alphaZ1, alphaZ2, alphaR)

		# Calculate the difference between mutated and unmutated phenotypes
		phenotypeChange = zEpistaticPhenotypes(UMgenoZ, epistaticCoeffs)-zEpistaticPhenotypes(genoZ, epistaticCoeffs)
		#print('original difference',phenotypeChange)
		phenotypeChange = np.array(phenotypeChange[phenotypeChange.any(axis=1)!=0])
		#print('difference without zeros', phenotypeChange)

		# Mutated gametes recombine into the new population
		# Each mating pair (monogamous) produces 4 offspring according to
		# random assortment and unlinked loci (those are the same, right?)
		offspring = Recombine(genoZ, genoR)
		OSgenoZ = np.array(offspring['OSZ'])
		OSgenoR = np.array(offspring['OSR'])

		# Viability selection on the new population
		# Calculate Z phenotypes
		OSphenoZ = zEpistaticPhenotypes(OSgenoZ, epistaticCoeffs)

		print('Mean offspring phenotypes prior to selection: ',np.mean(OSphenoZ,axis=0))
		# Calculate fitnesses
		#fitnesses = np.empty(OSphenoZ.shape[0])
		fitnesses = Fitness(OSphenoZ, omega, theta)

		# Choose survivors interpreting fitnesses as a vector of survival probabilities
		survivors = np.array(np.random.binomial(1,fitnesses),dtype=bool)

		# Redefine the genoZ and genoR vectors to be the survivors
		genoZ = np.array(OSgenoZ[survivors,:,:])
		genoR = np.array(OSgenoR[survivors,:])
		# Also chop down the phenoZ array to only the survivors
		phenoZ = np.array(OSphenoZ[survivors,:])

		# Regulate population if we have more survivors than we can support
		if genoZ.shape[0]>K:
			print('Regulating population down to carrying capacity.')
			sample = np.random.choice(range(genoZ.shape[0]),size=K,replace=False)
			genoZ = np.array(genoZ[sample,:,:])
			phenoZ = np.array(phenoZ[sample,:])
			genoR = np.array(genoR[sample,:])
		elif genoZ.shape[0]<2:
			print('The population went extinct. Breaking loop and moving to next run.')
			break

		print('Mean offspring phenotypes after selection',np.mean(phenoZ,axis=0))

		# Calculate individual mutational correlation phenotypes (r_mu) for current population
		phenoR = rPhenotypes(genoR)

		print('Population contains ',genoZ.shape[0],' individuals.')

		# UPDATE AVERAGE r_mu
		mean_r_mu[gen] = np.mean(phenoR)

		# UPDATE G MATRIX
		G[gen,:,:] = np.cov(phenoZ,rowvar=0)

		# UDATE M MATRIX
		# Around line 72, we saved the unmutated genotypes as UMgenoZ and
		# calculated the differences between the mutated phenotypes and unmutated.
		# Calculate the M matrix for the effects of the mutations that went into
		# the newest generation. i.e. the mutations that occurred during meiosis this time
		if phenotypeChange.shape[0]!=0: # often there will be no mutations so we sometimes have to set M to zeros
			M[gen,:,:] = np.cov(phenotypeChange,rowvar=0)
		else:
			M[gen,:,:] = np.zeros(shape=[2,2])

		print('Effects of mutation matrix M, is ',M[gen,:,:])

		# Save the average trait values
		zz[gen,:] = np.mean(phenoZ,axis=0)
		print('Mean trait values (phenotype) are ',zz[gen,:])

		# Save mean fitness value
		mean_fitness[gen] = np.mean(fitnesses)
		print('Mean fitness is ',mean_fitness[gen])

	return({
		'G':G,
		'M':M,
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

	# Choose mutation locations for Z genotypes
	mutZlocs = np.array(np.random.binomial(np.repeat(1,genoZ.shape[0]*2*nZloci), mu),dtype=bool)
	mutZlocs.shape = [genoZ.shape[0],2*nZloci] # number of individuals high, 2*nZloci wide (diploid)
	mutZlocs = np.repeat(mutZlocs, 2, axis=0) # repeat each line twice
	mutZlocs.shape = [genoZ.shape[0],2,2*nZloci] # reshape to [number of individuals deep, 2 high, 2*nZloci wide]

	if np.sum(mutZlocs) != 0:
		print('Mutations in genoZ')

	# Choose mutation locations for R genotypes
	mutRlocs = np.array(np.random.binomial(np.repeat(1,2*nRloci*genoR.shape[0]), mu),dtype=bool)
	mutRlocs.shape = [genoR.shape[0],2*nRloci] # number of individuals high, 2*nRloci wide (diploid)

	if np.sum(mutRlocs) != 0:
		print('Mutations in genoR')

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
		mutR = np.random.normal(0,np.sqrt(alphaR),np.sum(mutRlocs[i,:]))

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

def zEpistaticPhenotypes(genoZ,epistaticCoeffs):
	additiveEffects = np.sum(genoZ, axis=2)
	Z1epistaticEffects = np.empty(shape=[genoZ.shape[0]])
	Z2epistaticEffects = np.empty(shape=[genoZ.shape[0]])
	# Add the effects on different chromosomes together (i.e. sum across chromosomes)
	# genoZ.shape should now be [population, 2, nZloci]
	genoZSum = genoZ[:,:,:nZloci] + genoZ[:,:,nZloci:]

	# Loop through each individual, calculating it's epistatic phenotype for each trait.
	for i in range(genoZSum.shape[0]):
		# sum the effects of each locus on trait 1
		# vector genoZ[i,0,:] %*% coeff_matrix ON Z1 %*% genoZ[i,1,:]transpose
		Z1epistaticEffects[i] = np.dot( np.dot(genoZSum[i,0,:], epistaticCoeffs[0,:,:]), genoZSum[i,1,:].T)
		# sum the effects of each locus on trait 2
		# vector genoZ[i,0,:] %*% coeff_matrix ON Z2
		Z2epistaticEffects[i] = np.dot( np.dot(genoZSum[i,0,:], epistaticCoeffs[1,:,:]), genoZSum[i,1,:].T)

	phenoZ = additiveEffects + np.array([Z1epistaticEffects,Z2epistaticEffects]).T

	return(phenoZ)

def Fitness(phenoZ, omega, theta):
	# Input: Z phenotypes, selection surface, fitness optimum
	# Fitness() returns the fitness vector for all individuals

	fit = np.empty(phenoZ.shape[0])
	for i in range(phenoZ.shape[0]):
		fit[i] = np.exp( -.5 * (np.mat(phenoZ[i,:]-theta)*np.mat(omega).I*np.mat(phenoZ[i,:]-theta).T) )
	return(fit)



