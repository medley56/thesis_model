# thesis_model
Evolutionary trajectory model describing genetic variance in a population over time.

simulation.py is a python module containing the functions necessary to simulate a 
timeseries of genetic data. i.e. we record the genetic covariance matrix G, mean 
mutational correlation r_mu, mean trait values (phenotype or breeding values), the
mutational effects covariance matrix M, along with metadata for each run.

Functions in simulation.py

MainModel(maxGen=500, K=100, theta=[1., -1.], omega=[[50., 0.],[0., 50.]],
	mu=0.0002, alphaZ1=0.01, alphaZ2=0.01, alphaR=0.01)
	Runs the whole show. This function contains the general structure of the model,
	including order of steps taken and also generating output to return. Note that 
	it runs with default values listed above (these will probably change).

Mutate(genoZ, genoR, phenoR, mu, alphaZ1, alphaZ2, alphaR)
	This function takes the current genotypes and mutates them according to
	parameter values. mu is the per locus mutation rate and Sigma (defined internally) 
	is the covariance matrix that governs the distribution of mutations entering the 
	population. Mutate also mutates the chromosomes that code for mutational correlation.

Recombine(genoZ, genoR)
	The mating function. We observe random mating with independent assortment of alleles. 
	Two random individuals are chosen to produce 4 offspring, randomly combined. The offspring
	are returned to MainModel. 

rPhenotypes(genoR)
	Calculates the mutational correlation variable from the genotypes genoR.

zPhenotypes(genoZ)
	Calculates the two trait phenotype from the genotypes genoZ.

Fitness(phenoZ, omega, theta)
	Calculates the fitness of an individual as a survival probability based on selection 
	strength matrix omega and trait optimum theta.
