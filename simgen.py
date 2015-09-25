"""
Evolutionary Trajectory Simulation Model
Written by Gavin Medley
University of Colorado
"""

# Import the simulation module for simulation functions
import simulation as sim

# Import necessary module dependencies
import numpy as np
import json
import os
import csv
import time

# Change the pwd to the one containing simulation.py
os.chdir('/home/gmedley/Documents/Thesis/TrajectoryModel/')

# Set the number of runs to execute for each set of parameter values
nRuns = 5

# Set all the parameters for each run. p is a list of dictionaries and
# each dictionary is a set of parameter values
p = {'maxGen':1000, 'K':512, 'theta':[10, -10], 'omega':[[100, 10],[10, 100]],
		'mu':0.0005, 'alphaZ1':0.05, 'alphaZ2':0.05, 'alphaR':0.05,
		'epiSigma':0.1, 'initSigma':1.0}

# Repeat p nRuns times. In other cases, we can manually repeat p to
# change parameter values for different runs. Perhaps we won't need to.
#p = list(np.repeat(p, nRuns))

timingFile = open('ModelTiming.txt','w')
startTime = time.time()
timingFile.write('startTime ='+str(startTime)+'\n')

for run in range(nRuns):
	# Let's do some timing calculations
	if run==1:
		endTime = time.time()
		elapsedTime = endTime-startTime
		estTotalTime = elapsedTime*nRuns*nRuns
		timingFile.write('endTime = '+str(endTime)+'\n')
		timingFile.write('Time elapsed after one run ('+str(p['maxGen'])+' generations) = '+str(elapsedTime)+'\n')
		timingFile.write('Estimated time to completion is '+str(estTotalTime)+' seconds.\n')
		timingFile.close()

    # Make a directory to store the data from the current run
	runfilename = 'MostRecentExecution/run'+str(run)
	os.makedirs(runfilename)

    # Store everything that MainModel returns
	output = sim.MainModel(
		maxGen=p['maxGen'], K=p['K'], theta=p['theta'],
		omega=p['omega'], mu=p['mu'], alphaZ1=p['alphaZ1'],
		alphaZ2=p['alphaZ2'], alphaR=p['alphaR'],
		epiSigma=p['epiSigma'], initSigma=p['initSigma'])

	# Write all the history values to files in their own directories
	np.save(runfilename+'/gmatrix_history.npy', output['G'])
	np.save(runfilename+'/mean_rmu_history.npy', output['r_mu'])
	np.save(runfilename+'/z_trait_history.npy', output['ztraits'])
	np.save(runfilename+'/fitness_history.npy', output['fitness'])
	np.save(runfilename+'/mmatrix_history.npy', output['M'])

	# json dump the parameter dictionary so we can load/use it during analysis
	json.dump(p,open('MostRecentExecution/params.txt','w'))

	with open('MostRecentExecution/metadata.csv','w') as metafile:
		writer = csv.writer(metafile, delimiter=':')
		for key, value in p.items():
			writer.writerow([key,value])