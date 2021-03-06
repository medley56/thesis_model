"""
Analysis of simulation data generated by trajectory.py
Written by Gavin Medley
University of Colorado
"""

# Import simanalysis module for analysis functions
import simanalysis as sa

# Import the data from files using ImportData()
timeseries = sa.ImportData()

sa.MVis(timeseries, smoothing=False, nPts=2)

#sa.MVis(rMuRuns, MRuns, smoothing=False, nPts=2)

#MVis(rMuRuns[0:6,:], paramRuns)