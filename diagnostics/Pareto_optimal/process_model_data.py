# This file is part of the Pareto_optimal diagnosis module of the MDTF code package (see LICENSE.txt)

# ======================================================================
# process_model_data.py
#
#   Called by Pareto_optimal.py
#   Interpolate Targeting GCM output to standard grids and calculate winter mean fields based on NCAR "NCL" package
#
#
#!/usr/bin/env python
# coding: utf-8

# ## import necessary libraries
import numpy
from netCDF4 import Dataset
import os
import subprocess
import time

def generate_ncl_plots(nclPlotFile):
    try:
        pipe = subprocess.Popen(['ncl {0}'.format(nclPlotFile)], shell=True, stdout=subprocess.PIPE)
        output = pipe.communicate()[0].decode()
        print('NCL routine {0} \n {1}'.format(nclPlotFile,output))
        while pipe.poll() is None:
            time.sleep(0.5)
    except OSError as e:
        print('WARNING',e.errno,e.strerror)
    return 0

# Process data from the targeting GCM, including calculation of winter mean (DJF) and interplotion to a standard grids

print("Calculating winter mean patterns from the targeting GCM using NCL ...")
generate_ncl_plots(os.environ["POD_HOME"]+"/m_win_mean.ncl")
