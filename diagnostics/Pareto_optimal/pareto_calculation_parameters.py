# This file is part of the Pareto_optimal diagnosis module of the MDTF code package (see LICENSE.txt)

# ======================================================================
# pareto_calculation_parameters.py
#
#   Called by Pareto_optimal.py
#   Set paremeters for Pareto_optimal diagnosis
#
#
#!/usr/bin/env python
# coding: utf-8
# 
import numpy
import glob
import os
#import json

degree_sign = u'\u00B0'

### create list of model names (alphabetical order)

target_model_names = 'GFDL-CM4'

cmip_model_names = numpy.array(( 'ACCESS-CM2', 'ACCESS-ESM1-5', 'AWI-CM-1-1-MR', 'BCC-CSM2-MR', 'CAMS-CSM1-0', 'CanESM5-CanOE', 'CanESM5', 'CESM2', 'CESM2-WACCM', 'CMCC-CM2-SR5', 'CNRM-CM6-1-HR', 'CNRM-CM6-1', 'CNRM-ESM2-1', 'E3SM-1-1', 'EC-Earth3', 'EC-Earth3-Veg-LR', 'EC-Earth3-Veg', 'FGOALS-g3', 'GFDL-ESM4', 'GISS-E2-1-G', 'HadGEM3-GC31-LL', 'HadGEM3-GC31-MM', 'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'KACE-1-0-G', 'MIROC6', 'MIROC-ES2L', 'MPI-ESM1-2-HR', 'NESM3', 'NorESM2-LM', 'NorESM2-MM', 'UKESM1-0-LL' ))

exp_name='CA'
#exp_name='SAM'


if exp_name=='CA':
     vlist = numpy.array(( 'pr','tos','ua' ))
     uwind_level=200
### specify the lat/lon bounding regions for the objective function calculations
     x_lat_lo, x_lat_hi, x_lon_lo, x_lon_hi = 30., 45., 232.5, 248; region = 'CA'
     y_lat_lo, y_lat_hi, y_lon_lo, y_lon_hi = -30., 10., 155., 270.; region = 'tropacific'
     z_lat_lo, z_lat_hi, z_lon_lo, z_lon_hi = 20., 50., 170., 250.; region = 'midlatpacific'  # for u200
### specify the lat/lon bounding regions for the plotting the spatial ditribution of each field and model simulations (biases)
     x_lat_lo_plt, x_lat_hi_plt, x_lon_lo_plt, x_lon_hi_plt = 25., 65., 190., 270.
     y_lat_lo_plt, y_lat_hi_plt, y_lon_lo_plt, y_lon_hi_plt = -45., 45., 120., 300.
     z_lat_lo_plt, z_lat_hi_plt, z_lon_lo_plt, z_lon_hi_plt = 10., 65., 150., 260.
if exp_name=='SAM':
     vlist = numpy.array(( 'pr','tos','ua' ))
     uwind_level=850
### specify the lat/lon bounding regions for the objective function calculations
     x_lat_lo, x_lat_hi, x_lon_lo, x_lon_hi = -15., 0., 285., 310.; region = 'South American monsoon'
   # y_lat_lo, y_lat_hi, y_lon_lo, y_lon_hi = -10., 10., 160., 270.; region = 'tropacific - Nino3.4'
     y_lat_lo, y_lat_hi, y_lon_lo, y_lon_hi = -10., 10., 160., 345.; region = 'tropacific - Nino3.4'
     z_lat_lo, z_lat_hi, z_lon_lo, z_lon_hi = -10.,  5., 296., 345.; region = 'equatorial Atlantic'  # for u850 to avoid missing data
### specify the lat/lon bounding regions for the plotting the spatial ditribution of each field and model simulations (biases)
     x_lat_lo_plt, x_lat_hi_plt, x_lon_lo_plt, x_lon_hi_plt = -40., 20., 240., 360.
  #  y_lat_lo_plt, y_lat_hi_plt, y_lon_lo_plt, y_lon_hi_plt = -45., 45., 120., 300.
     y_lat_lo_plt, y_lat_hi_plt, y_lon_lo_plt, y_lon_hi_plt = -45., 45., 120., 360.
     z_lat_lo_plt, z_lat_hi_plt, z_lon_lo_plt, z_lon_hi_plt = -40., 25., 210., 355.

season='djf';

pareto_k_values = [1,2,3]

N_pareto_loops=5

# -DO NOT make any changes on the following codes if not necesaary .

para={}

para["degree_sign"]=degree_sign

para["target_model_names"]=target_model_names
para["cmip_model_names"]=cmip_model_names
para["vlist"]=vlist

para["x_lat_lo"]=x_lat_lo
para["y_lat_lo"]=y_lat_lo
para["z_lat_lo"]=z_lat_lo
para["x_lat_hi"]=x_lat_hi
para["y_lat_hi"]=y_lat_hi
para["z_lat_hi"]=z_lat_hi
para["x_lon_lo"]=x_lon_lo
para["y_lon_lo"]=y_lon_lo
para["z_lon_lo"]=z_lon_lo
para["x_lon_hi"]=x_lon_hi
para["y_lon_hi"]=y_lon_hi
para["z_lon_hi"]=z_lon_hi

para["x_lat_lo_plt"]=x_lat_lo_plt
para["y_lat_lo_plt"]=y_lat_lo_plt
para["z_lat_lo_plt"]=z_lat_lo_plt
para["x_lat_hi_plt"]=x_lat_hi_plt
para["y_lat_hi_plt"]=y_lat_hi_plt
para["z_lat_hi_plt"]=z_lat_hi_plt
para["x_lon_lo_plt"]=x_lon_lo_plt
para["y_lon_lo_plt"]=y_lon_lo_plt
para["z_lon_lo_plt"]=z_lon_lo_plt
para["x_lon_hi_plt"]=x_lon_hi_plt
para["y_lon_hi_plt"]=y_lon_hi_plt
para["z_lon_hi_plt"]=z_lon_hi_plt

para["season"]=season
para["pareto_k_values"]=pareto_k_values
para["N_pareto_loops"]=N_pareto_loops
para["uwind_level"]=uwind_level
para["exp_name"]=exp_name

numpy.save("pareto_parameters.npy",para)
