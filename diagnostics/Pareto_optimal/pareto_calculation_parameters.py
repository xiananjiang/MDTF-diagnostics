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
import subprocess
import os.path

## List of model names (alphabetical order)

target_model_names = 'GFDL-CM4'

cmip_model_names = numpy.array(( 'ACCESS-CM2', 'ACCESS-ESM1-5', 'AWI-CM-1-1-MR', 'BCC-CSM2-MR', 'CAMS-CSM1-0', 'CanESM5-CanOE', 'CanESM5', 'CESM2', 'CESM2-WACCM', 'CMCC-CM2-SR5', 'CNRM-CM6-1-HR', 'CNRM-CM6-1', 'CNRM-ESM2-1', 'E3SM-1-1', 'EC-Earth3', 'EC-Earth3-Veg-LR', 'EC-Earth3-Veg', 'FGOALS-g3', 'GFDL-ESM4', 'GISS-E2-1-G', 'HadGEM3-GC31-LL', 'HadGEM3-GC31-MM', 'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'KACE-1-0-G', 'MIROC6', 'MIROC-ES2L', 'MPI-ESM1-2-HR', 'NESM3', 'NorESM2-LM', 'NorESM2-MM', 'UKESM1-0-LL' ))

exp_name='CA'
#exp_name='SAM'

if exp_name=='CA':
     vlist = numpy.array(( 'pr','tos','ua' )) # this list of variable names, labels, and units needs to be consistent with the "varlist" in the "settings.jsonc" file
     vlist_label = numpy.array(( 'Precip','SST','U200' ))
     vlist_unit = numpy.array(( 'mm day$^{\,-1}$',u'\u00B0C','m s$^{-1}$' ))
     uwind_level=200
     if 'ua' in vlist:
        file_exists = os.path.exists(os.environ["OBS_DATA"]+"/OBS_ua_1980-2010_climatology_djf.nc")
        if (file_exists):
           subprocess.call(['/bin/csh', '-c', "rm "+os.environ["OBS_DATA"]+"/OBS_ua_1980-2010_climatology_djf.nc"])
        subprocess.call(['/bin/csh', '-c', "ln -s "+os.environ["OBS_DATA"]+"/OBS_ua_1980-2010_climatology_djf_"+exp_name+".nc "+os.environ["OBS_DATA"]+"/OBS_ua_1980-2010_climatology_djf.nc"])
        subprocess.call(['/bin/csh', '-c', "ls -l "+os.environ["OBS_DATA"]+"/OBS_ua_1980-2010_climatology_djf.nc"])
        file_exists = os.path.exists(os.environ["OBS_DATA"]+"/cmip6_models_ua_1980-2010_climatology_djf.nc")
        if (file_exists):
           subprocess.call(['/bin/csh', '-c', "rm "+os.environ["OBS_DATA"]+"/cmip6_models_ua_1980-2010_climatology_djf.nc"])
        subprocess.call(['/bin/csh', '-c', "ln -s "+os.environ["OBS_DATA"]+"/cmip6_models_ua_1980-2010_climatology_djf_"+exp_name+".nc "+os.environ["OBS_DATA"]+"/cmip6_models_ua_1980-2010_climatology_djf.nc"])
        subprocess.call(['/bin/csh', '-c', "ls -l "+os.environ["OBS_DATA"]+"/cmip6_models_ua_1980-2010_climatology_djf.nc"])
          
## specify the lat/lon bounding regions for the objective function calculations: x, y, z corresponding to the 1-3 variables in vlist.
     x_lat_lo, x_lat_hi, x_lon_lo, x_lon_hi = 30., 45., 232.5, 248; region = 'CA'
     y_lat_lo, y_lat_hi, y_lon_lo, y_lon_hi = -30., 10., 155., 270.; region = 'tropacific'
     z_lat_lo, z_lat_hi, z_lon_lo, z_lon_hi = 20., 50., 170., 250.; region = 'midlatpacific'  
## specify the lat/lon bounding regions for the plotting the spatial ditribution of each field and model simulations (biases)
     x_lat_lo_plt, x_lat_hi_plt, x_lon_lo_plt, x_lon_hi_plt = 25., 65., 190., 270.
     y_lat_lo_plt, y_lat_hi_plt, y_lon_lo_plt, y_lon_hi_plt = -45., 45., 120., 300.
     z_lat_lo_plt, z_lat_hi_plt, z_lon_lo_plt, z_lon_hi_plt = 10., 65., 150., 260.

if exp_name=='SAM':
     vlist = numpy.array(( 'pr','tos','ua' ))
     vlist_label = numpy.array(( 'Precip','SST','U850' ))
     vlist_unit = numpy.array(( 'mm day$^{\,-1}$',u'\u00B0C','m s$^{-1}$' ))
     uwind_level=850
     if 'ua' in vlist:
        file_exists = os.path.exists(os.environ["OBS_DATA"]+"/OBS_ua_1980-2010_climatology_djf.nc")
        if (file_exists):
           subprocess.call(['/bin/csh', '-c', "rm "+os.environ["OBS_DATA"]+"/OBS_ua_1980-2010_climatology_djf.nc"])
        subprocess.call(['/bin/csh', '-c', "ln -s "+os.environ["OBS_DATA"]+"/OBS_ua_1980-2010_climatology_djf_"+exp_name+".nc "+os.environ["OBS_DATA"]+"/OBS_ua_1980-2010_climatology_djf.nc"])
        file_exists = os.path.exists(os.environ["OBS_DATA"]+"/cmip6_models_ua_1980-2010_climatology_djf.nc")
        if (file_exists):
           subprocess.call(['/bin/csh', '-c', "rm "+os.environ["OBS_DATA"]+"/cmip6_models_ua_1980-2010_climatology_djf.nc"])
        subprocess.call(['/bin/csh', '-c', "ln -s "+os.environ["OBS_DATA"]+"/cmip6_models_ua_1980-2010_climatology_djf_"+exp_name+".nc "+os.environ["OBS_DATA"]+"/cmip6_models_ua_1980-2010_climatology_djf.nc"])
          
## specify the lat/lon bounding regions for the objective function calculations
     x_lat_lo, x_lat_hi, x_lon_lo, x_lon_hi = -15., 0., 285., 310.; region = 'South American monsoon'
     y_lat_lo, y_lat_hi, y_lon_lo, y_lon_hi = -10., 10., 160., 345.; region = 'tropical pacific and Atlantic SST'
     z_lat_lo, z_lat_hi, z_lon_lo, z_lon_hi = -10.,  5., 296., 345.; region = 'equatorial Atlantic'  
## specify the lat/lon bounding regions for the plotting the spatial ditribution of each field and model simulations (biases)
     x_lat_lo_plt, x_lat_hi_plt, x_lon_lo_plt, x_lon_hi_plt = -40., 20., 240., 360.
     y_lat_lo_plt, y_lat_hi_plt, y_lon_lo_plt, y_lon_hi_plt = -45., 45., 120., 360.
     z_lat_lo_plt, z_lat_hi_plt, z_lon_lo_plt, z_lon_hi_plt = -40., 25., 210., 355.

## define winter season
season='djf';

## define number of model combinations to be included for caculation of pareto_optimal fronts
## e.g., k=1 reprsents combinations with one model from total N models, k=2 represents all combinations with 2 models from total N models, and so on.
pareto_k_values = [1,2,3]   # total model combinations with selecting upto 3 models from all total N models

## -DO NOT make changes on the following codes if not necesaary .
para={}

para["target_model_names"]=target_model_names
para["cmip_model_names"]=cmip_model_names
para["vlist"]=vlist
para["vlist_label"]=vlist_label
para["vlist_unit"]=vlist_unit

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
para["uwind_level"]=uwind_level
para["exp_name"]=exp_name

numpy.save("pareto_parameters.npy",para)
