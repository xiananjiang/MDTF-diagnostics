# This file is part of the Pareto_optimal diagnosis module of the MDTF code package (see LICENSE.txt)

# ======================================================================
# pareto_calculation_parameters.py
#
#   Called by Pareto_optimal.py
#   Input preprocessed climatological fields from OBS and multiple CMIP6 simulations
#
#
#!/usr/bin/env python
# coding: utf-8

import numpy
from netCDF4 import Dataset
import os

# load parameters 
para = numpy.load("pareto_parameters.npy",allow_pickle=True)

target_model_names=para[()]["target_model_names"]
cmip_model_names=para[()]["cmip_model_names"]
vlist=para[()]["vlist"]

model_names=numpy.hstack((numpy.array(target_model_names), cmip_model_names))

# the objective function bounding boxes
x_lat_lo=para[()]["x_lat_lo"]
y_lat_lo=para[()]["y_lat_lo"]
z_lat_lo=para[()]["z_lat_lo"]
x_lat_hi=para[()]["x_lat_hi"]
y_lat_hi=para[()]["y_lat_hi"]
z_lat_hi=para[()]["z_lat_hi"]
x_lon_lo=para[()]["x_lon_lo"]
y_lon_lo=para[()]["y_lon_lo"]
z_lon_lo=para[()]["z_lon_lo"]
x_lon_hi=para[()]["x_lon_hi"]
y_lon_hi=para[()]["y_lon_hi"]
z_lon_hi=para[()]["z_lon_hi"]

# lat/lon bounding regions of the map plots
x_lat_lo_plt=para[()]["x_lat_lo_plt"]
y_lat_lo_plt=para[()]["y_lat_lo_plt"]
z_lat_lo_plt=para[()]["z_lat_lo_plt"]
x_lat_hi_plt=para[()]["x_lat_hi_plt"]
y_lat_hi_plt=para[()]["y_lat_hi_plt"]
z_lat_hi_plt=para[()]["z_lat_hi_plt"]
x_lon_lo_plt=para[()]["x_lon_lo_plt"]
y_lon_lo_plt=para[()]["y_lon_lo_plt"]
z_lon_lo_plt=para[()]["z_lon_lo_plt"]
x_lon_hi_plt=para[()]["x_lon_hi_plt"]
y_lon_hi_plt=para[()]["y_lon_hi_plt"]
z_lon_hi_plt=para[()]["z_lon_hi_plt"]

nmods = len(model_names)

season=para[()]["season"]
uwind_level=para[()]["uwind_level"]
exp_name=para[()]["exp_name"]

# Import preprocessed observational dataset climatologies
# All data sets regridded onto a common 2.5-degree grid (72x144 latxlon)

# OPEN PR OBSERVATIONS
ncfile = Dataset(os.environ["OBS_DATA"]+'/OBS_'+vlist[0]+'_1980-2010_climatology_djf.nc', 'r', format='NETCDF4')
x_lat = ncfile.variables['lat'][:]
x_lon = ncfile.variables['lon'][:]
x_lat_inds = numpy.where((x_lat>=x_lat_lo_plt) & (x_lat<=x_lat_hi_plt))[0]
x_lon_inds = numpy.where((x_lon>=x_lon_lo_plt) & (x_lon<=x_lon_hi_plt))[0]
obs_field_x = ncfile.variables[vlist[0]][x_lat_inds[0]:(x_lat_inds[-1]+1), x_lon_inds[0]:(x_lon_inds[-1]+1)]
x_regional_nlat, x_regional_nlon = obs_field_x.shape

x_regional_lat_vals = x_lat[x_lat_inds[0]:(x_lat_inds[-1]+1)]
x_regional_lon_vals = x_lon[x_lon_inds[0]:(x_lon_inds[-1]+1)]

# OPEN TS OBSERVATIONS
ncfile = Dataset(os.environ["OBS_DATA"]+'/OBS_'+vlist[1]+'_1980-2010_climatology_djf.nc', 'r', format='NETCDF4')
y_lat = ncfile.variables['lat'][:]
y_lon = ncfile.variables['lon'][:]
y_lat_inds = numpy.where((y_lat>=y_lat_lo_plt) & (y_lat<=y_lat_hi_plt))[0]
y_lon_inds = numpy.where((y_lon>=y_lon_lo_plt) & (y_lon<=y_lon_hi_plt))[0]
obs_field_y = ncfile.variables[vlist[1]][y_lat_inds[0]:(y_lat_inds[-1]+1), y_lon_inds[0]:(y_lon_inds[-1]+1)]
y_regional_nlat, y_regional_nlon = obs_field_y.shape

y_regional_lat_vals = y_lat[y_lat_inds[0]:(y_lat_inds[-1]+1)]
y_regional_lon_vals = y_lon[y_lon_inds[0]:(y_lon_inds[-1]+1)]

# land sea mask
ncfile = Dataset(os.environ["OBS_DATA"]+'/landsea.nc', 'r', 'NetCDF4')
landsea_data = ncfile.variables['landsea'][y_lat_inds[0]:(y_lat_inds[-1]+1), y_lon_inds[0]:(y_lon_inds[-1]+1)]

# OPEN UA OBSERVATIONS
ncfile = Dataset(os.environ["OBS_DATA"]+'/OBS_'+vlist[2]+'_1980-2010_climatology_djf.nc', 'r', format='NETCDF4')
z_lat = ncfile.variables['lat'][:]
z_lon = ncfile.variables['lon'][:]
z_lat_inds = numpy.where((z_lat>=z_lat_lo_plt) & (z_lat<=z_lat_hi_plt))[0]
z_lon_inds = numpy.where((z_lon>=z_lon_lo_plt) & (z_lon<=z_lon_hi_plt))[0]
obs_field_z = numpy.zeros((len(z_lat_inds),len(z_lon_inds)))
obs_field_z = ncfile.variables[vlist[2]][z_lat_inds[0]:(z_lat_inds[-1]+1), z_lon_inds[0]:(z_lon_inds[-1]+1)]
z_regional_nlat, z_regional_nlon = obs_field_z.shape

z_regional_lat_vals = z_lat[z_lat_inds[0]:(z_lat_inds[-1]+1)]
z_regional_lon_vals = z_lon[z_lon_inds[0]:(z_lon_inds[-1]+1)]

# set up data
model_data_hist_x = numpy.zeros((len(model_names), x_regional_nlat, x_regional_nlon))
model_data_hist_y = numpy.zeros((len(model_names), y_regional_nlat, y_regional_nlon))
model_data_hist_z = numpy.zeros((len(model_names), z_regional_nlat, z_regional_nlon))

# import precipiation data from the targeting GCM (units: mm day-1, converted from kg m-2 s-1 in in original model data file)
ncfile = Dataset(os.environ["WK_DIR"]+'/model/netCDF/model_'+vlist[0]+'_climatology_djf.nc', 'r', format='NETCDF4')
model_data_hist_x[0,:,:] = ncfile.variables[vlist[0]][x_lat_inds[0]:(x_lat_inds[-1]+1), x_lon_inds[0]:(x_lon_inds[-1]+1)]
ncfile.close()

# import skin temperature data from the targeting GCM (units: degC, converted from K in original model data file)
ncfile = Dataset(os.environ["WK_DIR"]+'/model/netCDF/model_'+vlist[1]+'_climatology_djf.nc', 'r', format='NETCDF4')
y_temp = ncfile.variables[vlist[1]][y_lat_inds[0]:(y_lat_inds[-1]+1), y_lon_inds[0]:(y_lon_inds[-1]+1)]
if vlist[1]=='tos':
    y_temp[landsea_data>1000000]=numpy.nan
model_data_hist_y[0,:,:] = y_temp
ncfile.close()

# import single pressure-level wind data for the targeting GCM (units: m s-1)
ncfile = Dataset(os.environ["WK_DIR"]+'/model/netCDF/model_'+vlist[2]+'_climatology_djf.nc', 'r', format='NETCDF4')
model_data_hist_z[0,:,:] = ncfile.variables[vlist[2]][z_lat_inds[0]:(z_lat_inds[-1]+1), z_lon_inds[0]:(z_lon_inds[-1]+1)]  
ncfile.close()

for i in range(1,nmods):

    modelname = model_names[i]
#   print(i,modelname)

# import preprocessed precipitation data for cmip6 models (units: mm day-1)
    ncfile = Dataset(os.environ["OBS_DATA"]+'/cmip6_models_'+vlist[0]+'_1980-2010_climatology_djf.nc', 'r', format='NETCDF4')
    model_data_hist_x[i,:,:] = ncfile.variables[vlist[0]][i-1,x_lat_inds[0]:(x_lat_inds[-1]+1), x_lon_inds[0]:(x_lon_inds[-1]+1)]
    ncfile.close()

# import preprocessed skin temperature data for cmip6 models (units: degC)
    ncfile = Dataset(os.environ["OBS_DATA"]+'/cmip6_models_'+vlist[1]+'_1980-2010_climatology_djf.nc', 'r', format='NETCDF4')
    model_data_hist_y[i:,:,:] = ncfile.variables[vlist[1]][i-1,y_lat_inds[0]:(y_lat_inds[-1]+1), y_lon_inds[0]:(y_lon_inds[-1]+1)]
    ncfile.close()

# import preprocessed single pressure-level wind data for cmip6 models (units: m s-1)
    ncfile = Dataset(os.environ["OBS_DATA"]+'/cmip6_models_'+vlist[2]+'_1980-2010_climatology_djf.nc', 'r', format='NETCDF4')
    model_data_hist_z[i:,:,:] = ncfile.variables[vlist[2]][i-1,z_lat_inds[0]:(z_lat_inds[-1]+1), z_lon_inds[0]:(z_lon_inds[-1]+1)]
    ncfile.close()
    model_data_hist_z[abs(model_data_hist_z)>100.]=numpy.nan

data={}

data["x_lat_inds"]=x_lat_inds
data["x_lon_inds"]=x_lon_inds
data["x_regional_nlat"]=x_regional_nlat
data["x_regional_nlon"]=x_regional_nlon
data["x_regional_lat_vals"]=x_regional_lat_vals
data["x_regional_lon_vals"]=x_regional_lon_vals

data["y_lat_inds"]=y_lat_inds
data["y_lon_inds"]=y_lon_inds
data["y_regional_nlat"]=y_regional_nlat
data["y_regional_nlon"]=y_regional_nlon
data["y_regional_lat_vals"]=y_regional_lat_vals
data["y_regional_lon_vals"]=y_regional_lon_vals

data["z_lat_inds"]=z_lat_inds
data["z_lon_inds"]=z_lon_inds
data["z_regional_nlat"]=z_regional_nlat
data["z_regional_nlon"]=z_regional_nlon
data["z_regional_lat_vals"]=z_regional_lat_vals
data["z_regional_lon_vals"]=z_regional_lon_vals

data["obs_field_x"]=obs_field_x
data["obs_field_y"]=obs_field_y
data["obs_field_z"]=obs_field_z

data["landsea_data"]=landsea_data

data["model_data_hist_x"]=model_data_hist_x
data["model_data_hist_y"]=model_data_hist_y
data["model_data_hist_z"]=model_data_hist_z

numpy.save("input_data.npy",data)
