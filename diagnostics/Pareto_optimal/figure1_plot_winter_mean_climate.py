# This file is part of the Pareto_optimal diagnosis module of the MDTF code package (see LICENSE.txt)
# ======================================================================
# figure1_plot_winter_mean_climate.py
#
#   Called by Pareto_optimal.py
#   Plot model biases in simulating climatological winter mean patterns in both CMIP6 and Targeting GCMs
#
#
#!/usr/bin/env python
# coding: utf-8

import numpy
from netCDF4 import Dataset
import matplotlib
import matplotlib.pyplot as mp
import matplotlib.colors as mc
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import itertools
import random
import numpy.random
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import scipy as sp
import scipy.stats
import scipy.ndimage
import os

mp.rcParams.update({'mathtext.default': 'regular'})

# load parameters 

para = numpy.load("pareto_parameters.npy",allow_pickle=True)

degree_sign = para[()]['degree_sign']

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

if uwind_level==200:
     umax1=71
     umin1=5
     uinterval1=5
     umax2=9.
     umin2=-9.
     uinterval2=1
if uwind_level==850:
     umax1=16
     umin1=-10
     uinterval1=2
     umax2=5.
     umin2=-5.
     uinterval2=0.5

data = numpy.load("input_data.npy",allow_pickle=True)

x_lat_inds=data[()]["x_lat_inds"]
x_lon_inds=data[()]["x_lon_inds"]
x_regional_nlat=data[()]["x_regional_nlat"]
x_regional_nlon=data[()]["x_regional_nlon"]
x_regional_lat_vals=data[()]["x_regional_lat_vals"]
x_regional_lon_vals=data[()]["x_regional_lon_vals"]

y_lat_inds=data[()]["y_lat_inds"]
y_lon_inds=data[()]["y_lon_inds"]
y_regional_nlat=data[()]["y_regional_nlat"]
y_regional_nlon=data[()]["y_regional_nlon"]
y_regional_lat_vals=data[()]["y_regional_lat_vals"]
y_regional_lon_vals=data[()]["y_regional_lon_vals"]

z_lat_inds=data[()]["z_lat_inds"]
z_lon_inds=data[()]["z_lon_inds"]
z_regional_nlat=data[()]["z_regional_nlat"]
z_regional_nlon=data[()]["z_regional_nlon"]
z_regional_lat_vals=data[()]["z_regional_lat_vals"]
z_regional_lon_vals=data[()]["z_regional_lon_vals"]

obs_field_x=data[()]["obs_field_x"]
obs_field_y=data[()]["obs_field_y"]
obs_field_z=data[()]["obs_field_z"]

landsea_data=data[()]["landsea_data"]

model_data_hist_x=data[()]["model_data_hist_x"]
model_data_hist_y=data[()]["model_data_hist_y"]
model_data_hist_z=data[()]["model_data_hist_z"]

save_dict = numpy.load('pareto_front_results_k1to5.npy',allow_pickle=True)
bias_values_x_target = save_dict[()]['bias_values_x_target']
bias_values_y_target = save_dict[()]['bias_values_y_target']
bias_values_z_target = save_dict[()]['bias_values_z_target']

# ## subset an existing colorbar so that ends aren't as dark
# make color map
minval=0.1 # for inferno:  0.18
maxval=1.0 # for inferno: 1.0
n=256
full_cmap = mp.get_cmap('gray')
cmap_partial_z = matplotlib.colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=full_cmap.name, a=minval, b=maxval), full_cmap(numpy.linspace(minval, maxval, n)))

# make color map
minval=0. # for inferno:  0.18
maxval=0.95 # for inferno: 1.0
n=256
full_cmap = mp.get_cmap('gist_earth_r')
cmap_partial = matplotlib.colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=full_cmap.name, a=minval, b=maxval), full_cmap(numpy.linspace(minval, maxval, n)))

## define a function that forces the middle of a colorbar to be at zero, even when asymmetric max/min
class MidpointNormalize(matplotlib.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return numpy.ma.masked_array(numpy.interp(value, x, y))

fontsize=12
agmt_levels=[6,30]
hatching='..'

############################## one ##############################
#print('one')
mp.rcParams['axes.linewidth'] = 0.3

fig = mp.figure(figsize=(8.25,5))
ax = fig.add_subplot(321, projection=ccrs.PlateCarree())

ax.text(s='a) OBS Precip clim (GPCP; shaded) and Multi-model bias (contours)',x=0.0,y=1.03,ha='left',va='bottom',transform=ax.transAxes,fontsize=fontsize*0.4)
lons,lats = numpy.meshgrid(x_regional_lon_vals, x_regional_lat_vals)
contour_levels = numpy.arange(0,10.,0.25)
if exp_name=='SAM':
    contour_levels = numpy.arange(0,11.,0.3)
if exp_name=='CA':
    ax.add_feature(cfeature.STATES.with_scale('50m'), facecolor='none', edgecolor='grey', linewidths=0.5)
ax.add_feature(cfeature.COASTLINE, facecolor='none', edgecolor='dimgray', linewidths=0.6)
cs=ax.contourf(lons, lats, obs_field_x, levels=contour_levels, extend='max', cmap=cmap_partial, linestyles='none')
cbar = fig.colorbar(cs, ax=ax, shrink=0.7)
cbar.set_label('mm day$^{\,-1}$', fontsize=fontsize*0.4)
cbar.ax.tick_params(labelsize=fontsize*0.4, width=0.3)
cbar.set_ticks([0,1,2,3,4,5,6,7,8,9,10])
if exp_name=='SAM':
    cbar.set_ticks([0,2,4,6,8,10,12])
cbar.solids.set_edgecolor("face")
contour_levels = [-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0.5,1,1.5,2,2.5,3.0,3.5,4.0]
if exp_name=='SAM':
    contour_levels = [-8.,-7.,-6.,-5.,-4.,-3.,-2.,-1.,1.,2.,3.,4.,5.,6.,7.,8.]
mmem_minus_obs = numpy.mean(model_data_hist_x,axis=0)-obs_field_x
ax.contour(lons, lats, mmem_minus_obs, levels=contour_levels, colors='black', linewidths=0.3)
cintvl_x=min(numpy.abs(contour_levels))

for c in cs.collections:
    c.set_edgecolor("face")
lon_wid=x_lon_hi-x_lon_lo
lat_wid=x_lat_hi-x_lat_lo
rec = mpatches.Rectangle(ax.projection.transform_point(x_lon_lo, x_lat_lo,ccrs.PlateCarree()), lon_wid, lat_wid, facecolor="none", edgecolor='black', linewidth=1, linestyle='-',zorder=2)
ax.add_patch(rec)

############################## two ##############################
print('two')
ax = fig.add_subplot(322, projection=ccrs.PlateCarree())
ax.text(s='d) ' + target_model_names + ' Precip clim (shaded) and bias (contours; RMSE: ' + str("{:.1f}".format(bias_values_x_target)) + ' mm day$^{-1}$ )',x=0.0,y=1.03,ha='left',va='bottom',transform=ax.transAxes,fontsize=fontsize*0.4)
lons,lats = numpy.meshgrid(x_regional_lon_vals, x_regional_lat_vals)
contour_levels = numpy.arange(0,10.,0.25)
if exp_name=='SAM':
    contour_levels = numpy.arange(0,11.,0.3)
for i in range(nmods):
      if model_names[i] in [target_model_names]:
          mmem=model_data_hist_x[i,:,:]
if exp_name=='CA':
    ax.add_feature(cfeature.STATES.with_scale('50m'), facecolor='none', edgecolor='grey', linewidths=0.5)
ax.add_feature(cfeature.COASTLINE, facecolor='none', edgecolor='dimgray', linewidths=0.6)
cs=ax.contourf(lons, lats, mmem, levels=contour_levels, extend='max', cmap=cmap_partial, linestyles='none')
cbar = fig.colorbar(cs, ax=ax, shrink=0.7)
cbar.set_label('mm day$^{\,-1}$', fontsize=fontsize*0.4)
cbar.ax.tick_params(labelsize=fontsize*0.4, width=0.3)
cbar.set_ticks([0,1,2,3,4,5,6,7,8,9,10])
if exp_name=='SAM':
    cbar.set_ticks([0,2,4,6,8,10,12])
cbar.solids.set_edgecolor("face")
contour_levels = [-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0.5,1,1.5,2,2.5,3.0,3.5,4.0]
if exp_name=='SAM':
    contour_levels = [-8.,-7.,-6.,-5.,-4.,-3.,-2.,-1.,1.,2.,3.,4.,5.,6.,7.,8.]
mmem_minus_obs = mmem-obs_field_x
ax.contour(lons, lats, mmem_minus_obs, levels=contour_levels, colors='black', linewidths=0.3)

for c in cs.collections:
    c.set_edgecolor("face")

############################## three ##############################
# SSTs/PRW
print('three')
clon=180.
ax = fig.add_subplot(323, projection=ccrs.PlateCarree(central_longitude=clon))
if vlist[1]=='tos':
    ax.text(s='b) OBS SST clim (HadISST; shaded) and Multi-model bias (contours)',x=0.0,y=1.03,ha='left',va='bottom',transform=ax.transAxes,fontsize=fontsize*0.4)
if vlist[1]=='prw':
    ax.text(s='b) OBS PRW clim (ERA-5; shaded) and Multi-model bias (contours)',x=0.0,y=1.03,ha='left',va='bottom',transform=ax.transAxes,fontsize=fontsize*0.4)
masked_sst = obs_field_y
if vlist[1]=='tos':
    masked_sst[landsea_data>1000000]=numpy.nan
lons,lats = numpy.meshgrid(y_regional_lon_vals, y_regional_lat_vals)
if vlist[1]=='tos':
    contour_levels = numpy.arange(10,31,1)
if vlist[1]=='prw':
    contour_levels = numpy.arange(10,55,3)
ax.add_feature(cfeature.COASTLINE, facecolor='none', edgecolor='dimgray', linewidths=0.6)
lons1=lons-clon
cs=ax.contourf(lons1, lats, masked_sst, levels=contour_levels, extend='both', cmap='RdYlBu_r', linestyles='none')
cbar = fig.colorbar(cs, ax=ax, shrink=0.7)
if vlist[1]=='tos':
    cbar.set_label(degree_sign+'C', fontsize=fontsize*0.4)
if vlist[1]=='prw':
    cbar.set_label('mm', fontsize=fontsize*0.4)
cbar.ax.tick_params(labelsize=fontsize*0.4, width=0.3)
cbar.solids.set_edgecolor("face")
mmem_minus_obs = (numpy.mean(model_data_hist_y,axis=0))-obs_field_y
if vlist[1]=='tos':
    contour_levels = numpy.hstack((numpy.arange(-2.50,-0.24,0.25),numpy.arange(0.25,2.5,0.25)))
    mmem_minus_obs[landsea_data>1000000]=numpy.nan
if vlist[1]=='prw':
    contour_levels = numpy.hstack((numpy.arange(-13.,-0.99,1.0),numpy.arange(1.0,13.0,1.0)))
ax.contour(lons1, lats, mmem_minus_obs, levels=contour_levels, colors='black', linewidths=0.3, linestyles=['--']*sum(contour_levels<0)+['-']*sum(contour_levels>0))
cintvl_y=min(numpy.abs(contour_levels))

for c in cs.collections:
    c.set_edgecolor("face")
lon_wid=y_lon_hi-y_lon_lo
lat_wid=y_lat_hi-y_lat_lo
rec = mpatches.Rectangle(ax.projection.transform_point(y_lon_lo, y_lat_lo,ccrs.PlateCarree()), lon_wid, lat_wid, facecolor="none", edgecolor='black', linewidth=1, linestyle='-',zorder=2)
ax.add_patch(rec)

############################## four ##############################
print('four')
clon=180.
ax = fig.add_subplot(324, projection=ccrs.PlateCarree(central_longitude=clon))
if vlist[1]=='tos':
    ax.text(s='e) ' + target_model_names + ' SST clim (shaded) and bias (contours; RMSE: ' + str("{:.1f}".format(bias_values_y_target)) + degree_sign +'C )',x=0.0,y=1.03,ha='left',va='bottom',transform=ax.transAxes,fontsize=fontsize*0.4)
if vlist[1]=='prw':
    ax.text(s='e) ' + target_model_names + ' PRW clim (shaded) and bias (contours; RMSE: ' + str("{:.1f}".format(bias_values_y_target)) + ' mm )',x=0.0,y=1.03,ha='left',va='bottom',transform=ax.transAxes,fontsize=fontsize*0.4)
for i in range(nmods):
      if model_names[i] in [target_model_names]:
          masked_sst=model_data_hist_y[i,:,:]
if vlist[1]=='tos':
    masked_sst[landsea_data>1000000]=numpy.nan
lons,lats = numpy.meshgrid(y_regional_lon_vals, y_regional_lat_vals)
if vlist[1]=='tos':
    contour_levels = numpy.arange(10,31,1)
if vlist[1]=='prw':
    contour_levels = numpy.arange(10,55,3)
ax.add_feature(cfeature.COASTLINE, facecolor='none', edgecolor='dimgray', linewidths=0.6)
lons1=lons-clon
cs=ax.contourf(lons1, lats, masked_sst, levels=contour_levels, extend='both', cmap='RdYlBu_r', linestyles='none')
cbar = fig.colorbar(cs, ax=ax, shrink=0.7)
if vlist[1]=='tos':
    cbar.set_label(degree_sign+'C', fontsize=fontsize*0.4)
if vlist[1]=='prw':
    cbar.set_label('mm', fontsize=fontsize*0.4)
cbar.ax.tick_params(labelsize=fontsize*0.4, width=0.3)
cbar.solids.set_edgecolor("face")
mmem_minus_obs = masked_sst-obs_field_y
if vlist[1]=='tos':
    contour_levels = numpy.hstack((numpy.arange(-2.50,-0.24,0.25),numpy.arange(0.25,2.5,0.25)))
    mmem_minus_obs[landsea_data>1000000]=numpy.nan
if vlist[1]=='prw':
    contour_levels = numpy.hstack((numpy.arange(-13.,-1.0,1.0),numpy.arange(1.0,13.0,1.0)))
ax.contour(lons1, lats, mmem_minus_obs, levels=contour_levels, colors='black', linewidths=0.3, linestyles=['--']*sum(contour_levels<0)+['-']*sum(contour_levels>0))
for c in cs.collections:
    c.set_edgecolor("face")

# WINDS
############################## five ##############################
print('five')
clon=180.
if exp_name=='SAM':
    clon=310.
ax = fig.add_subplot(325, projection=ccrs.PlateCarree(central_longitude=clon))
ax.text(s='c) OBS U'+str(uwind_level)+' clim (ERA-5; shaded) and Multi-model bias (contours)',x=0.0,y=1.03,ha='left',va='bottom',transform=ax.transAxes,fontsize=fontsize*0.4)
lons,lats = numpy.meshgrid(z_regional_lon_vals, z_regional_lat_vals)
mmem = obs_field_z
mmem[numpy.abs(mmem)>100]=numpy.nan
contour_levels = numpy.arange(umin1,umax1,uinterval1)
ax.add_feature(cfeature.COASTLINE, facecolor='none', edgecolor='dimgray', linewidths=0.6)
lons1=lons-clon
cs=ax.contourf(lons1, lats, mmem, levels=contour_levels, extend='both', cmap='RdYlBu_r', linestyles='none')
cbar = fig.colorbar(cs, ax=ax, shrink=0.7)
cbar.set_label('m s$^{-1}$', fontsize=fontsize*0.4)
cbar.ax.tick_params(labelsize=fontsize*0.4, width=0.3)
cbar.solids.set_edgecolor("face")
mmem_minus_obs = numpy.mean(model_data_hist_z,axis=0)-obs_field_z
contour_levels = numpy.hstack((numpy.arange(umin2,-uinterval2,uinterval2),numpy.arange(uinterval2,umax2,uinterval2)))
ax.contour(lons1, lats, mmem_minus_obs, levels=contour_levels, colors='black', linewidths=0.3, linestyles=['--']*sum(contour_levels<0)+['-']*sum(contour_levels>0))
cintvl_z=min(numpy.abs(contour_levels))

for c in cs.collections:
    c.set_edgecolor("face")
lon_wid=z_lon_hi-z_lon_lo
lat_wid=z_lat_hi-z_lat_lo
rec = mpatches.Rectangle(ax.projection.transform_point(z_lon_lo, z_lat_lo,ccrs.PlateCarree()), lon_wid, lat_wid, facecolor="none", edgecolor='black', linewidth=1, linestyle='-',zorder=2)
ax.add_patch(rec)

############################## six ##############################
print('six')
clon=180.
if exp_name=='SAM':
    clon=310.
ax = fig.add_subplot(326, projection=ccrs.PlateCarree(central_longitude=clon))
ax.text(s='f) ' + target_model_names +' U'+str(uwind_level)+' clim (shaded) and bias (contours; RMSE: ' + str("{:.1f}".format(bias_values_z_target)) + ' m s$^{-1}$ )',x=0.0,y=1.03,ha='left',va='bottom',transform=ax.transAxes,fontsize=fontsize*0.4)
lons,lats = numpy.meshgrid(z_regional_lon_vals, z_regional_lat_vals)
for i in range(nmods):
      if model_names[i] in [target_model_names]:
          mmem=model_data_hist_z[i,:,:]
mmem[numpy.abs(mmem)>100]=numpy.nan
contour_levels = numpy.arange(umin1,umax1,uinterval1)
ax.add_feature(cfeature.COASTLINE, facecolor='none', edgecolor='dimgray', linewidths=0.6)
lons1=lons-clon
cs=ax.contourf(lons1, lats, mmem, levels=contour_levels, extend='both', cmap='RdYlBu_r', linestyles='none')
cbar = fig.colorbar(cs, ax=ax, shrink=0.7)
cbar.set_label('m s$^{-1}$', fontsize=fontsize*0.4)
cbar.ax.tick_params(labelsize=fontsize*0.4, width=0.3)
cbar.solids.set_edgecolor("face")
contour_levels = numpy.hstack((numpy.arange(umin2,-uinterval2,uinterval2),numpy.arange(uinterval2,umax2,uinterval2)))
mmem_minus_obs = mmem-obs_field_z
ax.contour(lons1, lats, mmem_minus_obs, levels=contour_levels, colors='black', linewidths=0.3, linestyles=['--']*sum(contour_levels<0)+['-']*sum(contour_levels>0))
for c in cs.collections:
    c.set_edgecolor("face")

ax.text(s='All fields are for boreal winter or austral summer seasonal mean (Dec-Jan).',x=-1.50,y=-0.15,ha='left',va='bottom',color='dimgray',transform=ax.transAxes,fontsize=fontsize*0.35)
if vlist[1]=='tos':
   ax.text(s='Contours (zero-lines omitted) with intervals of '+str("{:.1f}".format(cintvl_x))+' mm day$^{-1}$ for Precip, '+str("{:.1f}".format(cintvl_y))+degree_sign+'C for SST, and '+ str("{:.1f}".format(cintvl_z))+' m s$^{-1}$ for U'+str(uwind_level)+', respectively.',x=-1.50,y=-0.22,ha='left',va='bottom',color='dimgray',transform=ax.transAxes,fontsize=fontsize*0.35)
if vlist[1]=='prw':
   ax.text(s='Contours (zero-lines omitted) with intervals of '+str("{:.1f}".format(cintvl_x))+' mm day$^{-1}$ for Precip, '+str("{:.1f}".format(cintvl_y))+' mm for PRW, and '+ str("{:.1f}".format(cintvl_z))+' m s$^{-1}$ for U'+str(uwind_level)+', respectively.',x=-1.50,y=-0.22,ha='left',va='bottom',color='dimgray',transform=ax.transAxes,fontsize=fontsize*0.35)
ax.text(s='Multi-model simulations are from the CMIP6 project.',x=-1.50,y=-0.29,ha='left',va='bottom',color='dimgray',transform=ax.transAxes,fontsize=fontsize*0.35)
ax.text(s='Rectanglar boxes represent analysis regions for the three values.',x=-1.50,y=-0.36,ha='left',va='bottom',color='dimgray',transform=ax.transAxes,fontsize=fontsize*0.35)

fig.savefig(os.environ["WK_DIR"]+"/model/PS/"+'spatial_pattern_multi_member_mean_fields.pdf', transparent=True, bbox_inches='tight', dpi=1200)

mp.show()
