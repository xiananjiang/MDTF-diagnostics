# This file is part of the Pareto_optimal diagnosis module of the MDTF code package (see LICENSE.txt)
# ======================================================================
# figure3_optimal_winter_mean_patterns.py
#
#   Called by Pareto_optimal.py
#   Plot winter mean patterns associated with optimal model sets for different variables
#
#
#!/usr/bin/env python
# coding: utf-8

import numpy
import matplotlib
import matplotlib.pyplot as mp
import matplotlib.colors as mc
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

# Read parameters
para = numpy.load("pareto_parameters.npy",allow_pickle=True)

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

season=para[()]["season"]

pareto_k_values=para[()]["pareto_k_values"]

exp_name=para[()]["exp_name"]
vlist=para[()]["vlist"]
vlist=para[()]["vlist"]
vlist_label=para[()]["vlist_label"]
vlist_unit=para[()]["vlist_unit"]

save_dir = './'
save_filename = 'pareto_front_results.npy'
save_dict = numpy.load(save_dir+save_filename,allow_pickle=True)

model_names = save_dict[()]['model_names']
nmods = len(model_names)

pareto_set_collect_2d_list = save_dict[()]['pareto_set_collect_2d_list']
pareto_set_collect_3d_list = save_dict[()]['pareto_set_collect_3d_list']

set_indices_collect_2d_list = save_dict[()]['set_indices_collect_2d_list']
set_indices_collect_3d_list = save_dict[()]['set_indices_collect_3d_list']

bias_values_subensembles_x = save_dict[()]['bias_values_subensembles_x']
bias_values_subensembles_y = save_dict[()]['bias_values_subensembles_y']
bias_values_subensembles_z = save_dict[()]['bias_values_subensembles_z']

model_combinations = save_dict[()]['model_combinations']
pareto_set_sizes_3d = save_dict[()]['pareto_set_sizes_3d']

pareto_set_collect = pareto_set_collect_3d_list[0]
set_indices_collect = set_indices_collect_3d_list[0]
length_pareto=len(pareto_set_collect)

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

model_data_hist_x0=data[()]["model_data_hist_x"]
model_data_hist_y0=data[()]["model_data_hist_y"]
model_data_hist_z0=data[()]["model_data_hist_z"]

# create dictionaries to be used below
dict_x = {
'nlat':x_regional_nlat,
'nlon':x_regional_nlon,
'obs_field':obs_field_x
}

dict_y = {
'nlat':y_regional_nlat,
'nlon':y_regional_nlon,
'obs_field':obs_field_y
}

dict_z = {
'LENS':False,
'nlat':z_regional_nlat,
'nlon':z_regional_nlon,
'obs_field':obs_field_z
}

## Paremeters to customize Figure 3.
if exp_name=='CA':
    contour_levels_xs = numpy.arange(0,12.,0.4)
    contour_levels_xc = numpy.hstack((numpy.arange(-4.,-0.5,0.5),numpy.arange(0.5,4.0,0.5)))
    contour_levels_ys = numpy.arange(10,31,1)
    contour_levels_yc = numpy.hstack((numpy.arange(-1.25,-0.24,0.25),numpy.arange(0.25,1.3,0.25)))
    contour_levels_zs = numpy.arange(5,71,5)
    contour_levels_zc = numpy.hstack((numpy.arange(-9.,-1.,1.),numpy.arange(1.,9.,1.)))
    cbar_ticks_xs=[0,1,2,3,4,5,6,7,8,9,10]
    plot_states=1
    clon_y=180.
    clon_z=180.
if exp_name=='SAM':
    contour_levels_xs = numpy.arange(0,11.,0.3)
    contour_levels_xc = numpy.hstack((numpy.arange(-8.,-1.,1.0),numpy.arange(1.,8.,1.0)))
    contour_levels_ys = numpy.arange(10,31,1)
    contour_levels_yc = numpy.hstack((numpy.arange(-1.25,-0.24,0.25),numpy.arange(0.25,1.3,0.25)))
    contour_levels_zs = numpy.arange(-10,16,2)
    contour_levels_zc = numpy.hstack((numpy.arange(-5.,-0.5,0.5),numpy.arange(0.5,5.,0.5)))
    cbar_ticks_xs=[0,2,4,6,8,10,12]
    plot_states=0
    clon_y=180.
    clon_z=310.

# make color map
minval=0.1 
maxval=1.0 
n=256
full_cmap = mp.get_cmap('gray')
cmap_partial_z = matplotlib.colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=full_cmap.name, a=minval, b=maxval), full_cmap(numpy.linspace(minval, maxval, n)))

# make color map
minval=0. # for inferno:  0.18
maxval=0.95 # for inferno: 1.0
n=256
full_cmap = mp.get_cmap('gist_earth_r')
cmap_partial = matplotlib.colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=full_cmap.name, a=minval, b=maxval), full_cmap(numpy.linspace(minval, maxval, n)))

# In[18]:
fontsize=12
agmt_levels=[6,30]
hatching='..'

fig = mp.figure(figsize=(8,7))

##### Model subset with minimum RMSE in variable_x
min_para=10000000.
for ipareto in range(length_pareto):
    if min_para >= pareto_set_collect[ipareto,0]: 
           ipar=ipareto
           min_para=pareto_set_collect[ipareto,0]

modelcomb=model_combinations[int(set_indices_collect[ipar])]

npmodels=len(modelcomb)
models=model_names[modelcomb[0]]
for ipmod in range(len(modelcomb)-1):
    models=models+', '+model_names[modelcomb[ipmod+1]]
print(models)

# set up data
model_data_hist_x = numpy.zeros((npmodels, x_regional_nlat, x_regional_nlon))
model_data_hist_y = numpy.zeros((npmodels, y_regional_nlat, y_regional_nlon))
model_data_hist_z = numpy.zeros((npmodels, z_regional_nlat, z_regional_nlon))

for i in range(npmodels):
    modelname = model_names[modelcomb[i]]
    model_data_hist_x[i,:,:] = model_data_hist_x0[modelcomb[i],:,:]
    model_data_hist_y[i,:,:] = model_data_hist_y0[modelcomb[i],:,:]
    model_data_hist_z[i,:,:] = model_data_hist_z0[modelcomb[i],:,:]

mp.rcParams['axes.linewidth'] = 0.3
#variable_x
ax = mp.subplot2grid((32,28),(3,0),colspan=9,rowspan=6, projection=ccrs.PlateCarree())
ax.text(s='Pareto-optimal set with minimum spatial RMSEs for '+vlist_label[0]+', '+vlist_label[1]+', and '+vlist_label[2],x=0.0,y=1.80,ha='left',va='bottom',transform=ax.transAxes,fontsize=fontsize*0.6)
ax.text(s='Min. '+vlist_label[0]+' RMSEs',x=0.5,y=1.55,ha='center',va='bottom',transform=ax.transAxes,fontsize=fontsize*0.45)
ax.text(s='('+models+')',x=0.5,y=1.45,ha='center',va='bottom',color='blue',transform=ax.transAxes,fontsize=fontsize*0.35)
ax.text(s=vlist_label[0]+' climatology (shaded) and bias (contours; relative to GPCP) in each model subset',x=0.8,y=1.15,ha='left',va='bottom',transform=ax.transAxes,fontsize=fontsize*0.5)
ax.text(s='RMSE: '+ "{:.2f}".format(pareto_set_collect[ipar,0])+' '+vlist_unit[0],x=0.63,y=0.99,ha='left',va='bottom',transform=ax.transAxes,fontsize=fontsize*0.3)
lons,lats = numpy.meshgrid(x_regional_lon_vals, x_regional_lat_vals)

if plot_states==1:
    ax.add_feature(cfeature.STATES.with_scale('50m'), facecolor='none', edgecolor='grey', linewidths=0.5)
ax.add_feature(cfeature.COASTLINE, facecolor='none', edgecolor='dimgray', linewidths=0.6)
contour_levels = contour_levels_xs
cs=ax.contourf(lons, lats, numpy.mean(model_data_hist_x,axis=0), levels=contour_levels, extend='max', cmap=cmap_partial, linestyles='none')
cbar = fig.colorbar(cs, ax=ax, shrink=0.6)
cbar.set_label(vlist_unit[0], fontsize=fontsize*0.3)
cbar.ax.tick_params(width=0.3,length=2.0,labelsize=fontsize*0.3)
cbar.set_ticks(cbar_ticks_xs)
cbar.solids.set_edgecolor("face")
cbar.outline.set_linewidth(0.3)
mmem_minus_obs = numpy.mean(model_data_hist_x,axis=0)-dict_x['obs_field']
contour_levels = contour_levels_xc
ax.contour(lons, lats, mmem_minus_obs, levels=contour_levels, colors='black', linewidths=0.3, linestyles=['--']*sum(contour_levels<0)+['-']*sum(contour_levels>0))
ax.text(s='Dashed (solid) contours for negative (positive) values with the zero-line omitted and intervals of '+ str("{:.1f}".format(min(numpy.abs(contour_levels))))+' '+vlist_unit[0],x=0.0,y=-0.15,ha='left',va='bottom',transform=ax.transAxes,fontsize=fontsize*0.35)

for c in cs.collections:
    c.set_edgecolor("face")
lon_wid=x_lon_hi-x_lon_lo
lat_wid=x_lat_hi-x_lat_lo
rec = mpatches.Rectangle(ax.projection.transform_point(x_lon_lo, x_lat_lo,ccrs.PlateCarree()), lon_wid, lat_wid, facecolor="none", edgecolor='black', linewidth=1, linestyle='-',zorder=2)
ax.add_patch(rec)

# variable_y
ax = mp.subplot2grid((32,28),(10,0),colspan=9,rowspan=6, projection=ccrs.PlateCarree(central_longitude=clon_y))
ax.text(s=vlist_label[1]+' climatology (shaded) and biases (contours; relative to HADISST) in model subsets',x=0.8,y=1.15,ha='left',va='bottom',transform=ax.transAxes,fontsize=fontsize*0.5)
ax.text(s='RMSE: '+ "{:.2f}".format(pareto_set_collect[ipar,1])+vlist_unit[1],x=0.72,y=1.005,ha='left',va='bottom',transform=ax.transAxes,fontsize=fontsize*0.3)
masked_sst = numpy.mean(model_data_hist_y,axis=0)
if vlist[1]=='tos':
   masked_sst[landsea_data>1000000]=numpy.nan
lons,lats = numpy.meshgrid(y_regional_lon_vals, y_regional_lat_vals)
ax.add_feature(cfeature.COASTLINE, facecolor='none', edgecolor='dimgray', linewidths=0.6)
lons1=lons-clon_y
contour_levels = contour_levels_ys
cs=ax.contourf(lons1, lats, masked_sst, levels=contour_levels, extend='both', cmap='RdYlBu_r', linestyles='none')
cbar = fig.colorbar(cs, ax=ax, shrink=0.6)
cbar.set_label(vlist_unit[1], fontsize=fontsize*0.3)
cbar.ax.tick_params(width=0.3,length=2.0,labelsize=fontsize*0.3)
cbar.solids.set_edgecolor("face")
mmem_minus_obs = (numpy.mean(model_data_hist_y,axis=0))-dict_y['obs_field']
if vlist[1]=='tos':
   mmem_minus_obs[landsea_data>1000000]=numpy.nan
contour_levels = contour_levels_yc
ax.contour(lons1, lats, mmem_minus_obs, levels=contour_levels, colors='black', linewidths=0.3, linestyles=['--']*sum(contour_levels<0)+['-']*sum(contour_levels>0))
ax.text(s='Dashed (solid) contours for negative (positive) values with the zero-line omitted and intervals of '+ str("{:.2f}".format(min(numpy.abs(contour_levels))))+vlist_unit[1],x=0.0,y=-0.15,ha='left',va='bottom',transform=ax.transAxes,fontsize=fontsize*0.35)

for c in cs.collections:
    c.set_edgecolor("face")
lon_wid=y_lon_hi-y_lon_lo
lat_wid=y_lat_hi-y_lat_lo
rec = mpatches.Rectangle(ax.projection.transform_point(y_lon_lo, y_lat_lo,ccrs.PlateCarree()), lon_wid, lat_wid, facecolor="none", edgecolor='black', linewidth=1, linestyle='-',zorder=2)
ax.add_patch(rec)

# variable_z
ax = mp.subplot2grid((32,28),(17,0),colspan=9,rowspan=6, projection=ccrs.PlateCarree(central_longitude=clon_z))
ax.text(s=vlist_label[2]+' climatology (shaded) and bias (contours; relative to ERA-5) in model subsets',x=0.8,y=1.15,ha='left',va='bottom',transform=ax.transAxes,fontsize=fontsize*0.5)
ax.text(s='RMSE: '+ "{:.2f}".format(pareto_set_collect[ipar,2])+' '+vlist_unit[2],x=0.70,y=0.99,ha='left',va='bottom',transform=ax.transAxes,fontsize=fontsize*0.3)
lons,lats = numpy.meshgrid(z_regional_lon_vals, z_regional_lat_vals)
mmem = numpy.mean(model_data_hist_z,axis=0)
ax.add_feature(cfeature.COASTLINE, facecolor='none', edgecolor='dimgray', linewidths=0.6)
lons1=lons-clon_z
contour_levels = contour_levels_zs
cs=ax.contourf(lons1, lats, mmem, levels=contour_levels, extend='both', cmap='RdYlBu_r', linestyles='none')
cbar = fig.colorbar(cs, ax=ax, shrink=0.6)
cbar.set_label(vlist_unit[2], fontsize=fontsize*0.3)
cbar.ax.tick_params(width=0.3,length=2.0,labelsize=fontsize*0.3)
cbar.solids.set_edgecolor("face")
mmem_minus_obs = numpy.mean(model_data_hist_z,axis=0)-dict_z['obs_field']
contour_levels = contour_levels_zc
ax.contour(lons1, lats, mmem_minus_obs, levels=contour_levels, colors='black', linewidths=0.3, linestyles=['--']*sum(contour_levels<0)+['-']*sum(contour_levels>0))
ax.text(s='Dashed (solid) contours for negative (positive) values with the zero-line omitted and intervals of '+str("{:.1f}".format(min(numpy.abs(contour_levels))))+' '+vlist_unit[2],x=0.0,y=-0.15,ha='left',va='bottom',transform=ax.transAxes,fontsize=fontsize*0.35)

for c in cs.collections:
    c.set_edgecolor("face")
lon_wid=z_lon_hi-z_lon_lo
lat_wid=z_lat_hi-z_lat_lo
rec = mpatches.Rectangle(ax.projection.transform_point(z_lon_lo, z_lat_lo,ccrs.PlateCarree()), lon_wid, lat_wid, facecolor="none", edgecolor='black', linewidth=1, linestyle='-',zorder=2)
ax.add_patch(rec)

##### Model subset with minimum RMSE in variable_y
min_para=10000000.
for ipareto in range(length_pareto):
    if min_para >= pareto_set_collect[ipareto,1]: 
           ipar=ipareto
           min_para=pareto_set_collect[ipareto,1]

modelcomb=model_combinations[int(set_indices_collect[ipar])]

npmodels=len(modelcomb)
models=model_names[modelcomb[0]]
for ipmod in range(len(modelcomb)-1):
    models=models+', '+model_names[modelcomb[ipmod+1]]
print(models)

# set up data
model_data_hist_x = numpy.zeros((npmodels, x_regional_nlat, x_regional_nlon))
model_data_hist_y = numpy.zeros((npmodels, y_regional_nlat, y_regional_nlon))
model_data_hist_z = numpy.zeros((npmodels, z_regional_nlat, z_regional_nlon))

for i in range(npmodels):
    modelname = model_names[modelcomb[i]]
    model_data_hist_x[i,:,:] = model_data_hist_x0[modelcomb[i],:,:]
    model_data_hist_y[i,:,:] = model_data_hist_y0[modelcomb[i],:,:]
    model_data_hist_z[i,:,:] = model_data_hist_z0[modelcomb[i],:,:]

ax = mp.subplot2grid((32,28),(3,9),colspan=9,rowspan=6, projection=ccrs.PlateCarree())
ax.text(s='Min. '+vlist_label[1]+' RMSEs',x=0.5,y=1.55,ha='center',va='bottom',transform=ax.transAxes,fontsize=fontsize*0.45)
ax.text(s='('+models+')',x=0.5,y=1.45,ha='center',va='bottom',color='blue',transform=ax.transAxes,fontsize=fontsize*0.35)
ax.text(s='RMSE: '+ "{:.2f}".format(pareto_set_collect[ipar,0])+' '+vlist_unit[0],x=0.63,y=0.99,ha='left',va='bottom',transform=ax.transAxes,fontsize=fontsize*0.3)
lons,lats = numpy.meshgrid(x_regional_lon_vals, x_regional_lat_vals)
if plot_states==1:
    ax.add_feature(cfeature.STATES.with_scale('50m'), facecolor='none', edgecolor='grey', linewidths=0.5)
ax.add_feature(cfeature.COASTLINE, facecolor='none', edgecolor='dimgray', linewidths=0.6)
contour_levels = contour_levels_xs
cs=ax.contourf(lons, lats, numpy.mean(model_data_hist_x,axis=0), levels=contour_levels, extend='max', cmap=cmap_partial, linestyles='none')
cbar = fig.colorbar(cs, ax=ax, shrink=0.6)
cbar.set_label(vlist_unit[0], fontsize=fontsize*0.3)
cbar.ax.tick_params(width=0.3,length=2.0,labelsize=fontsize*0.3)
cbar.set_ticks(cbar_ticks_xs)
cbar.solids.set_edgecolor("face")
cbar.outline.set_linewidth(0.3)
contour_levels = contour_levels_xc
mmem_minus_obs = numpy.mean(model_data_hist_x,axis=0)-dict_x['obs_field']
ax.contour(lons, lats, mmem_minus_obs, levels=contour_levels, colors='black', linewidths=0.3, linestyles=['--']*sum(contour_levels<0)+['-']*sum(contour_levels>0))

for c in cs.collections:
    c.set_edgecolor("face")
lon_wid=x_lon_hi-x_lon_lo
lat_wid=x_lat_hi-x_lat_lo
rec = mpatches.Rectangle(ax.projection.transform_point(x_lon_lo, x_lat_lo,ccrs.PlateCarree()), lon_wid, lat_wid, facecolor="none", edgecolor='black', linewidth=1, linestyle='-',zorder=2)
ax.add_patch(rec)

# variable_y
ax = mp.subplot2grid((32,28),(10,9),colspan=9,rowspan=6, projection=ccrs.PlateCarree(central_longitude=clon_y))
ax.text(s='RMSE: '+ "{:.2f}".format(pareto_set_collect[ipar,1])+vlist_unit[1],x=0.72,y=1.005,ha='left',va='bottom',transform=ax.transAxes,fontsize=fontsize*0.3)
masked_sst = numpy.mean(model_data_hist_y,axis=0)
if vlist[1]=='tos':
   masked_sst[landsea_data>1000000]=numpy.nan
lons,lats = numpy.meshgrid(y_regional_lon_vals, y_regional_lat_vals)
ax.add_feature(cfeature.COASTLINE, facecolor='none', edgecolor='dimgray', linewidths=0.6)
lons1=lons-clon_y
contour_levels = contour_levels_ys
cs=ax.contourf(lons1, lats, masked_sst, levels=contour_levels, extend='both', cmap='RdYlBu_r', linestyles='none')
cbar = fig.colorbar(cs, ax=ax, shrink=0.6)
cbar.set_label(vlist_unit[1], fontsize=fontsize*0.3)
cbar.ax.tick_params(width=0.3,length=2.0,labelsize=fontsize*0.3)
cbar.solids.set_edgecolor("face")
mmem_minus_obs = (numpy.mean(model_data_hist_y,axis=0))-dict_y['obs_field']
if vlist[1]=='tos':
   mmem_minus_obs[landsea_data>1000000]=numpy.nan
contour_levels = contour_levels_yc
ax.contour(lons1, lats, mmem_minus_obs, levels=contour_levels, colors='black', linewidths=0.3, linestyles=['--']*sum(contour_levels<0)+['-']*sum(contour_levels>0))

for c in cs.collections:
    c.set_edgecolor("face")
lon_wid=y_lon_hi-y_lon_lo
lat_wid=y_lat_hi-y_lat_lo
rec = mpatches.Rectangle(ax.projection.transform_point(y_lon_lo, y_lat_lo,ccrs.PlateCarree()), lon_wid, lat_wid, facecolor="none", edgecolor='black', linewidth=1, linestyle='-',zorder=2)
ax.add_patch(rec)

# variable_z
ax = mp.subplot2grid((32,28),(17,9),colspan=9,rowspan=6, projection=ccrs.PlateCarree(central_longitude=clon_z))
ax.text(s='RMSE: '+ "{:.2f}".format(pareto_set_collect[ipar,2])+' '+vlist_unit[2],x=0.70,y=0.99,ha='left',va='bottom',transform=ax.transAxes,fontsize=fontsize*0.3)
lons,lats = numpy.meshgrid(z_regional_lon_vals, z_regional_lat_vals)
ax.add_feature(cfeature.COASTLINE, facecolor='none', edgecolor='dimgray', linewidths=0.6)
lons1=lons-clon_z
mmem = numpy.mean(model_data_hist_z,axis=0)
contour_levels = contour_levels_zs
cs=ax.contourf(lons1, lats, mmem, levels=contour_levels, extend='both', cmap='RdYlBu_r', linestyles='none')
cbar = fig.colorbar(cs, ax=ax, shrink=0.6)
cbar.set_label(vlist_unit[2], fontsize=fontsize*0.3)
cbar.ax.tick_params(width=0.3,length=2.0,labelsize=fontsize*0.3)
cbar.solids.set_edgecolor("face")
contour_levels = contour_levels_zc
mmem_minus_obs = numpy.mean(model_data_hist_z,axis=0)-dict_z['obs_field']
ax.contour(lons1, lats, mmem_minus_obs, levels=contour_levels, colors='black', linewidths=0.3, linestyles=['--']*sum(contour_levels<0)+['-']*sum(contour_levels>0))

for c in cs.collections:
    c.set_edgecolor("face")
lon_wid=z_lon_hi-z_lon_lo
lat_wid=z_lat_hi-z_lat_lo
rec = mpatches.Rectangle(ax.projection.transform_point(z_lon_lo, z_lat_lo,ccrs.PlateCarree()), lon_wid, lat_wid, facecolor="none", edgecolor='black', linewidth=1, linestyle='-',zorder=2)
ax.add_patch(rec)

## Model subset with minimum RMSE in variable_z
min_para=10000000.
for ipareto in range(length_pareto):
    if min_para >= pareto_set_collect[ipareto,2]: 
           ipar=ipareto
           min_para=pareto_set_collect[ipareto,2]

modelcomb=model_combinations[int(set_indices_collect[ipar])]

npmodels=len(modelcomb)
models=model_names[modelcomb[0]]
for ipmod in range(len(modelcomb)-1):
    models=models+', '+model_names[modelcomb[ipmod+1]]
print(models)

# set up data
model_data_hist_x = numpy.zeros((npmodels, x_regional_nlat, x_regional_nlon))
model_data_hist_y = numpy.zeros((npmodels, y_regional_nlat, y_regional_nlon))
model_data_hist_z = numpy.zeros((npmodels, z_regional_nlat, z_regional_nlon))

for i in range(npmodels):
    modelname = model_names[modelcomb[i]]
    model_data_hist_x[i,:,:] = model_data_hist_x0[modelcomb[i],:,:]
    model_data_hist_y[i,:,:] = model_data_hist_y0[modelcomb[i],:,:]
    model_data_hist_z[i,:,:] = model_data_hist_z0[modelcomb[i],:,:]

# variable_x
ax = mp.subplot2grid((32,28),(3,18),colspan=9,rowspan=6, projection=ccrs.PlateCarree())
ax.text(s='Min. '+vlist_label[2]+' RMSEs',x=0.5,y=1.55,ha='center',va='bottom',transform=ax.transAxes,fontsize=fontsize*0.45)
ax.text(s='('+models+')',x=0.5,y=1.45,ha='center',va='bottom',color='blue',transform=ax.transAxes,fontsize=fontsize*0.35)
ax.text(s='RMSE: '+ "{:.2f}".format(pareto_set_collect[ipar,0])+' '+vlist_unit[0],x=0.63,y=0.99,ha='left',va='bottom',transform=ax.transAxes,fontsize=fontsize*0.3)
lons,lats = numpy.meshgrid(x_regional_lon_vals, x_regional_lat_vals)
if plot_states==1:
    ax.add_feature(cfeature.STATES.with_scale('50m'), facecolor='none', edgecolor='grey', linewidths=0.5)
ax.add_feature(cfeature.COASTLINE, facecolor='none', edgecolor='dimgray', linewidths=0.6)
contour_levels = contour_levels_xs
cs=ax.contourf(lons, lats, numpy.mean(model_data_hist_x,axis=0), levels=contour_levels, extend='max', cmap=cmap_partial, linestyles='none')
cbar = fig.colorbar(cs, ax=ax, shrink=0.6)
cbar.set_label(vlist_unit[0], fontsize=fontsize*0.3)
cbar.ax.tick_params(width=0.3,length=2.0,labelsize=fontsize*0.3)
cbar.set_ticks(cbar_ticks_xs)
cbar.solids.set_edgecolor("face")
cbar.outline.set_linewidth(0.3)
mmem_minus_obs = numpy.mean(model_data_hist_x,axis=0)-dict_x['obs_field']
contour_levels = contour_levels_xc
ax.contour(lons, lats, mmem_minus_obs, levels=contour_levels, colors='black', linewidths=0.3, linestyles=['--']*sum(contour_levels<0)+['-']*sum(contour_levels>0))

for c in cs.collections:
    c.set_edgecolor("face")
lon_wid=x_lon_hi-x_lon_lo
lat_wid=x_lat_hi-x_lat_lo
rec = mpatches.Rectangle(ax.projection.transform_point(x_lon_lo, x_lat_lo,ccrs.PlateCarree()), lon_wid, lat_wid, facecolor="none", edgecolor='black', linewidth=1, linestyle='-',zorder=2)
ax.add_patch(rec)

# variable_y
ax = mp.subplot2grid((32,28),(10,18),colspan=9,rowspan=6, projection=ccrs.PlateCarree(central_longitude=clon_y))
ax.text(s='RMSE: '+ "{:.2f}".format(pareto_set_collect[ipar,1])+vlist_unit[1],x=0.72,y=1.005,ha='left',va='bottom',transform=ax.transAxes,fontsize=fontsize*0.3)
masked_sst = numpy.mean(model_data_hist_y,axis=0)
if vlist[1]=='tos':
   masked_sst[landsea_data>1000000]=numpy.nan
lons,lats = numpy.meshgrid(y_regional_lon_vals, y_regional_lat_vals)
ax.add_feature(cfeature.COASTLINE, facecolor='none', edgecolor='dimgray', linewidths=0.6)
lons1=lons-clon_y
contour_levels = contour_levels_ys
cs=ax.contourf(lons1, lats, masked_sst, levels=contour_levels, extend='both', cmap='RdYlBu_r', linestyles='none')
cbar = fig.colorbar(cs, ax=ax, shrink=0.6)
cbar.set_label(vlist_unit[1], fontsize=fontsize*0.3)
cbar.ax.tick_params(width=0.3,length=2.0,labelsize=fontsize*0.3)
cbar.solids.set_edgecolor("face")
cbar.outline.set_linewidth(0.3)
mmem_minus_obs = (numpy.mean(model_data_hist_y,axis=0))-dict_y['obs_field']
if vlist[1]=='tos':
   mmem_minus_obs[landsea_data>1000000]=numpy.nan
contour_levels = contour_levels_yc
ax.contour(lons1, lats, mmem_minus_obs, levels=contour_levels, colors='black', linewidths=0.3, linestyles=['--']*sum(contour_levels<0)+['-']*sum(contour_levels>0))

for c in cs.collections:
    c.set_edgecolor("face")
lon_wid=y_lon_hi-y_lon_lo
lat_wid=y_lat_hi-y_lat_lo
rec = mpatches.Rectangle(ax.projection.transform_point(y_lon_lo, y_lat_lo,ccrs.PlateCarree()), lon_wid, lat_wid, facecolor="none", edgecolor='black', linewidth=1, linestyle='-',zorder=2)
ax.add_patch(rec)

# variable_z
ax = mp.subplot2grid((32,28),(17,18),colspan=9,rowspan=6, projection=ccrs.PlateCarree(central_longitude=clon_z))
ax.text(s='RMSE: '+ "{:.2f}".format(pareto_set_collect[ipar,2])+' '+vlist[2],x=0.70,y=0.99,ha='left',va='bottom',transform=ax.transAxes,fontsize=fontsize*0.3)
lons,lats = numpy.meshgrid(z_regional_lon_vals, z_regional_lat_vals)
mmem = numpy.mean(model_data_hist_z,axis=0)
ax.add_feature(cfeature.COASTLINE, facecolor='none', edgecolor='dimgray', linewidths=0.6)
lons1=lons-clon_z
contour_levels = contour_levels_zs
cs=ax.contourf(lons1, lats, mmem, levels=contour_levels, extend='both', cmap='RdYlBu_r', linestyles='none')
cbar = fig.colorbar(cs, ax=ax, shrink=0.6)
cbar.set_label(vlist_unit[2], fontsize=fontsize*0.3)
cbar.ax.tick_params(width=0.3,length=2.0,labelsize=fontsize*0.3)
cbar.solids.set_edgecolor("face")
mmem_minus_obs = numpy.mean(model_data_hist_z,axis=0)-dict_z['obs_field']
contour_levels = contour_levels_zc
ax.contour(lons1, lats, mmem_minus_obs, levels=contour_levels, colors='black', linewidths=0.3, linestyles=['--']*sum(contour_levels<0)+['-']*sum(contour_levels>0))

for c in cs.collections:
    c.set_edgecolor("face")
lon_wid=z_lon_hi-z_lon_lo
lat_wid=z_lat_hi-z_lat_lo
rec = mpatches.Rectangle(ax.projection.transform_point(z_lon_lo, z_lat_lo,ccrs.PlateCarree()), lon_wid, lat_wid, facecolor="none", edgecolor='black', linewidth=1, linestyle='-',zorder=2)
ax.add_patch(rec)

ax.text(s='Left column: Climatological winter mean '+vlist_label[0]+' (upper), ' + vlist_label[1] + ' (middle), '+vlist_label[2]+' (lower) panels in the model subset with minimum spatial RMSEs of '+vlist_label[0]+'.',x=-2.54,y=-0.3,ha='left',va='bottom',color='dimgray',transform=ax.transAxes,fontsize=fontsize*0.4)
ax.text(s='Middle column: Same as the left column but for the model subset with minimum spatial RMSEs of ' + vlist_label[1]+'.',x=-2.54,y=-0.4,ha='left',va='bottom',color='dimgray',transform=ax.transAxes,fontsize=fontsize*0.4)
ax.text(s='Right column: Same as the left column but for the model subset with minimum spatial RMSEs of '+vlist_label[2]+'.',x=-2.54,y=-0.5,ha='left',va='bottom',color='dimgray',transform=ax.transAxes,fontsize=fontsize*0.4)
ax.text(s='These model subsets corresponding to minimum RMSEs in '+vlist_label[0]+', ' + vlist_label[1] + ', and ' + vlist_label[2]+ ' are identified by the 3D Pareto-optimal analysis.',x=-2.54,y=-0.6,ha='left',va='bottom',color='dimgray',transform=ax.transAxes,fontsize=fontsize*0.4)

fig.savefig(os.environ["WK_DIR"]+"/model/PS/"+'spatial_patterns_with_minimum_rmse_in_pareto_front.pdf', transparent=True, bbox_inches='tight', dpi=1200)

