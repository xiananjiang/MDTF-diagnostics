# This file is part of the Pareto_optimal diagnosis module of the MDTF code package (see LICENSE.txt)
# ======================================================================
# calculate_optimal_front.py
#
#   Called by Pareto_optimal.py
#   Set paremeters for Pareto_optimal diagnosis
#
#
#!/usr/bin/env python
# coding: utf-8

# ## import necessary libraries
import numpy
from netCDF4 import Dataset
import matplotlib.pyplot as mp
import matplotlib.colors as mc
import matplotlib.cm as cm
import mpl_toolkits.mplot3d
import matplotlib
import scipy.ndimage
import datetime

import itertools
import random
import numpy.random
import scipy.stats
import os

mp.rcParams.update({'mathtext.default': 'regular'})

#from mpl_toolkits import basemap
#import mpl_toolkits.axes_grid1

#get_ipython().run_line_magic('matplotlib', 'inline')

# In[2]:

para = numpy.load("pareto_parameters.npy",allow_pickle=True)

degree_sign = para[()]['degree_sign']

target_model_names=para[()]["target_model_names"]
cmip_model_names=para[()]["cmip_model_names"]
model_names=numpy.hstack((numpy.array(target_model_names), cmip_model_names))

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
season=para[()]["season"]
pareto_k_values=para[()]["pareto_k_values"]
N_pareto_loops=para[()]["N_pareto_loops"]
uwind_level=para[()]["uwind_level"]

nmods = len(model_names)

# ## import preprocessed observational data set climatologies
# Pre-processing not shown here:
# * All data sets regridded onto a common 2.5-degree grid (72x144 latxlon)

data = numpy.load("input_data.npy",allow_pickle=True)

x_lat=data[()]["x_regional_lat_vals"]
x_lon=data[()]["x_regional_lon_vals"]
x_lat_inds = numpy.where((x_lat>=x_lat_lo) & (x_lat<=x_lat_hi))[0]
x_lon_inds = numpy.where((x_lon>=x_lon_lo) & (x_lon<=x_lon_hi))[0]
obs_field_x=data[()]["obs_field_x"][x_lat_inds[0]:(x_lat_inds[-1]+1), x_lon_inds[0]:(x_lon_inds[-1]+1)]
x_regional_nlat, x_regional_nlon = obs_field_x.shape
x_regional_lat_vals = x_lat[x_lat_inds[0]:(x_lat_inds[-1]+1)]
x_regional_lon_vals = x_lon[x_lon_inds[0]:(x_lon_inds[-1]+1)]

y_lat=data[()]["y_regional_lat_vals"]
y_lon=data[()]["y_regional_lon_vals"]
y_lat_inds = numpy.where((y_lat>=y_lat_lo) & (y_lat<=y_lat_hi))[0]
y_lon_inds = numpy.where((y_lon>=y_lon_lo) & (y_lon<=y_lon_hi))[0]
obs_field_y=data[()]["obs_field_y"][y_lat_inds[0]:(y_lat_inds[-1]+1), y_lon_inds[0]:(y_lon_inds[-1]+1)]
y_regional_nlat, y_regional_nlon = obs_field_y.shape
y_regional_lat_vals = y_lat[y_lat_inds[0]:(y_lat_inds[-1]+1)]
y_regional_lon_vals = y_lon[y_lon_inds[0]:(y_lon_inds[-1]+1)]

z_lat=data[()]["z_regional_lat_vals"]
z_lon=data[()]["z_regional_lon_vals"]
z_lat_inds = numpy.where((z_lat>=z_lat_lo) & (z_lat<=z_lat_hi))[0]
z_lon_inds = numpy.where((z_lon>=z_lon_lo) & (z_lon<=z_lon_hi))[0]
obs_field_z=data[()]["obs_field_z"][z_lat_inds[0]:(z_lat_inds[-1]+1), z_lon_inds[0]:(z_lon_inds[-1]+1)]
z_regional_nlat, z_regional_nlon = obs_field_z.shape
z_regional_lat_vals = z_lat[z_lat_inds[0]:(z_lat_inds[-1]+1)]
z_regional_lon_vals = z_lon[z_lon_inds[0]:(z_lon_inds[-1]+1)]

landsea_data=data[()]["landsea_data"][y_lat_inds[0]:(y_lat_inds[-1]+1), y_lon_inds[0]:(y_lon_inds[-1]+1)]

model_data_hist_x=data[()]["model_data_hist_x"][:, x_lat_inds[0]:(x_lat_inds[-1]+1), x_lon_inds[0]:(x_lon_inds[-1]+1)]
model_data_hist_y=data[()]["model_data_hist_y"][:, y_lat_inds[0]:(y_lat_inds[-1]+1), y_lon_inds[0]:(y_lon_inds[-1]+1)]
model_data_hist_z=data[()]["model_data_hist_z"][:, z_lat_inds[0]:(z_lat_inds[-1]+1), z_lon_inds[0]:(z_lon_inds[-1]+1)]

# NOW CALCULATE BIAS AND CONVERGENCE
bias_values_x = numpy.zeros((nmods))
bias_values_y = numpy.zeros((nmods))
bias_values_z = numpy.zeros((nmods))

for i in range(nmods):
    hist_field_x = model_data_hist_x[i,:,:]
    hist_field_y = model_data_hist_y[i,:,:]
    hist_field_z = model_data_hist_z[i,:,:]

    bias_values_x[i] = numpy.sqrt( numpy.mean((hist_field_x - obs_field_x)**2.) )
    bias_values_y[i] = numpy.sqrt( numpy.mean((hist_field_y - obs_field_y)**2.) )
    bias_values_z[i] = numpy.sqrt( numpy.mean((hist_field_z - obs_field_z)**2.) )

    if model_names[i] in [target_model_names]:
#      print(model_names[i])
       bias_values_x_target=bias_values_x[i]
       bias_values_y_target=bias_values_y[i]
       bias_values_z_target=bias_values_z[i]

mmem_bias_x = numpy.sqrt( numpy.mean( (numpy.mean(model_data_hist_x, axis=0) - obs_field_x)**2. ))
mmem_bias_y = numpy.sqrt( numpy.mean( (numpy.mean(model_data_hist_y, axis=0) - obs_field_y)**2. ))
mmem_bias_z = numpy.sqrt( numpy.mean( (numpy.mean(model_data_hist_z, axis=0) - obs_field_z)**2. ))

# In[18]:
# create dictionaries to be used below
dict_x = {
'bias_values_mods':bias_values_x,
'mmem_bias':mmem_bias_x,
'nlat':x_regional_nlat,
'nlon':x_regional_nlon,
'lats':x_regional_lat_vals,
'lons':x_regional_lon_vals,
'fields_hist_mods':model_data_hist_x,
'obs_field':obs_field_x,
}

dict_y = {
'bias_values_mods':bias_values_y,
'mmem_bias':mmem_bias_y,
'nlat':y_regional_nlat,
'nlon':y_regional_nlon,
'lats':y_regional_lat_vals,
'lons':y_regional_lon_vals,
'fields_hist_mods':model_data_hist_y,
'obs_field':obs_field_y,
}

dict_z = {
'bias_values_mods':bias_values_z,
'mmem_bias':mmem_bias_z,
'nlat':z_regional_nlat,
'nlon':z_regional_nlon,
'lats':z_regional_lat_vals,
'lons':z_regional_lon_vals,
'fields_hist_mods':model_data_hist_z,
'obs_field':obs_field_z,
}


# In[19]:

pareto_set_collect_2d_list = []
pareto_set_collect_3d_list = []
set_indices_collect_2d_list = []   #jxa
set_indices_collect_3d_list = []   #jxa


# ## Set up the ```N choose k``` combinations of models
# The cell below does values from ```k=1``` up to ```k=5```
# In[20]:
#k=3

all_combinations = []
model_numbers = numpy.arange(nmods, dtype=int)

#k_values = [1,2,3]
k_values = pareto_k_values
N_ens_count = 0
for k_idx in range(len(k_values)):
    k = k_values[k_idx]
    model_combinations_tmp = list(itertools.combinations(model_numbers, k))
    all_combinations.append(model_combinations_tmp)
    N_ens_count += len(model_combinations_tmp)

model_combinations = [numpy.array(item) for sublist in all_combinations for item in sublist]


# ## Set up objective function calculations to be ready for input into Pareto front calculation
# 
# These scripts use ```pareto.py```, an [evolutionary algorithm by Woodruff and Herman](https://github.com/matthewjwoodruff/pareto.py).
# 
# * ```DATESTRING``` automatically gets the date (Year-Month-Day_Hour:Minute:Second) so that the Pareto front information will be saved with the date attached (to keep track of when successive scripts are run)
# * The Pareto front is calculated in 2D for each combination in (precip., skin temp., and 200 hPa winds), and then it is done in 3D for all fields together
# * The variable ```N_pareto_loops``` specifies the number of successive times to run the ```pareto.py``` script
# 
# The cell below prepares all subensemble means for input into Pareto front calculation

# In[22]:


DATESTRING = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

#N_pareto_loops=5

# do N choose k subensembles
# for each, calculate ensemble mean

N_ens = N_ens_count

subensembles_hist_x = numpy.zeros((N_ens, dict_x['nlat'], dict_x['nlon']))
subensembles_hist_y = numpy.zeros((N_ens, dict_y['nlat'], dict_y['nlon']))
subensembles_hist_z = numpy.zeros((N_ens, dict_z['nlat'], dict_z['nlon']))

for i in range(N_ens):
    subensembles_hist_x[i,:,:] = numpy.mean(dict_x['fields_hist_mods'][model_combinations[i],:,:], axis=0)
    subensembles_hist_y[i,:,:] = numpy.mean(dict_y['fields_hist_mods'][model_combinations[i],:,:], axis=0)
    subensembles_hist_z[i,:,:] = numpy.mean(dict_z['fields_hist_mods'][model_combinations[i],:,:], axis=0)

bias_values_subensembles_x = numpy.zeros((N_ens))
bias_values_subensembles_y = numpy.zeros((N_ens))
bias_values_subensembles_z = numpy.zeros((N_ens))

for i in range(N_ens):
    hist_field_x = subensembles_hist_x[i,:,:]
    hist_field_y = subensembles_hist_y[i,:,:]
    hist_field_z = subensembles_hist_z[i,:,:]

    bias_values_subensembles_x[i] = numpy.sqrt( numpy.mean( (hist_field_x - dict_x['obs_field'])**2.) )
    bias_values_subensembles_y[i] = numpy.sqrt( numpy.mean( (hist_field_y - dict_y['obs_field'])**2.) )
    bias_values_subensembles_z[i] = numpy.sqrt( numpy.mean( (hist_field_z - dict_z['obs_field'])**2.) )


#jxa - to output target model
N_ens_count_target=0
for i in range(N_ens):
    nmodel_comb = model_combinations[i]
    length = len(nmodel_comb)
    for im in range(length):
       imodel=nmodel_comb[im]
       if model_names[imodel] in [target_model_names]:
#         print(model_names[nmodel_comb])
          N_ens_count_target += 1

N_ens_target=N_ens_count_target

bias_values_subensembles_x_target = numpy.zeros((N_ens_target))
bias_values_subensembles_y_target = numpy.zeros((N_ens_target))
bias_values_subensembles_z_target = numpy.zeros((N_ens_target))

N_ens_count_target=0
for i in range(N_ens):
    nmodel_comb = model_combinations[i]
    length = len(nmodel_comb)
    for im in range(length):
       imodel=nmodel_comb[im]
       if model_names[imodel] in [target_model_names]:
          bias_values_subensembles_x_target[N_ens_count_target]=bias_values_subensembles_x[i]
          bias_values_subensembles_y_target[N_ens_count_target]=bias_values_subensembles_y[i]
          bias_values_subensembles_z_target[N_ens_count_target]=bias_values_subensembles_z[i]
          N_ens_count_target += 1

# Calculating Pareto information (which_combo in 2D then 3D last)

# In[26]:


for which_combo in [1,2,3]:
    
    ##########
    # CALCULATING PARETO FRONT INFO
    # FIRST PARETO LOOP IS DONE HERE 
    print('calculating Pareto front for 2D combo '+str(which_combo))

    if which_combo==1:
        pareto_array = numpy.vstack((bias_values_subensembles_x, bias_values_subensembles_y)).T
    elif which_combo==2:
        pareto_array = numpy.vstack((bias_values_subensembles_x, bias_values_subensembles_z)).T
    elif which_combo==3:
        pareto_array = numpy.vstack((bias_values_subensembles_y, bias_values_subensembles_z)).T
    numpy.savetxt('data.txt', pareto_array, delimiter=',')
    os.system("python "+os.environ["POD_HOME"]+"/pareto.py data.txt --delimiter=',' --output='pareto_output.txt'")
    pareto_set = numpy.loadtxt('pareto_output.txt', delimiter=',')
    n_optima = pareto_set.shape[0]
    n_col = pareto_set.shape[1]

    if which_combo==1:
        col1_orig = numpy.copy(bias_values_subensembles_x)
        col2_orig = numpy.copy(bias_values_subensembles_y)
        col1 = numpy.copy(bias_values_subensembles_x)
        col2 = numpy.copy(bias_values_subensembles_y)
    elif which_combo==2:
        col1_orig = numpy.copy(bias_values_subensembles_x)
        col2_orig = numpy.copy(bias_values_subensembles_z)
        col1 = numpy.copy(bias_values_subensembles_x)
        col2 = numpy.copy(bias_values_subensembles_z)
    elif which_combo==3:
        col1_orig = numpy.copy(bias_values_subensembles_y)
        col2_orig = numpy.copy(bias_values_subensembles_z)
        col1 = numpy.copy(bias_values_subensembles_y)
        col2 = numpy.copy(bias_values_subensembles_z)

    set_indices = numpy.zeros((pareto_set.shape[0]), dtype=int)
    for i in range(pareto_set.shape[0]):
        set_indices[i] = numpy.where( (col1==pareto_set[i,0])&(col2==pareto_set[i,1]) )[0]

    pareto_set_collect = numpy.empty((0,2))
    pareto_set_collect = numpy.append(pareto_set_collect, pareto_set, axis=0)
    set_indices_collect = numpy.empty((0))   #jxa
    set_indices_collect = numpy.append(set_indices_collect, set_indices) #jxa
    # and then get rid of them
    col1[set_indices] = 999.
    col2[set_indices] = 999.

#   set_indices_collect = numpy.append(set_indices_collect, set_indices) #jxa
    # EXTRA PARETO FRONTS ARE DONE HERE, AS LONG AS N_pareto_loops>=1
    for loop in range(1,N_pareto_loops):
        print('calculating Pareto front '+str(loop+1))
        # now find indices where this front occurs

        pareto_array = numpy.vstack((col1, col2)).T
        numpy.savetxt('data.txt', pareto_array, delimiter=',')
#       os.system("python3 /work1/jiang/MDTF-diagnostics-3.0-beta.3/diagnostics/Pareto_optimal/pareto.py data.txt --delimiter=',' --output='pareto_output.txt'")
        os.system("python "+os.environ["POD_HOME"]+"/pareto.py data.txt --delimiter=',' --output='pareto_output.txt'")
        pareto_set = numpy.loadtxt('pareto_output.txt', delimiter=',')

        pareto_set_collect = numpy.append(pareto_set_collect, pareto_set, axis=0)

        set_indices = numpy.zeros((pareto_set.shape[0]), dtype=int)
        for i in range(pareto_set.shape[0]):
            set_indices[i] = numpy.where( (col1==pareto_set[i,0])&(col2==pareto_set[i,1]) )[0]
        # and then get rid of them
        set_indices_collect = numpy.append(set_indices_collect, set_indices) #jxa
        col1[set_indices] = 999.
        col2[set_indices] = 999.

        n_col = pareto_set.shape[1]    
        n_optima = pareto_set_collect.shape[0]
    
    pareto_set_collect_2d_list.append(pareto_set_collect)
    set_indices_collect_2d_list.append(set_indices_collect)  #jxa


# PARETO CALCULATIONS IN 3D
print('calculating 3D Pareto front')

#dict_x=dict_x
#dict_y=dict_y
#dict_z=dict_z

##########
# CALCULATING PARETO FRONT INFO
# FIRST PARETO LOOP IS DONE HERE
print('calculating first Pareto front for 3D surface')
pareto_array = numpy.vstack((bias_values_subensembles_x, bias_values_subensembles_y, bias_values_subensembles_z)).T
numpy.savetxt('data.txt', pareto_array, delimiter=',')
#os.system("python3 /work1/jiang/MDTF-diagnostics-3.0-beta.3/diagnostics/Pareto_optimal/pareto.py data.txt --delimiter=',' --output='pareto_set.txt'")
os.system("python "+os.environ["POD_HOME"]+"/pareto.py data.txt --delimiter=',' --output='pareto_set.txt'")
pareto_set = numpy.loadtxt('pareto_set.txt', delimiter=',')

# collect indices
set_indices = numpy.zeros(pareto_set.shape[0], dtype=int)
for i in range(pareto_set.shape[0]):
    #a = numpy.where( (bias_values_subensembles_x==pareto_set[i,0])&(bias_values_subensembles_y==pareto_set[i,1])&(bias_values_subensembles_z==pareto_set[i,2]) )[0][0]
    #print(a)
    set_indices[i] = numpy.where( (bias_values_subensembles_x==pareto_set[i,0])&(bias_values_subensembles_y==pareto_set[i,1])&(bias_values_subensembles_z==pareto_set[i,2]) )[0][0]

pareto_set_sizes_3d=[]
n_optima = pareto_set.shape[0]
pareto_set_sizes_3d.append(n_optima)
n_col = pareto_set.shape[1]

pareto_set_collect = numpy.empty((0,3))
pareto_set_collect = numpy.append(pareto_set_collect, pareto_set, axis=0)
set_indices_collect = numpy.empty((0))
set_indices_collect = numpy.append(set_indices_collect, set_indices)

col1_orig = numpy.copy(bias_values_subensembles_x)
col2_orig = numpy.copy(bias_values_subensembles_y)
col3_orig = numpy.copy(bias_values_subensembles_z)
col1 = numpy.copy(bias_values_subensembles_x)
col2 = numpy.copy(bias_values_subensembles_y)
col3 = numpy.copy(bias_values_subensembles_z)

col1[set_indices] = 999.
col2[set_indices] = 999.
col3[set_indices] = 999.

# EXTRA PARETO FRONTS ARE DONE HERE, AS LONG AS N_pareto_loops>=1
for loop in range(1,N_pareto_loops):
    print('calculating Pareto front '+str(loop+1))
    # now find indices where this front occurs

    pareto_array = numpy.vstack((col1, col2, col3)).T
    numpy.savetxt('data.txt', pareto_array, delimiter=',')
#   os.system("python3 /work1/jiang/MDTF-diagnostics-3.0-beta.3/diagnostics/Pareto_optimal/pareto.py data.txt --delimiter=',' --output='pareto_set.txt'")
    os.system("python "+os.environ["POD_HOME"]+"/pareto.py data.txt --delimiter=',' --output='pareto_set.txt'")
    pareto_set = numpy.loadtxt('pareto_set.txt', delimiter=',')

    pareto_set_collect = numpy.append(pareto_set_collect, pareto_set, axis=0)

    set_indices = numpy.zeros(pareto_set.shape[0], dtype=int)
    for i in range(pareto_set.shape[0]):
        set_indices[i] = numpy.where( (col1==pareto_set[i,0])&(col2==pareto_set[i,1])&(col3==pareto_set[i,2]) )[0][0]
    set_indices_collect = numpy.append(set_indices_collect, set_indices)

    n_col = pareto_set.shape[1]    
    n_optima = pareto_set_collect.shape[0]
    pareto_set_sizes_3d.append(pareto_set.shape[0])

    col1[set_indices] = 999.
    col2[set_indices] = 999.
    col3[set_indices] = 999.
    
pareto_set_collect_3d_list.append(pareto_set_collect)
set_indices_collect_3d_list.append(set_indices_collect)  #jxa


# # Calculate all biases for LENS
# In[27]:

k=5

all_combinations_LENS = []
model_numbers = numpy.arange(40, dtype=int)

k_values = [1,2,3,4,5]
N_ens_count = 0
for k_idx in range(len(k_values)):
    k = k_values[k_idx]
    model_combinations_tmp = list(itertools.combinations(model_numbers, k))
    all_combinations_LENS.append(model_combinations_tmp)
    N_ens_count += len(model_combinations_tmp)
#print(all_combinations[1])
model_combinations_LENS = [numpy.array(item) for sublist in all_combinations_LENS for item in sublist]


# In[28]:

#dict_x=dict_x
#dict_y=dict_y
#dict_z=dict_z


# # Save all information
# In[33]:


save_dict = {}

save_dict['pareto_set_collect_2d_list'] = pareto_set_collect_2d_list
save_dict['pareto_set_collect_3d_list'] = pareto_set_collect_3d_list

#jxa
save_dict['set_indices_collect_2d_list'] = set_indices_collect_2d_list
save_dict['set_indices_collect_3d_list'] = set_indices_collect_3d_list   #jxa

save_dict['bias_values_subensembles_x'] = bias_values_subensembles_x
save_dict['bias_values_subensembles_y'] = bias_values_subensembles_y
save_dict['bias_values_subensembles_z'] = bias_values_subensembles_z

save_dict['bias_values_subensembles_x_target'] = bias_values_subensembles_x_target
save_dict['bias_values_subensembles_y_target'] = bias_values_subensembles_y_target
save_dict['bias_values_subensembles_z_target'] = bias_values_subensembles_z_target

save_dict['k'] = k
save_dict['N_pareto_loops'] = N_pareto_loops

save_dict['N_ens'] = N_ens
save_dict['model_combinations'] = model_combinations

save_dict['N_ens_target'] = N_ens_target

save_dict['dict_x'] = dict_x
save_dict['dict_y'] = dict_y
save_dict['dict_z'] = dict_z

save_dict['bias_values_x_target'] = bias_values_x_target
save_dict['bias_values_y_target'] = bias_values_y_target
save_dict['bias_values_z_target'] = bias_values_z_target

save_dict['pareto_set_sizes_3d'] = pareto_set_sizes_3d
save_dict['model_names'] = model_names

save_dir = './'
save_filename = 'pareto_front_results_k1to5.npy'
numpy.save(save_dir + save_filename, save_dict)
