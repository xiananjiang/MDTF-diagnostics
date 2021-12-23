# This file is part of the Pareto_optimal diagnosis module of the MDTF code package (see LICENSE.txt)
# ======================================================================
# figure2_plot_optimal_fronts.py
#
#   Called by Pareto_optimal.py
#   Plot 2D and 3D optimal fronts
#
#
#!/usr/bin/env python
# coding: utf-8

# In[1]:

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

import matplotlib.patches

mp.rcParams.update({'mathtext.default': 'regular'})

#from mpl_toolkits import basemap
#import mpl_toolkits.axes_grid1

#get_ipython().run_line_magic('matplotlib', 'inline')

para = numpy.load("pareto_parameters.npy",allow_pickle=True)
target_model_names=para[()]["target_model_names"]
degree_sign = para[()]['degree_sign']
season=para[()]["season"]
uwind_level=para[()]["uwind_level"]
exp_name=para[()]["exp_name"]
vlist=para[()]["vlist"]

# ## use gridspec and custom 3d axis for plotting
# In[4]:

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.axis3d import Axis
import matplotlib.pyplot as plt
import matplotlib.projections as proj
from matplotlib.colors import colorConverter

# ## open data
# Specify ```DATESTRING``` from file you saved from the ```pareto_calculations*.ipynb``` script

save_dir = './'
save_filename = 'pareto_front_results_k1to5.npy'

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

bias_values_subensembles_x_target = save_dict[()]['bias_values_subensembles_x_target']
bias_values_subensembles_y_target = save_dict[()]['bias_values_subensembles_y_target']
bias_values_subensembles_z_target = save_dict[()]['bias_values_subensembles_z_target']

col1_orig = numpy.copy(bias_values_subensembles_x)
col2_orig = numpy.copy(bias_values_subensembles_y)
col3_orig = numpy.copy(bias_values_subensembles_z)

#
k = save_dict[()]['k']
N_pareto_loops = save_dict[()]['N_pareto_loops']

N_ens = save_dict[()]['N_ens']
N_ens_target = save_dict[()]['N_ens_target']

model_combinations = save_dict[()]['model_combinations']

dict_x = save_dict[()]['dict_x']
dict_y = save_dict[()]['dict_y']
dict_z = save_dict[()]['dict_z']

pareto_set_sizes_3d = save_dict[()]['pareto_set_sizes_3d']

# In[7]:
precip_pareto_rmse_vals = pareto_set_collect_3d_list[0][:,0]
precip_ALL_rmse_vals = bias_values_subensembles_x
precip_CMIP5_rmse_vals = dict_x['bias_values_mods']

# In[10]:
(0.869-0.604)/(2.938-0.619)

# In[16]:

markersize_small = 10
markersize_verysmall = 3
markersize_big = 20
fontsize=8
hfont={'fontname':'Helvetica'}

fig = mp.figure(figsize=(5,7))

if vlist[1]=='tos':
   vlist2='SST'
if vlist[1]=='prw':
   vlist2='PRW'

xlabel='Precip RMSE (mm day$^{-1}$)'
if vlist[1]=='tos':
    ylabel='SST RMSE ('+degree_sign+'C)'
if vlist[1]=='prw':
    ylabel='PRW RMSE (mm)'
zlabel='U'+str(uwind_level)+' RMSE (m s$^{-1}$)'

fig_titles = ['','(a)','(b)','(c)','(d)']
titles_instances = []

if uwind_level==200:
    precip_lim = (0,2.5)
    if vlist[1]=='tos':
        sst_lim = (0,2.0)
        sst_ticks = [0,0.5,1,1.5,2]
    if vlist[1]=='prw':
        sst_lim = (0,6.0)
        sst_ticks = [0,1,2,3,4,5,6]
    u200_lim = (0,11.0)

    precip_ticks = [0,1,2]
    sst_ticks = [0,0.5,1,1.5,2]
    u200_ticks = [0,2,4,6,8,10]

if uwind_level==850:
    precip_lim = (0,8.0)
    precip_ticks = [0,2,4,6,8]
    if vlist[1]=='tos':
        sst_lim = (0,2.0)
        sst_ticks = [0,0.5,1,1.5,2]
    if vlist[1]=='prw':
        sst_lim = (0,6.0)
        sst_ticks = [0,1,2,3,4,5,6]
        precip_lim = (0,6.0)
        precip_ticks = [0,2,4,6]
    u200_lim = (0,5.0)
    u200_ticks = [0,1,2,3,4,5]

pr_error = 0.0816906403487*1.96
sst_error = 0.02964993495*1.96
u200_error = 0.264875178502*1.96

# 1 is precip and SSTs
# 2 is precip and u winds
# 3 is SSTs and winds

n_pareto_sgcm_tot = numpy.zeros((3))
n_pareto_sgcm_percent = numpy.zeros((3))
percent_tmods_pareto_2d = numpy.zeros((3))
psort_2d = numpy.zeros((3))

bias_sort = numpy.argsort(dict_x['bias_values_mods'])
bias_sort2 = numpy.argsort(dict_y['bias_values_mods'])
bias_sort3 = numpy.argsort(dict_z['bias_values_mods'])
ranka = numpy.zeros((len(model_names)),dtype=int)
rankb = numpy.zeros((len(model_names)),dtype=int)
rankc = numpy.zeros((len(model_names)),dtype=int)

for i in range(nmods):
    ranka[i]=i+1
    for j in range(nmods):
        if bias_sort2[j] == bias_sort[i]:
             rankb[i]=j+1
    for j in range(nmods):
        if bias_sort3[j] == bias_sort[i]:
             rankc[i]=j+1

for which_combo in [1,2,3]:
    
    pareto_set_collect = pareto_set_collect_2d_list[which_combo-1]
    length_pareto = len(pareto_set_collect)
    
    set_indices_collect = set_indices_collect_2d_list[which_combo-1]
    length_induces = len(set_indices_collect)

    if length_pareto != length_induces:
        print('length_pareto and length_induces do not match!')  

    print('plotting 2D combo '+str(which_combo))

    colors = cm.nipy_spectral(numpy.linspace(0.1,1,nmods+1,endpoint=True))
    
#   ax = mp.subplot2grid((26,20),((which_combo-1)*9+2,0),colspan=6,rowspan=6)
    ax = mp.subplot2grid((28,20),((which_combo-1)*9+2,0),colspan=6,rowspan=6)

    title = ax.text(s=fig_titles[which_combo],x=0,y=1.03,fontsize=fontsize,ha='left',va='bottom',transform=ax.transAxes)
    titles_instances.append(title)
    # add 3D title that is flush with (a)
    if which_combo==1:
        title = ax.text(s=fig_titles[4],x=1.4,y=1.03,fontsize=fontsize,ha='left',va='bottom',transform=ax.transAxes)
        titles_instances.append(title)
    
    if which_combo==1:
        for i in range(nmods):
           #clabel='('+"{0:2d}".format(i+1)+','+"{0:2d}".format(bias_sort2[bias_sort[i]])+','+"{0:2d}".format(bias_sort3[bias_sort[i]])+') '
            clabel='('+"{0:2d}".format(ranka[i])+','+"{0:2d}".format(rankb[i])+','+"{0:2d}".format(rankc[i])+') '
            if model_names[bias_sort][i] in [target_model_names]:
                ax.scatter(dict_x['bias_values_mods'][bias_sort][i], dict_y['bias_values_mods'][bias_sort][i], s=markersize_big/0.5, marker='*', facecolor='None', edgecolor='magenta', label=clabel+model_names[bias_sort][i], linewidth=1, zorder=8, rasterized=False)
            else:
                ax.scatter(dict_x['bias_values_mods'][bias_sort][i], dict_y['bias_values_mods'][bias_sort][i], s=markersize_big, marker='o', facecolor='None', edgecolor=colors[i,:], label=clabel+model_names[bias_sort][i], linewidth=1, zorder=3, rasterized=False)
    elif which_combo==2:
        for i in range(nmods):
            if model_names[bias_sort][i] in [target_model_names]:
                ax.scatter(dict_x['bias_values_mods'][bias_sort][i], dict_z['bias_values_mods'][bias_sort][i], s=markersize_big/0.5, marker='*', facecolor='None', edgecolor='magenta', label=clabel+model_names[bias_sort][i], linewidth=1, zorder=8, rasterized=False)
            else:
                ax.scatter(dict_x['bias_values_mods'][bias_sort][i], dict_z['bias_values_mods'][bias_sort][i], s=markersize_big, marker='o', facecolor='None', edgecolor=colors[i,:], label=clabel+model_names[bias_sort][i], linewidth=1, zorder=3, rasterized=False)
    elif which_combo==3:
        for i in range(nmods):
            if model_names[bias_sort][i] in [target_model_names]:
                ax.scatter(dict_y['bias_values_mods'][bias_sort][i], dict_z['bias_values_mods'][bias_sort][i], s=markersize_big/0.5, marker='*', facecolor='None', edgecolor='magenta', label=clabel+model_names[bias_sort][i], linewidth=1, zorder=8, rasterized=False)
            else:
                ax.scatter(dict_y['bias_values_mods'][bias_sort][i], dict_z['bias_values_mods'][bias_sort][i], s=markersize_big, marker='o', facecolor='None', edgecolor=colors[i,:], label=clabel+model_names[bias_sort][i], linewidth=1, zorder=3, rasterized=False)

    # MMEM
    if which_combo==1:
        ax.scatter(dict_x['mmem_bias'], dict_y['mmem_bias'], s=markersize_big/1.5, marker='s', edgecolor='0', facecolor='None', label='CMIP6 ensemble mean', linewidth=1, zorder=7, rasterized=False)
        ellipse_plot = matplotlib.patches.Ellipse(xy=[0,0], width=2*pr_error, height=2*sst_error, angle=0, facecolor='orange', edgecolor='darkorange', zorder=5)
        ax.add_artist(ellipse_plot)
    elif which_combo==2:
        ax.scatter(dict_x['mmem_bias'], dict_z['mmem_bias'], s=markersize_big/1.5, marker='s', edgecolor='0', facecolor='None', label='CMIP6 ensemble mean', linewidth=1, zorder=7, rasterized=False)
        ellipse_plot = matplotlib.patches.Ellipse(xy=[0,0], width=2*pr_error, height=2*u200_error, angle=0, facecolor='orange', edgecolor='darkorange', zorder=5)
        ax.add_artist(ellipse_plot)
    elif which_combo==3:
        ax.scatter(dict_y['mmem_bias'], dict_z['mmem_bias'], s=markersize_big/1.5, marker='s', edgecolor='0', facecolor='None', label='CMIP6 ensemble mean', linewidth=1, zorder=7, rasterized=False)
        ellipse_plot = matplotlib.patches.Ellipse(xy=[0,0], width=2*sst_error, height=2*u200_error, angle=0, facecolor='orange', edgecolor='darkorange', zorder=5)
        ax.add_artist(ellipse_plot)
        
    # Target model SUBENSEMBLES
    if (which_combo==1):
        ax.scatter(bias_values_subensembles_x_target, bias_values_subensembles_y_target, marker='.', s=markersize_verysmall/3, facecolor='silver', color='silver', label=target_model_names+' subens. ('+str(N_ens_target)+')', zorder=2, rasterized=True)
    elif (which_combo==2):
        ax.scatter(bias_values_subensembles_x_target, bias_values_subensembles_z_target, marker='.', s=markersize_verysmall/3, facecolor='silver', color='silver', label=target_model_names+' subens. ('+str(N_ens_target)+')', zorder=2, rasterized=True)
    elif (which_combo==3):
        ax.scatter(bias_values_subensembles_y_target, bias_values_subensembles_z_target, marker='.', s=markersize_verysmall/3, facecolor='silver', edgecolor='silver', label=target_model_names+' subens. ('+str(N_ens_target)+')', zorder=2, rasterized=True)

    
    if which_combo==1: #xy
        ax.scatter(bias_values_subensembles_x[39:], bias_values_subensembles_y[39:], marker='.', s=markersize_verysmall, edgecolor='0.5', facecolor='0.5', zorder=1, label='CMIP6 subens. ('+str(N_ens)+')', rasterized=True)
        ax.set_xlabel(xlabel, size=fontsize-1, labelpad=-0.1)
        ax.set_ylabel(ylabel, size=fontsize-1, labelpad=-0.1)
        ax.set_ylim(sst_lim)
        ax.set_xlim(precip_lim)
        ax.set_xticks(precip_ticks)
        ax.set_yticks(sst_ticks)
    elif which_combo==2:
        ax.scatter(bias_values_subensembles_x[39:], bias_values_subensembles_z[39:], marker='.', s=markersize_verysmall, edgecolor='0.5', facecolor='0.5', zorder=1, label='CMIP6 subens. ('+str(N_ens)+')', rasterized=True)
        ax.set_xlabel(xlabel, size=fontsize-1, labelpad=-0.1)
        ax.set_ylabel(zlabel, size=fontsize-1, labelpad=-0.1)
        ax.set_ylim(u200_lim)
        ax.set_xlim(precip_lim)
        ax.set_xticks(precip_ticks)
        ax.set_yticks(u200_ticks)
    elif which_combo==3:
        ax.scatter(bias_values_subensembles_y[39:], bias_values_subensembles_z[39:], marker='.', s=markersize_verysmall, edgecolor='0.5', facecolor='0.5', zorder=1, label='CMIP6 subens. ('+str(N_ens)+')', rasterized=True)
        ax.set_xlabel(ylabel, size=fontsize-1, labelpad=-0.1)
        ax.set_ylabel(zlabel, size=fontsize-1, labelpad=-0.1)
        ax.set_xlim(sst_lim)
        ax.set_ylim(u200_lim)
        ax.set_xticks(sst_ticks)
        ax.set_yticks(u200_ticks)
        
    ax.tick_params(direction='out', length=2, labelsize=fontsize-3, pad=0.5)

    ax.scatter(pareto_set_collect[:,0], pareto_set_collect[:,1], marker='.', s=markersize_verysmall, edgecolor='firebrick', facecolor='firebrick', zorder=2, label='Pareto-optimal set (2D)', rasterized=True)

    n_pareto_sgcm=0
    pareto_set_collect_sgcm=pareto_set_collect
    for ipareto in range(length_pareto):
        if target_model_names in model_names[model_combinations[int(set_indices_collect[ipareto])]]:
             pareto_set_collect_sgcm[n_pareto_sgcm,0]=pareto_set_collect[ipareto,0]
             pareto_set_collect_sgcm[n_pareto_sgcm,1]=pareto_set_collect[ipareto,1]
             n_pareto_sgcm += 1
    n_pareto_sgcm_tot[which_combo-1]=length_pareto
    n_pareto_sgcm_percent[which_combo-1]=n_pareto_sgcm/length_pareto * 100.
    ax.scatter(pareto_set_collect_sgcm[:n_pareto_sgcm,0], pareto_set_collect_sgcm[:n_pareto_sgcm,1], marker='.', s=markersize_verysmall, edgecolor='magenta', facecolor='magenta', zorder=6, label='Pareto-optimal set: '+target_model_names, rasterized=True)

    tmods_pareto = numpy.zeros((nmods))
    for ipareto in range(length_pareto):
         for imod in range(nmods):
           if imod in model_combinations[int(set_indices_collect[ipareto])]:
              tmods_pareto[imod] += 1
    percent_tmods_pareto = tmods_pareto/length_pareto*100
    psort = numpy.argsort(-tmods_pareto)
    percent_tmods_pareto_2d[which_combo-1] = percent_tmods_pareto[psort][0]
    psort_2d[which_combo-1] = psort[0]

#   ax.text('this is a test',x=5.4,y=1.03,fontsize=fontsize,ha='left',va='bottom',transform=ax.transAxes)

    if which_combo==1:
        handles,labels = ax.get_legend_handles_labels()
#       labels = labels[:-3]+[labels[-2]]+[labels[-3]]+[labels[-1]]
#       handles = handles[:-3]+[handles[-2]]+[handles[-3]]+[handles[-1]]
#jxa
        for i in range(nmods):
            if model_names[bias_sort][i] in [target_model_names]:
                labels = [labels[i]]+labels[0:i]+labels[i+1:]
                handles = [handles[i]]+handles[0:i]+handles[i+1:]
    ax.grid()
    ax.set_axisbelow(True)
        
#jxa
ax = mp.subplot2grid((32,20),(26,8),colspan=11,rowspan=4)
ax.axis('off')
title = ax.text(x=-0.1, y=0.61, s='Note: The above 3 model ranks are for RMSE of Precip, ' + vlist2 + ', U'+str(uwind_level)+', respectively.',fontsize=fontsize*0.55,ha='left',va='bottom',transform=ax.transAxes)
title = ax.text(x=-0.1, y=0.49, s='With all combinations of N = ' + str(nmods) + ' models up to sub-ensembles of k=3.',fontsize=fontsize*0.55,ha='left',va='bottom',transform=ax.transAxes)
title = ax.text(x=-0.1, y=0.37, s='Pareto 2D Set (' + vlist2 + '-Precip):  All ' + str(int(n_pareto_sgcm_tot[0])) + ', '+target_model_names+' ' +  str("{:.1f}".format(n_pareto_sgcm_percent[0])) + '% (#1 ' + model_names[int(psort_2d[0])] + ': '+"{:.1f}".format(percent_tmods_pareto_2d[0])+'%)',fontsize=fontsize*0.55,ha='left',va='bottom',transform=ax.transAxes)
title = ax.text(x=0.11, y=0.25, s='(U'+str(uwind_level)+'-Precip): All ' + str(int(n_pareto_sgcm_tot[1])) + ', '+target_model_names+' ' +  str("{:.1f}".format(n_pareto_sgcm_percent[1])) + '% (#1 ' + model_names[int(psort_2d[1])] + ': '+"{:.1f}".format(percent_tmods_pareto_2d[1])+'%)',fontsize=fontsize*0.55,ha='left',va='bottom',transform=ax.transAxes)
title = ax.text(x=0.11, y=0.13, s='(U'+str(uwind_level)+'-' + vlist2 + '):    All ' + str(int(n_pareto_sgcm_tot[2])) + ', '+target_model_names+' ' +  str("{:.1f}".format(n_pareto_sgcm_percent[2])) + '% (#1 ' + model_names[int(psort_2d[2])] + ': '+"{:.1f}".format(percent_tmods_pareto_2d[2])+'%)',fontsize=fontsize*0.55,ha='left',va='bottom',transform=ax.transAxes)

####################################################################################################
print('plotting 3D combo')

pareto_set_collect = pareto_set_collect_3d_list[0]
set_indices_collect = set_indices_collect_3d_list[0]

#jxa
ax = mp.subplot2grid((28,20),(0,9),colspan=11,rowspan=11,projection='3d')

# COLORIZING PARETO FRONT
min_val = 0.3
max_val = 0.7
colors = [ [i/(len(pareto_set_sizes_3d)-1)]*pareto_set_sizes_3d[i] for i in range(len(pareto_set_sizes_3d)) ]
colors = numpy.array(([item for sublist in colors for item in sublist]))*(max_val-min_val)+min_val
cmap = mp.get_cmap('inferno')
colors = [cmap(i) for i in colors]

n_pareto_sgcm=0
length_pareto=len(pareto_set_collect)
for ipareto in range(length_pareto):
    if target_model_names in model_names[model_combinations[int(set_indices_collect[ipareto])]]:
             colors[ipareto]=(1,0,1,1.0)     #magenta for candidate model
             n_pareto_sgcm += 1
n_pareto_sgcm_tot_3d=length_pareto
n_pareto_sgcm_percent_3d=n_pareto_sgcm/length_pareto * 100.

ax.scatter(bias_values_subensembles_x[38:],bias_values_subensembles_y[38:],bias_values_subensembles_z[38:],facecolor='0.5',edgecolor='0.5',marker='.', s=markersize_verysmall, zorder=0, alpha=1, rasterized=True)
ax.scatter(pareto_set_collect[:,0][::-1],pareto_set_collect[:,1][::-1],pareto_set_collect[:,2][::-1],facecolor=colors[::-1],edgecolor=colors[::-1], marker='.', s=markersize_verysmall, alpha=1, zorder=1, rasterized=True, label='Pareto-optimal set (3D)')

tmods_pareto = numpy.zeros((nmods))
for ipareto in range(length_pareto):
    for imod in range(nmods):
       if imod in model_combinations[int(set_indices_collect[ipareto])]:
            tmods_pareto[imod] += 1
percent_tmods_pareto = tmods_pareto/length_pareto*100
psort = numpy.argsort(-tmods_pareto)

xlab = ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=-2)
ylab = ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=-2)
ax.zaxis.set_rotate_label(False)  # disable automatic rotation
zlab = ax.set_zlabel(zlabel, fontsize=fontsize, labelpad=-2, rotation=90)

ax.scatter(0,0,0,marker='*',s=150,color='0.25',edgecolor='0')
ax.text(s='0', x=0, y=0, z=-1, fontsize=fontsize, ha='center', va='center')

if uwind_level==200:
    ax.set_xlim(0,2.5)
    ax.set_zlim(0,8)

    precip_ticks = [0,0.5,1,1.5,2,2.5]
    u200_ticks = [0,1,2,3,4,5,6,7,8]
   
    precip_ticklabels = ['','','1','','2','']
    u200_ticklabels = ['0','','2','','4','','6','','8']

    if vlist[1]=='tos':
        ax.set_ylim(0,2.0)
        sst_ticks = [0,0.5,1,1.5,2]
        sst_ticklabels = ['','','1','','2']
    if vlist[1]=='prw':
        ax.set_ylim(0,6.0)
        sst_ticks = [0,1,2,3,4,5,6]
        sst_ticklabels = ['0','1','2','3','4','5','6']

if uwind_level==850:
    ax.set_xlim(0,8.0)
    ax.set_zlim(0,4.0)

    precip_ticks = [0,1,2,3,4,5,6,7,8]
    u200_ticks = [0,0.5,1,1.5,2,2.5,3,3.5,4]

    precip_ticklabels = ['','','2','','4','','6','','8']
    u200_ticklabels = ['0','','1','','2','','3','','4']

    if vlist[1]=='tos':
        ax.set_ylim(0,2.0)
        sst_ticks = [0,0.5,1,1.5,2]
        sst_ticklabels = ['','','1','','2']
    if vlist[1]=='prw':
        ax.set_ylim(0,6.0)
        sst_ticks = [0,1,2,3,4,5,6]
        sst_ticklabels = ['0','1','2','3','4','5','6']

ax.set_xticks(precip_ticks)
ax.set_yticks(sst_ticks)
ax.set_zticks(u200_ticks)

ax.set_xticklabels(precip_ticklabels)
ax.set_yticklabels(sst_ticklabels)
ax.set_zticklabels(u200_ticklabels)

ax.view_init(10,225) # elevation, azimuthal

ax.xaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('None')
ax.yaxis.pane.fill = False
ax.yaxis.pane.set_edgecolor('None')
ax.zaxis.pane.fill = False
ax.zaxis.pane.set_edgecolor('None')

[t.set_va('center') for t in ax.get_yticklabels()]
[t.set_ha('left') for t in ax.get_yticklabels()]
[t.set_va('center') for t in ax.get_xticklabels()]
[t.set_ha('right') for t in ax.get_xticklabels()]
[t.set_va('center') for t in ax.get_zticklabels()]
[t.set_ha('left') for t in ax.get_zticklabels()]

ax.tick_params(labelsize=fontsize-1, pad=0)

#ax.yaxis.set_gridline_color((0, 'black'))
################################################################################
################################################################################
################################################################################
# create legend #
#ax_outer_legend = mp.subplot2grid((26,20),(14,9),colspan=11,rowspan=12, frameon=False) #fig.add_subplot(111, frameon=False)
ax_outer_legend = mp.subplot2grid((32,20),(14,8),colspan=11,rowspan=11, frameon=False) #fig.add_subplot(111, frameon=False)
ax_outer_legend.axes.get_xaxis().set_visible(False)
ax_outer_legend.axes.get_yaxis().set_visible(False)

handles2,labels2 = ax.get_legend_handles_labels()
handles+=handles2
labels+=labels2
#labels = labels[:-2]+[labels[-1]]+[labels[-2]]
#handles = handles[:-2]+[handles[-1]]+[handles[-2]]
title = ax_outer_legend.text(x=0.09, y=1.00, s='Model with rank by (Precip, '+ vlist2 +', U'+str(uwind_level)+')',fontsize=fontsize*0.9,ha='left',color='blue',va='bottom')

N_scatter=15
offsets=numpy.random.normal(loc=0.5, scale=0.25, size=N_scatter)
fig_outer_legend_two = ax_outer_legend.legend(handles[-5:], labels[-5:], fontsize=fontsize*0.8, ncol=2, scatterpoints=N_scatter, scatteryoffsets=offsets, borderpad=0., borderaxespad=0., handletextpad=0., handlelength=1.0, loc='lower center', bbox_to_anchor=(0.5,-0.17), bbox_transform=ax_outer_legend.transAxes, edgecolor='0', framealpha=0, columnspacing=0, labelspacing=0.12, fancybox=False)

fig_outer_legend_one = ax_outer_legend.legend(handles[:-5], labels[:-5], fontsize=fontsize*0.8, ncol=2, scatterpoints=1, borderpad=0., borderaxespad=0., handletextpad=-0.5, loc='lower left', bbox_to_anchor=(-0.1,0.05), bbox_transform=ax_outer_legend.transAxes, edgecolor='0', framealpha=0, columnspacing=0, labelspacing=0.1, fancybox=False)

# change 3D info
cmap_values = offsets*(max_val-min_val)+min_val
fig_outer_legend_two.legendHandles[-1].set_color(cmap(cmap_values))
mp.gca().add_artist(fig_outer_legend_two)

#jxa
ax = mp.subplot2grid((32,20),(30,9),colspan=11,rowspan=1, frameon=False)
ax.axis('off')
title = ax.text(x=-0.19, y=1.28, s='Pareto 3D Set:  All ' + str(n_pareto_sgcm_tot_3d) + ', '+target_model_names+' ' +  str("{:.1f}".format(n_pareto_sgcm_percent_3d)) + '% (#1 ' + model_names[psort][0] + ': '+"{:.1f}".format(percent_tmods_pareto[psort][0])+'%)',fontsize=fontsize*0.55,ha='left',va='bottom',transform=ax.transAxes)

################################################################################

#jxa
fig.savefig(os.environ["WK_DIR"]+"/model/PS/"+'pareto_fronts_2d_and_3d.pdf', transparent=True, bbox_extra_artists=[xlab,ylab,zlab]+titles_instances+[fig_outer_legend_one,fig_outer_legend_two], dpi=1200)
#fig.savefig('pareto_fronts_2d_and_3d.pdf', transparent=True, bbox_extra_artists=[xlab,ylab,zlab]+titles_instances+[fig_outer_legend_one,fig_outer_legend_two], dpi=1200)

#mp.show()
