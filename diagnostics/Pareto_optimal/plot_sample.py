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
#ax.add_feature(cfeature.LAND, facecolor='gray', edgecolor='gray', linewidths=0.6)
#ax.set_extent([y_lon_lo_plt, y_lon_hi_plt, y_lat_lo_plt, y_lat_hi_plt], ccrs.PlateCarree())
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

