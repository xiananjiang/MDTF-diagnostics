; This file is part of the Pareto_optimal module of the MDTF code package (see LICENSE.txt)

begin

;print_clock("NCL: Calculating winter mean (DJF) state for the targeting model output...")

wk_dir = getenv("WK_DIR")
pr_file   = getenv("PR_FILE")
tos_file  = getenv("TOS_FILE")
ua_file   = getenv("UA_FILE")

varn = (/"pr","tos","ua"/)
infile = (/pr_file,tos_file,ua_file/)

do j=0,2

var=varn(j)
;print(var)

a = addfile(infile(j),"r")
v1 = a->$var$
time     = cd_calendar(v1&time, -1)
v1&time  =  time
time     = doubletoint(v1&time)
mon      = time%100
indi     = ind(mon.ge.12.or.mon.le.2)
v2       = v1(indi,:,:)
v3       = dim_avg_n_Wrap(v2,0)

lat=fspan(-88.75,88.75,72)
lat@units="degree_north"
lat@long_name="lat"
lon=fspan(0,357.5,144)
lon@units="degree_east"
lon@long_name="lon"

v4      = linint2_Wrap(v3&lon,v3&lat,v3,True,lon,lat,0)

if (var .eq. "pr" ) then
; convert to mm day-1 to match the pre-processed CMIP6 model output
v4 = v4*86400.
end if

v3!0    = "lat"
v3&lat  = lat
v3!1    = "lon"
v3&lon  = lon

system("rm -f "+wk_dir+"/model/netCDF/model_"+var+"_climatology_djf.nc")
fout = addfile(wk_dir+"/model/netCDF/model_"+var+"_climatology_djf.nc","c")
fout->$var$    = v4
if (var .eq. "pr" ) then
fout->$var$@units="mm day-1"
end if
if (var .eq. "tos" ) then
fout->$var$@units="degC"
end if
delete([/v1,v2,v3,v4,a,time,mon,indi,lat,lon,fout/])

end do

end
