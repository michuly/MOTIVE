import sys
sys.path.append('/analysis/michalshaham/CrocoTools/Python_Kau/')
from tools.R_tools_new_michal import ncdimcopy, ncvarcopy
from netCDF4 import Dataset
import numpy as np

#  source ~/EastMedRun/EMed3km/module_load_2022
#  sample 1278 1556 360 639 temp_feb-apr.nc SST_2_7N_131_126W_feb.nc
#  ncea -F -d time,1,540 vort_feb-APR.nc out.nc

src='/atlantic3/michalshaham/EPAC2km/OUTPUT/HIS/vort_feb-APR.nc'
dst='/atlantic3/michalshaham/EPAC2km/OUTPUT/HIS/vort_subset2.nc'
src = Dataset(src, 'r')
dst = Dataset(dst, 'w')
var_name = 'rvort'

exclude_dims=['depth']
new_dims=dict(depth=1)
exclude_var=[var_name, 'depth']

ncdimcopy(src, dst, exclude = exclude_dims)
for var, value in zip(list(new_dims.keys()), list(new_dims.values())):
    dst.createDimension(var, value)

ncvarcopy(src, dst, exclude = exclude_var)

dims = src.variables[var_name].dimensions

##### this is where the subset takes place
subset = src.variables[var_name][:,0:1,:,:]
new_vars= {'depth':[-2], var_name:subset}

for var, value in zip(list(new_vars.keys()), list(new_vars.values())):
    dst.createVariable(var, np.dtype('float32').char, dims)
    dst.variables[var][:] = value

src.close()
dst.close()
