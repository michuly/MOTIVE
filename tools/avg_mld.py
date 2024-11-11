import sys
sys.path.append('/analysis/michalshaham/CrocoTools/Python_Kau/')
from R_tools_new_michal import gridDict, ncload, wrtNcfile_2d, wrtNcVars_2d, vert_integ_weights, zlevs, u2rho, v2rho, \
    Forder
import numpy as np
import os
from tools.get_file_list import get_file_list
from netCDF4 import Dataset

path = '/atlantic3/michalshaham/EMedCroco3km_A/'
pattern_mld_data = 'OUTPUT/his/EMed3km_data_mld.*.nc'
grd_name="INPUT/EMed3km_grd.nc"

with Dataset(os.path.join(path, grd_name), 'r') as dat_grd:
    pm = ncload(dat_grd, 'pm')
    pn = ncload(dat_grd, 'pn')
    eta_size = dat_grd.dimensions['eta_rho'].size
    eta = np.arange(eta_size)
    xi_size = dat_grd.dimensions['xi_rho'].size
    xi = np.arange(xi_size)
    X, Y = np.meshgrid(xi, eta, indexing='ij')
grd = gridDict(path, grd_name, ij=None)
nums, mld_files = get_file_list(path, pattern_mld_data, num_pattern=r'\b(\d{5})\b')
mixed_layer=np.zeros((682, 452))
for i, mld_file in enumerate(mld_files):
    if i>30:
        break
    dat_mld = Dataset(mld_file, 'r')
    for j in range(dat_mld.dimensions['time'].size):
        mixed_layer += ncload(dat_mld, 'mld', itime=j)

mldname = '/atlantic3/michalshaham/EMedCroco3km_A/INPUT/EMed3km_mld.nc'
wrtNcfile_2d(mldname, grd)
wrtNcVars_2d(mldname, vardict=dict(ml_depth=Forder(mixed_layer)), dim_names=('eta_rho', 'xi_rho'))
nco = Dataset(mldname, 'a')
nco.createDimension('eta_u', 452)
nco.createDimension('xi_u', 681)
nco.createDimension('eta_v', 451)
nco.createDimension('xi_v', 682)
nco.createDimension('eta_psi', 451)
nco.createDimension('xi_psi', 681)
nco.close()

