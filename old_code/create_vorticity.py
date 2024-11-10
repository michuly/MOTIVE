import sys
sys.path.append('/analysis/michalshaham/CrocoTools/Python_Kau/')
import numpy as np
import os
import netCDF4 as nc
# import R_tools_new_goth as tN
from tools.get_file_list import get_file_list
from R_tools_new_michal import ncload, vort, wrtNcVars, Forder

"""
build netcdf of vorticity
options are in "calculate_vort": horizontal slices, vertical slices, or all.
"""
zeta_min, zeta_max = -1.5, 1.5
path = '/southern/rbarkan/data/EPAC2km/'
pattern_his = 'OUTPUT/HIS/z_EPAC2km_his.14????.nc'
path_vort_zlev = '/atlantic3/michalshaham/EPAC2km/OUTPUT/EXT'
vort_ver_name = 'z_EPAC2km_vort_ver.%s.nc'
vort_hor_name = 'z_EPAC2km_vort_hor.%s.nc'
grd_name='Epac2km_grd.nc'

calculate_vort = ['all']


def horizontal_slices(dat_his, time_ind, depth_ind):
    u = dat_his.variables['u'][time_ind, depth_ind, :, :]
    v = dat_his.variables['v'][time_ind, depth_ind, :, :]
    return vort(np.asfortranarray(u.T, dtype=np.float32), np.asfortranarray(v.T, dtype=np.float32), pm, pn, simple=True)


def vertical_slices(dat_his, time_ind, eta_ind):
    u = dat_his.variables['u'][time_ind, :, eta_ind - 2:eta_ind + 2, :]
    v = dat_his.variables['v'][time_ind, :, eta_ind - 1:eta_ind + 2, :]
    pm_tmp = pm[:, eta_ind - 2:eta_ind + 2]
    pn_tmp = pn[:, eta_ind - 2:eta_ind + 2]
    # print(u.shape, v.shape, pn.shape, pm.shape)
    return vort(np.asfortranarray(u.T, dtype=np.float32), np.asfortranarray(v.T, dtype=np.float32),
                              pm_tmp, pn_tmp, simple=True)


with nc.Dataset(os.path.join(path, grd_name), 'r') as dat_grd:
    pm = ncload(dat_grd, 'pm')
    pn = ncload(dat_grd, 'pn')
    ind_equator = np.argmin(np.abs(dat_grd.variables['lat_rho'][:, 0]))


nums, his_files = get_file_list(path, pattern_his)
for num, his_file in zip(nums, his_files[:2]):

    print(his_file)
    dat_his = nc.Dataset(his_file, 'r')
    if 'all' in calculate_vort:
        vort_mat = np.empty((dat_his.dimensions['xi_rho'].size, dat_his.dimensions['eta_rho'].size, dat_his.dimensions['depth'].size, dat_his.dimensions['time'].size))
        vort_mat.fill(np.nan)
    if 'hor' in calculate_vort:
        vort_mat_hor = np.empty((dat_his.dimensions['xi_rho'].size, dat_his.dimensions['eta_rho'].size, 3, dat_his.dimensions['time'].size))
        vort_mat_hor.fill(np.nan)
    if 'ver' in calculate_vort:
        vort_mat_ver = np.empty((dat_his.dimensions['xi_rho'].size, 4, dat_his.dimensions['depth'].size, dat_his.dimensions['time'].size))
        vort_mat_ver.fill(np.nan)

    print('U shape ', (12, 103, 722, 2001))
    print('V shape ', (12, 103, 721, 2002))
    for i in range(dat_his.dimensions['time'].size):
        print(i, ' ', end='')

        if 'all' in calculate_vort:
            u = ncload(dat_his, 'u', itime=i)
            v = ncload(dat_his, 'v', itime=i)
            vort_mat[:,:,:,i] = vort(u, v, pm, pn, simple=True)

        # surface
        if 'hor' in calculate_vort:
            for j in [1,2]:
                vort_mat_hor[:,:,j,i]=horizontal_slices(dat_his, i, j)

        # equator
        if 'ver' in calculate_vort:
            vort_mat_ver[:, :, :, i]=vertical_slices(dat_his, i, ind_equator)

    print()
    dat_his.close()

    if 'all' in calculate_vort:
        wrtNcVars(his_file, dict(vort=Forder(vort_mat)))

    # surface
    if 'hor' in calculate_vort:
        print(os.path.join(path_vort_zlev, vort_hor_name % num))
        with nc.Dataset(os.path.join(path_vort_zlev, vort_hor_name % num), 'w') as nco:
            nco.createDimension('depth', 3)
            nco.createDimension('time', None)
            nco.createDimension('eta', 722)
            nco.createDimension('xi', 2002)
            nco.createVariable('vort', np.dtype('float32').char, ('xi', 'eta', 'depth', 'time'))
            nco.variables['vort'][:] = vort_mat_hor

    # equator
    if 'ver' in calculate_vort:
        print(os.path.join(path_vort_zlev, vort_ver_name % num))
        with nc.Dataset(os.path.join(path_vort_zlev, vort_ver_name % num), 'w') as nco:
            nco.createDimension('depth', 103)
            nco.createDimension('time', None)
            nco.createDimension('eta', 4)
            nco.createDimension('xi', 2002)
            nco.createVariable('vort', np.dtype('float32').char, ('xi', 'eta', 'depth', 'time'))
            nco.variables['vort'][:] = vort_mat_ver

