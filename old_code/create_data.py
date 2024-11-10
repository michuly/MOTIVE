import sys

sys.path.append('/analysis/michalshaham/CrocoTools/Python_Kau/')
import numpy as np
import os
import netCDF4 as nc
# import R_tools_new_goth as tN
from tools.get_file_list import get_file_list
from R_tools_new_michal import ncload, vort, wrtNcVars, Forder, DvDz, DuDz, DuDy, zlevs, gridDict

"""
build netcdf of any type of data
horizontal slices, vertical slices
"""

input_path = '/southern/rbarkan/data/EPAC2km/'
output_path = '/southern/michalshaham/Data/EPAC2km/'
pattern_his = 'OUTPUT/HIS/EPAC2km_his.14????.nc'
grd_name = 'Epac2km_grd.nc'

orientation='ver'  # 'all', 'ver', or 'hor'
data_file_name = 'OUTPUT/HIS/EPAC2km_his.%d.nc'
data_var = 'dvdz'

def calc_data(u, z_r, z_w):
    print(u.shape, z_r.shape, z_w.shape)
    return DvDz(u, z_r, z_w, simple=False, coord='r')
    # return DuDz(u, z_r, z_w, simple=False, coord='r')

def calc_data2(u, z_r, z_w):
    print(u.shape, z_r.shape, z_w.shape)
    return DvDy(u, z_r, z_w, simple=False, coord='r')
    # return DuDz(u, z_r, z_w, simple=False, coord='r')


def horizontal_slices(dat_his, time_ind, depth_ind):
    # u = dat_his.variables['u'][time_ind, depth_ind, :, :]
    # return calc_data(u, z_r, z_w)
    v = dat_his.variables['v'][time_ind, depth_ind, :, :]
    return calc_data(v, z_r, z_w)


def vertical_slices(dat_his, time_ind, eta_ind):
    # u = dat_his.variables['u'][time_ind, :, eta_ind - 2:eta_ind + 2, :]
    # return Forder(calc_data(np.asfortranarray(u.T, dtype=np.float32), z_r[:, eta_ind - 2:eta_ind + 2, :], z_w[:, eta_ind - 2:eta_ind + 2, :]))
    v = dat_his.variables['v'][time_ind, :, eta_ind - 1:eta_ind + 2, :]
    print(v.shape, z_r.shape, z_w.shape)
    return Forder(calc_data(np.asfortranarray(v.T, dtype=np.float32), z_r[:, eta_ind - 2:eta_ind + 2, :], z_w[:, eta_ind - 2:eta_ind + 2, :]))



# load grd data
with nc.Dataset(os.path.join(input_path, grd_name), 'r') as dat_grd:
    pm = ncload(dat_grd, 'pm')
    pn = ncload(dat_grd, 'pn')
    eta_ind = np.argmin(np.abs(dat_grd.variables['lat_rho'][:, 0] - 2))
    print('eta ind: ', eta_ind)

nums, his_files = get_file_list(input_path, pattern_his)
for num, his_file in zip(nums, his_files):

    print(his_file)
    dat_his = nc.Dataset(his_file, 'r')

    if orientation=='all':
        data_mat = np.empty((dat_his.dimensions['time'].size, dat_his.dimensions['s_rho'].size, dat_his.dimensions['eta_rho'].size, dat_his.dimensions['xi_rho'].size))
        data_mat.fill(np.nan)
    elif orientation=='hor':
        data_mat_hor = np.empty(
            (dat_his.dimensions['time'].size, 1, dat_his.dimensions['eta_rho'].size, dat_his.dimensions['xi_rho'].size))
        data_mat_hor.fill(np.nan)
    elif orientation=='ver':
        data_mat_ver = np.empty(
            (dat_his.dimensions['time'].size, dat_his.dimensions['s_rho'].size, 4, dat_his.dimensions['xi_rho'].size))
        data_mat_ver.fill(np.nan)

    print('U shape ', (12, 100, 722, 2001))
    print('V shape ', (12, 100, 721, 2002))

    for itime in range(dat_his.dimensions['time'].size):
        print(itime, ' ', end='')
        grd = gridDict(input_path, grd_name, ij=None)
        (z_r, z_w) = zlevs(grd, dat_his, itime=itime)
        # z_r = Forder(z_r)
        # z_w = Forder(z_w)

        if orientation=='all':
            u = ncload(dat_his, 'u', itime=itime)
            # v = ncload(dat_his, 'v', itime=i)
            data_mat[:, :, :, itime] = calc_data(u, z_r, z_w)

        # surface
        elif orientation=='hor':
            for j in [1, 2]:
                data_mat_hor[itime, j, :, :, :] = horizontal_slices(dat_his, itime, j)

        # equator
        elif orientation=='ver':
            data_mat_ver[itime, :, :, :,] = vertical_slices(dat_his, itime, eta_ind)

    if orientation=='all':
        wrtNcVars(his_file, dict(vort=Forder(data_mat)))

    # surface
    elif orientation=='hor':
        file_path=os.path.join(output_path, data_file_name % num)
        print(file_path)
        if os.path.exists(file_path):
            nco = nc.Dataset(file_path, 'a')
        else:
            nco = nc.Dataset(file_path, 'w')
            nco.createDimension('s_rho', 1)
            nco.createDimension('time', None)
            nco.createDimension('eta_rho', 722)
            nco.createDimension('xi_rho', 2002)
            nco.createDimension('eta_v', 721)
            nco.createDimension('xi_u', 2001)
            nco.createVariable('ocean_time', np.dtype('float32').char, ('time', ))

        nco.variables['ocean_time'][:] = dat_his.variables['ocean_time']
        # nco.createVariable('lon_rho', np.dtype('float32').char, ('eta_rho', 'xi_rho'))
        # nco.variables['lon_rho'][:] = dat_his.variables['lon_rho'][:]
        # nco.createVariable('lat_rho', np.dtype('float32').char, ('eta_rho', 'xi_rho'))
        # nco.variables['lat_rho'][:] = dat_his.variables['lat_rho'][:]

        # nco.createVariable(data_var, np.dtype('float32').char, ('time', 's_rho', 'eta_rho', 'xi_u'))
        if data_var not in nco.variables.keys():
            nco.createVariable(data_var, np.dtype('float32').char, ('time', 's_rho', 'eta_v', 'xi_rho'))
        nco.variables[data_var][:] = data_mat_hor
        nco.close()

    # equator
    elif orientation=='ver':
        file_path=os.path.join(output_path, data_file_name % num)
        print(file_path)
        if os.path.exists(file_path):
            nco = nc.Dataset(file_path, 'a')
        else:
            nco = nc.Dataset(file_path, 'w')
            nco.createDimension('s_rho', 100)
            nco.createDimension('time', None)
            nco.createDimension('eta_rho', 4)
            nco.createDimension('xi_rho', 2002)
            nco.createDimension('eta_v', 3)
            nco.createDimension('xi_u', 2001)
            nco.createVariable('ocean_time', np.dtype('float32').char, ('time', ))

        print(dat_his.variables['ocean_time'][:])
        nco.variables['ocean_time'][:] = dat_his.variables['ocean_time'][:]

        if data_var not in nco.variables.keys():
            nco.createVariable(data_var, np.dtype('float32').char, ('time', 's_rho', 'eta_rho', 'xi_rho'))
        # nco.createVariable(data_var, np.dtype('float32').char, ('xi_rho', 'eta_v', 'depth', 'time'))
        nco.variables[data_var][:] = data_mat_ver
        nco.close()

    dat_his.close()
