import sys
sys.path.append('/analysis/michalshaham/CrocoTools/Python_Kau/')
import numpy as np
import os
import netCDF4 as nc
# import R_tools_new_goth as tN
from tools.get_file_list import get_file_list
from R_tools_new_michal import gridDict, ncload, wrtNcfile_2d, wrtNcVars_2d, v2rho, u2rho, Forder

"""
build netcdf of vorticity
options are in "calculate_vort": horizontal slices, vertical slices, or all.
"""
path = '/southern/rbarkan/data/EPAC2km/'
pattern_his = 'OUTPUT/HIS/EPAC2km_his.14????.nc'
path_data = '/atlantic3/michalshaham/EPAC2km/OUTPUT/HIS'
data_name = 'surface_data_5_%d.nc'
grd_name='Epac2km_grd.nc'
saving_step = 2


with nc.Dataset(os.path.join(path, grd_name), 'r') as dat_grd:
    lon_rho = Forder(ncload(dat_grd, 'lon_rho'))
    lat_rho = Forder(ncload(dat_grd, 'lat_rho'))
grd = gridDict(path, grd_name, ij=None)

nums, his_files = get_file_list(path, pattern_his)
with nc.Dataset(his_files[0], 'r') as dat_his:
    time_dim_size = dat_his.dimensions['time'].size
    time_dim_size_tot = time_dim_size * len(nums)
    eta_rho_dim_size = dat_his.dimensions['eta_rho'].size
    xi_rho_dim_size = dat_his.dimensions['xi_rho'].size

    u_dat = np.zeros((time_dim_size * saving_step, eta_rho_dim_size, xi_rho_dim_size))
    v_dat = np.zeros((time_dim_size * saving_step, eta_rho_dim_size, xi_rho_dim_size))
    temp_dat = np.zeros((time_dim_size * saving_step, eta_rho_dim_size, xi_rho_dim_size))

print(time_dim_size)


###########  need to figure out the surface depth

masked_value = -99999
t_ind=0
ocean_time=[]
save_ind=0
outfile = os.path.join(path_data, data_name % int(save_ind/saving_step))

for his_file in his_files:
    print(his_file)
    dat_his = nc.Dataset(his_file, 'r')

    for j in range(dat_his.dimensions['time'].size):
        print(t_ind, end=" ")
        u_dat[t_ind, :, :] = Forder(u2rho(ncload(dat_his, 'u', itime=j)[:,:,-1]))
        v_dat[t_ind, :, :] = Forder(v2rho(ncload(dat_his, 'v', itime=j)[:,:,-1]))
        temp_dat[t_ind, :, :] = dat_his.variables['temp'][j,-1,:,:]
        ocean_time.append(int(dat_his.variables['ocean_time'][j]))
        t_ind +=1

    print()
    dat_his.close()
    save_ind+=1

    if save_ind%saving_step == 0:
        print('SAVING: ', outfile, save_ind)

        wrtNcfile_2d(outfile, grd)
        wrtNcVars_2d(outfile, vardict=dict(ocean_time=np.array(ocean_time)), dim_names=('time',))
        wrtNcVars_2d(outfile, vardict=dict(lon_rho=lon_rho), dim_names=('eta_rho', 'xi_rho'))
        wrtNcVars_2d(outfile, vardict=dict(lat_rho=lat_rho), dim_names=('eta_rho', 'xi_rho'))
        wrtNcVars_2d(outfile, vardict=dict(u=u_dat, v=v_dat, temp=temp_dat), dim_names=('time', 'eta_rho', 'xi_rho'))

        # initializing for next iteration
        t_ind=0
        u_dat = np.zeros((time_dim_size * saving_step, eta_rho_dim_size, xi_rho_dim_size))
        v_dat = np.zeros((time_dim_size * saving_step, eta_rho_dim_size, xi_rho_dim_size))
        ocean_time=[]
        outfile = os.path.join(path_data, data_name % int(save_ind / saving_step))


# Save the last iteration if it was not saving_step
if save_ind % saving_step != 0:
    print('Saving: ', outfile, save_ind)
    wrtNcfile_2d(outfile, grd)
    wrtNcVars_2d(outfile, vardict=dict(ocean_time=np.array(ocean_time)), dim_names=('time',))
    wrtNcVars_2d(outfile, vardict=dict(lon_rho=lon_rho), dim_names=('eta_rho', 'xi_rho'))
    wrtNcVars_2d(outfile, vardict=dict(lat_rho=lat_rho), dim_names=('eta_rho', 'xi_rho'))
    wrtNcVars_2d(outfile, vardict=dict(u=u_dat, v=v_dat, temp=temp_dat), dim_names=('time', 'eta_rho', 'xi_rho'))

