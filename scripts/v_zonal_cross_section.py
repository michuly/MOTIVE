import numpy as np

from simulation_parameters import *
from imports_file import *

"""
get two plots:
1. u at 0N 140W, depth vs. time
2. u at 104W, temporal average, detph vs. latitude"""
### get history file names
his_files, tot_depths, time_dim = get_concatenate_parameters(min_num, max_num)
depths = tot_depths
### save an empty psd file ###
dst_path_v = os.path.join(data_path_his, "v_1N_daily_avg.nc")
print('Saving v into data file:', dst_path_v)

with Dataset(os.path.join(grd_path, grd_name)) as dat_grd:
    if to_slice:
        lon_array = dat_grd.variables['lon_rho'][lat_ind_1N, min_xi_rho:max_xi_rho+1]
    else:
        lon_array = dat_grd.variables['lon_rho'][lat_ind_1N, :]

### concatenate time to one series ###
min_num, max_num = 141743, 141743
time_step = 12
ind_time = 0
time_size = time_step * len(his_files)
v = np.zeros((time_size, 88, len_xi_rho))
v.fill(np.nan)
ocean_time = np.zeros(time_size)
ocean_time.fill(np.nan)
for i in range(len(his_files)):
    his_file = his_files[i]
    print('Uploading variables: u  from:', i, ind_time, (ind_time+time_step), his_file)
    sys.stdout.flush()
    dat_his = Dataset(his_file, 'r')
    if to_slice:  # Shape: time, depth, y, x?e
        v[ind_time:(ind_time + time_step), :, :] = dat_his.variables['v'][:, :, lat_ind_1N, min_xi_rho:max_xi_rho+1]
    else:
        v[ind_time:(ind_time + time_step), :, :] = dat_his.variables['v'][:, :, lat_ind_1N, :]
    ocean_time[ind_time:(ind_time+time_step)] = dat_his.variables['ocean_time'][:]
    dat_his.close()
    ind_time = ind_time + time_step

print('Calculating gradient...')
dvz = np.gradient(v,depths,axis=1)
print('Check dimensions: ', lon_array.shape, len_xi_rho, v.shape)

print('Saving 1N meridional velocity...')
# if not os.path.exists(dst_path):
dat_dst = Dataset(dst_path_v, 'w')
dat_dst.createDimension('depths', len(tot_depths))
dat_dst.createVariable('depths', np.dtype('float32').char, ('depths',))
dat_dst.variables['depths'][:] = tot_depths
dat_dst.createDimension('lon', len_xi_rho)
dat_dst.createVariable('lon', np.dtype('float32').char, ('lon',))
dat_dst.variables['lon'][:] = lon_array
dat_dst.createDimension('ocean_time', time_step)
dat_dst.createVariable('ocean_time', np.dtype('float32').char, ('ocean_time',))
dat_dst.variables['ocean_time'][:] = ocean_time
dat_dst.createVariable('v', np.dtype('float32').char, ('ocean_time','depths','lon'))
dat_dst.variables['v'][:] = v
dat_dst.createVariable('dvz', np.dtype('float32').char, ('ocean_time','depths','lon'))
dat_dst.variables['dvz'][:] = dvz
dat_dst.close()
print('DONE: saved v to data file ', dst_path_v)

