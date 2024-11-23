import numpy as np

from simulation_parameters import *
from imports_file import *

"""
get two plots:
1. u at 0N 140W, depth vs. time
2. u at 104W, temporal average, detph vs. latitude"""
### get history file names
min_num, max_num = 141743-24*1, 141743+24*1
his_files, tot_depths, time_dim = get_concatenate_parameters(min_num, max_num)
### save an empty psd file ###
dst_path = os.path.join(data_path_his, "SST_daily_avg.nc")
print('Saving v into data file:', dst_path)

with Dataset(os.path.join(grd_path, grd_name)) as dat_grd:
    if to_slice:
        lon_array = dat_grd.variables['lon_rho'][lat_ind, min_xi_rho:max_xi_rho+1]
        lat_array = dat_grd.variables['lon_rho'][min_eta_rho:max_eta_rho+1, lon_ind]
    else:
        lon_array = dat_grd.variables['lon_rho'][lat_ind_1N, :]
        lat_array = dat_grd.variables['lon_rho'][:, lon_ind]

### concatenate time to one series ###
time_jump = 1
if time_jump > 1:
    time_step = int(np.floor(time_dim / time_jump))
else:
    time_step = time_dim
time_size = time_step * len(his_files)
print("Time parameters: ", time_size, time_dim, time_step, time_jump)
ind_time = 0
sst = np.zeros((time_size, len_eta_rho, len_xi_rho))
sst.fill(np.nan)
ocean_time = np.zeros(time_size)
ocean_time.fill(np.nan)
for i in range(len(his_files)):
    his_file = his_files[i]
    print('Uploading variables: v  from:', i, ind_time, (ind_time+time_step), his_file)
    sys.stdout.flush()
    dat_his = Dataset(his_file, 'r')
    if to_slice:  # Shape: time, depth, y, x?e
        sst[ind_time:(ind_time + time_step), :, :] = dat_his.variables['temp'][::time_jump, 0, min_eta_rho:max_eta_rho + 1, min_xi_rho:max_xi_rho + 1]
    else:
        sst[ind_time:(ind_time + time_step), :, :] = dat_his.variables['temp'][::time_jump, 0, :, :]
    ocean_time[ind_time:(ind_time+time_step)] = dat_his.variables['ocean_time'][:]
    dat_his.close()
    ind_time = ind_time + time_step

print('Check dimensions: ', len_eta_rho, len_xi_rho, sst.shape)

n_chunks = sst.shape[0] // 24
sst = sst[:n_chunks * 24, :, :]
sst=sst.reshape(-1, 24, sst.shape[1], sst.shape[2]).mean(axis=1)
ocean_time = ocean_time[:n_chunks * 24][::24]
print('Check dimensions: ', sst.shape, ocean_time.shape)

print('Saving SST...')
sys.stdout.flush()
# if not os.path.exists(dst_path):
dat_dst = Dataset(dst_path, 'w')
dat_dst.createDimension('lat', len(lat_array))
dat_dst.createVariable('lat', np.dtype('float32').char, ('lat',))
dat_dst.variables['lat'][:] = lat_array
dat_dst.createDimension('lon', len_xi_rho)
dat_dst.createVariable('lon', np.dtype('float32').char, ('lon',))
dat_dst.variables['lon'][:] = lon_array
dat_dst.createDimension('ocean_time', len(ocean_time))
dat_dst.createVariable('ocean_time', np.dtype('float32').char, ('ocean_time',))
dat_dst.variables['ocean_time'][:] = ocean_time
dat_dst.createVariable('sst', np.dtype('float32').char, ('ocean_time','lat','lon'))
dat_dst.variables['sst'][:] = sst
dat_dst.close()
print('DONE: saved SST to data file ', dst_path)
sys.stdout.flush()
