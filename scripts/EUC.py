from simulation_parameters import *
from imports_file import *

"""
get two plots:
1. u at 0N 140W, depth vs. time
2. u at 104W, temporal average, detph vs. latitude"""
### get history file names
his_files, tot_depths, time_dim = get_concatenate_parameters(depths ,min_num, max_num)
depths = tot_depths
### save an empty psd file ###
dst_path_tavg = os.path.join(data_path_his, "u_tavg_test_xi_%d_%d_eta_%d_%d.nc" % (min_xi_u, max_xi_u, min_eta_v, max_eta_v))
dst_path_0_140 = os.path.join(data_path_his, "u_0_140_test_xi_%d_%d_eta_%d_%d.nc" % (min_xi_u, max_xi_u, min_eta_v, max_eta_v))
print('Saving u into data file:', dst_path_tavg)
print('Saving u into data file:', dst_path_0_140)


with Dataset(os.path.join(grd_path, grd_name)) as dat_grd:
    if to_slice:
        lat_array = dat_grd.variables['lat_rho'][min_eta_rho:max_eta_rho+1, lon_ind]
    else:
        lat_array = dat_grd.variables['lat_rho'][:, lon_ind]

### concatenate time to one series ###
time_step = 12
ind_time = 0
time_size = time_step * len(his_files)
u_tavg = np.zeros((time_size, 88, len_eta_rho))
u_tavg.fill(np.nan)
u_0_140 = np.zeros((time_size, 88))
u_0_140.fill(np.nan)
ocean_time = np.zeros(time_size)
ocean_time.fill(np.nan)
for i in range(len(his_files)):
    his_file = his_files[i]
    print('Uploading variables: u  from:', i, ind_time, (ind_time+time_step), his_file)
    sys.stdout.flush()
    dat_his = Dataset(his_file, 'r')
    u_tavg[ind_time:(ind_time+time_step), :, :] = dat_his.variables['u'][:, :, :, lon_ind]
    u_0_140[ind_time:(ind_time+time_step), :] = dat_his.variables['u'][:, :, lat_ind, lon_ind]
    ocean_time[ind_time:(ind_time+time_step)] = dat_his.variables['ocean_time'][:]
    dat_his.close()
    ind_time = ind_time + time_step

print('Check dimensions: ', lat_array.shape, len_eta_rho, u_tavg.shape, u_0_140.shape)

print('Saving time average zonal velocity...')
# if not os.path.exists(dst_path):
dat_dst = Dataset(dst_path_tavg, 'w')
dat_dst.createDimension('depths', len(tot_depths))
dat_dst.createVariable('depths', np.dtype('float32').char, ('depths',))
dat_dst.variables['depths'][:] = tot_depths
dat_dst.createDimension('lat', len_eta_rho)
dat_dst.createVariable('lat', np.dtype('float32').char, ('lat',))
dat_dst.variables['lat'][:] = lat_array
dat_dst.createVariable('u', np.dtype('float32').char, ('depths','lat'))
dat_dst.variables['u'][:] = u_tavg.mean(axis=0)
dat_dst.close()
print('DONE: saved u to data file ', dst_path_tavg)

print('Saving 0N 140W zonal velocity...')
# if not os.path.exists(dst_path):
dat_dst = Dataset(dst_path_0_140, 'w')
dat_dst.createDimension('depths', len(tot_depths))
dat_dst.createVariable('depths', np.dtype('float32').char, ('depths',))
dat_dst.variables['depths'][:] = tot_depths
dat_dst.createDimension('ocean_time', time_size)
dat_dst.createVariable('ocean_time', np.dtype('float32').char, ('ocean_time',))
dat_dst.variables['ocean_time'][:] = ocean_time
dat_dst.createVariable('u', np.dtype('float32').char, ('ocean_time','depths'))
dat_dst.variables['u'][:] = u_0_140
dat_dst.close()
print('DONE: saved u to data file ', dst_path_0_140)

