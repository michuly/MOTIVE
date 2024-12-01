"""
calculate w' - frequency higher than 15days(?) and remove barotropic w (depth averaged)
and adds density
"""
from simulation_parameters import *
from imports_file import *

### get history file names
min_num, max_num = 141743-24*100, 141743+24*100
his_files, tot_depths, time_dim = get_concatenate_parameters(min_num, max_num, pattern_his_file=pattern_his_2N)
depths = tot_depths
### save an empty psd file ###
dst_path_w = os.path.join(data_path_his, "w_prime.nc")
print('Saving w into data file:', dst_path_w)

with Dataset(os.path.join(grd_path, grd_name)) as dat_grd:
    if to_slice:
        lon_array = dat_grd.variables['lon_rho'][lat_ind_2N, min_xi_rho:max_xi_rho+1]
    else:
        lon_array = dat_grd.variables['lon_rho'][lat_ind_2N, :]

### concatenate time to one series ###
time_jump = 1
if time_jump > 1:
    time_step = int(np.floor(time_dim / time_jump))
else:
    time_step = time_dim
time_size = time_step * len(his_files)
print("Time parameters: ", time_size, time_dim, time_step, time_jump)
ind_time = 0
w = np.zeros((time_size, 88, len_xi_rho))
w.fill(np.nan)
rho1 = np.zeros((time_size, 88, len_xi_rho))
rho1.fill(np.nan)
ocean_time = np.zeros(time_size)
ocean_time.fill(np.nan)
for i in range(len(his_files)):
    his_file = his_files[i]
    print('Uploading variables: w  from:', i, ind_time, (ind_time+time_step), his_file)
    sys.stdout.flush()
    dat_his = Dataset(his_file, 'r')
    if to_slice:  # Shape: time, depth, y, x?e
        w[ind_time:(ind_time + time_step), :, :] = dat_his.variables['w'][::time_jump, :, 20, min_xi_rho:max_xi_rho+1]
        rho1[ind_time:(ind_time + time_step), :, :] = dat_his.variables['rho'][::time_jump, :, 20, min_xi_rho:max_xi_rho+1]
    else:
        w[ind_time:(ind_time + time_step), :, :] = dat_his.variables['w'][::time_jump, :, 20, :]
        rho1[ind_time:(ind_time + time_step), :, :] = dat_his.variables['rho'][::time_jump, :, 20, :]
    ocean_time[ind_time:(ind_time+time_step)] = dat_his.variables['ocean_time'][:]
    dat_his.close()
    ind_time = ind_time + time_step

print('Calculating gradient and averages...')
sys.stdout.flush()
w_baro=np.mean(w, axis=1)
print('Check dimensions: ', lon_array.shape, len_xi_rho, w.shape, w_baro.shape)
sys.stdout.flush()

w = w-butter_sos2_filter(w, filter_width=24*15, dt=1, axis=0, filter_order=6)-w_baro[:,np.newaxis,:]

print('Saving 1N vertical velocity...')
sys.stdout.flush()
# if not os.path.exists(dst_path):
dat_dst = Dataset(dst_path_w, 'w')
dat_dst.createDimension('depths', len(tot_depths))
dat_dst.createVariable('depths', np.dtype('float32').char, ('depths',))
dat_dst.variables['depths'][:] = tot_depths
dat_dst.createDimension('lon', len_xi_rho)
dat_dst.createVariable('lon', np.dtype('float32').char, ('lon',))
dat_dst.variables['lon'][:] = lon_array
dat_dst.createDimension('ocean_time', len(ocean_time))
dat_dst.createVariable('ocean_time', np.dtype('float32').char, ('ocean_time',))
dat_dst.variables['ocean_time'][:] = ocean_time
dat_dst.createVariable('w', np.dtype('float32').char, ('ocean_time','depths','lon'))
dat_dst.variables['w'][:] = w
dat_dst.createVariable('rho1', np.dtype('float32').char, ('ocean_time','depths','lon'))
dat_dst.variables['rho1'][:] = rho1
dat_dst.close()
print('DONE: saved w to data file ', dst_path_w)
sys.stdout.flush()
