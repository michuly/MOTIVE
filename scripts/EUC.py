from simulation_parameters import *
from imports_file import *

"""
get two plots:
1. u at 0N 140W, depth vs. time
2. u at 104W, temporal average, detph vs. latitude"""
### get history file names
his_files, tot_depths, time_dim = get_concatenate_parameters(min_num=0, max_num=0)
# his_files, tot_depths, time_dim = get_concatenate_parameters(min_num=141035, max_num=141060)
depths = tot_depths
### save an empty psd file ###
dst_path_tavg = os.path.join(data_path_his, "u_v_140W_tavg.nc")
dst_path_0_140 = os.path.join(data_path_his, "u_v_0N_140W.nc")
print('Saving u into data file:', dst_path_tavg)
print('Saving u into data file:', dst_path_0_140)


with Dataset(os.path.join(grd_path, grd_name)) as dat_grd:
    lat_psi = dat_grd.variables['lat_psi'][:, lon_ind]
    lat_rho = dat_grd.variables['lat_rho'][:, lon_ind]

### concatenate time to one series ###
time_step = 12
ind_time = 0
time_size = time_step * len(his_files)
u_tavg = np.zeros((time_size, 88, len(lat_rho)))
u_tavg.fill(np.nan)
u_0_140 = np.zeros((time_size, 88))
u_0_140.fill(np.nan)
v_tavg = np.zeros((time_size, 88, len(lat_psi)))
v_tavg.fill(np.nan)
v_0_140 = np.zeros((time_size, 88))
v_0_140.fill(np.nan)
ocean_time = np.zeros(time_size)
ocean_time.fill(np.nan)
print('Check dimensions: ', len(ocean_time), len(his_files))
for i in range(len(his_files)):
    his_file = his_files[i]
    print('Uploading variables: u, v  from:', i, ind_time, (ind_time+time_step), his_file)
    sys.stdout.flush()
    dat_his = Dataset(his_file, 'r')
    u_tavg[ind_time:(ind_time + time_step), :, :]= dat_his.variables['u'][:, :, :, lon_ind]
    v_tavg[ind_time:(ind_time + time_step), :, :] = dat_his.variables['v'][:, :, :, lon_ind]
    u_0_140[ind_time:(ind_time + time_step), :] = dat_his.variables['u'][:, :, lat_ind, lon_ind]
    v_0_140[ind_time:(ind_time + time_step), :] = dat_his.variables['v'][:, :, lat_ind, lon_ind]
    ocean_time[ind_time:(ind_time+time_step)] = dat_his.variables['ocean_time'][:]
    dat_his.close()
    ind_time = ind_time + time_step

print('Check dimensions: ', lat_rho.shape, lat_psi.shape, u_tavg.shape, v_tavg.shape, u_0_140.shape)

print('Saving time average zonal velocity...')
# if not os.path.exists(dst_path):
dat_dst = Dataset(dst_path_tavg, 'w')
dat_dst.createDimension('depths', len(tot_depths))
dat_dst.createVariable('depths', np.dtype('float32').char, ('depths',))
dat_dst.variables['depths'][:] = tot_depths
dat_dst.createDimension('lat_rho', len(lat_rho))
dat_dst.createVariable('lat_rho', np.dtype('float32').char, ('lat_rho',))
dat_dst.variables['lat_rho'][:] = lat_rho
dat_dst.createDimension('lat_psi', len(lat_psi))
dat_dst.createVariable('lat_psi', np.dtype('float32').char, ('lat_psi',))
dat_dst.variables['lat_psi'][:] = lat_psi
dat_dst.createVariable('u', np.dtype('float32').char, ('depths','lat_rho'))
dat_dst.variables['u'][:] = u_tavg.mean(axis=0)
dat_dst.createVariable('v', np.dtype('float32').char, ('depths','lat_psi'))
dat_dst.variables['v'][:] = v_tavg.mean(axis=0)
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
dat_dst.createVariable('v', np.dtype('float32').char, ('ocean_time','depths'))
dat_dst.variables['v'][:] = v_0_140
dat_dst.close()
print('DONE: saved u to data file ', dst_path_0_140)

