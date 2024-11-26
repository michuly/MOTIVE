"""
calculate density using Fortran tools
"""
import sys
sys.path.append('/analysis/michalshaham/CrocoTools/Python_Kau/')
from simulation_parameters import *
from imports_file import *
from R_tools_new_michal import gridDict, zlevs, rho_eos

"""
get two plots:
1. u at 0N 140W, depth vs. time
2. u at 104W, temporal average, detph vs. latitude"""
### get history file names
min_num, max_num = 141743-24*1, 141743+24*1
his_files, _, time_dim = get_concatenate_parameters(min_num, max_num, pattern_his_file=pattern_his_sigma)
### save an empty psd file ###
dst_path = os.path.join(data_path_his, "rho_2N.nc")
print('Saving rho into data file:', dst_path)

with Dataset(os.path.join(grd_path, grd_name)) as dat_grd:
    if to_slice:
        lon_array = dat_grd.variables['lon_rho'][lat_ind_2N, min_xi_rho:max_xi_rho+1]
        # lat_array = dat_grd.variables['lat_rho'][min_eta_rho:max_eta_rho+1, lon_ind]
    else:
        lon_array = dat_grd.variables['lon_rho'][lat_ind_2N, :]
        # lat_array = dat_grd.variables['lat_rho'][:, lon_ind]
grd = gridDict(grd_path, grd_name, ij=None)

### concatenate time to one series ###
time_jump = 1
if time_jump > 1:
    time_step = int(np.floor(time_dim / time_jump))
else:
    time_step = time_dim
time_size = time_step * len(his_files)
print("Time parameters: ", time_size, time_dim, time_step, time_jump)
ind_time = 0
rho_mat = np.zeros((time_size, 88, len_xi_rho))
rho_mat.fill(np.nan)
ocean_time = np.zeros(time_size)
ocean_time.fill(np.nan)
for i in range(len(his_files)):
    his_file = his_files[i]
    dat_his = Dataset(his_file, 'r')
    print('Uploading variables: temp and salinity  from:', i, ind_time, (ind_time + time_step), his_file)
    for j, his_ind  in enumerate(np.arange(dat_his.dimensions['time'].size)[::time_jump]):

        print('Uploading variables: temp and salinity from:', j, ind_time+j, his_ind)
        z_r, z_w = zlevs(grd, dat_his, itime=his_ind)
        z_w=(z_w[:,:,1:]+z_w[:,:,:-1])/2
        print(z_r.shape, z_r[0,0,0], z_w.shape)
        sys.stdout.flush()
        if to_slice:  # Shape: time, depth, y, x?e
            temp = dat_his.variables['temp'][his_ind, :, :, min_xi_rho:max_xi_rho+1]
            salt = dat_his.variables['salt'][his_ind, :, :, min_xi_rho:max_xi_rho+1]
        else:
            temp = dat_his.variables['temp'][his_ind, :, :, :]
            salt = dat_his.variables['salt'][his_ind, :, :, :]
        print(salt.shape, temp.shape)

        print('Calculating density...')
        sys.stdout.flush()
        rho = rho_eos(T=temp, S=salt, z_r=z_r.transpose(), z_w=z_w.transpose(), rho0=dat_his.rho0)
        print('Mean and std rho:', rho.mean(), rho.std())
        print('Inetpolating rho onto depths... Shapes:', rho.shape)
        sys.stdout.flush()
        rho = linear_interp(rho.transpose(), z_r, tot_depths).transpose()
        rho_mat[ind_time+j, :, :] = rho
        ocean_time[ind_time+j] = dat_his.variables['ocean_time'][j]
    dat_his.close()
    print('Check (j-1)==time_step: ', j, time_step)
    ind_time = ind_time + time_step

print('Check dimensions: ', lon_array.shape, len_xi_rho, rho.shape)
sys.stdout.flush()

# n_chunks = rho.shape[0] // 24
# rho = rho[:n_chunks * 24, :, :]
# rho=rho.reshape(-1, 24, rho.shape[1], rho.shape[2]).mean(axis=1)
# ocean_time = ocean_time[:n_chunks * 24][::24]
print('Check dimensions: ', rho.shape, ocean_time.shape)

print('Saving 1N meridional velocity...')
sys.stdout.flush()
# if not os.path.exists(dst_path):
dat_dst = Dataset(dst_path, 'w')
dat_dst.createDimension('depths', len(tot_depths))
dat_dst.createVariable('depths', np.dtype('float32').char, ('depths',))
dat_dst.variables['depths'][:] = tot_depths
dat_dst.createDimension('lon', len_xi_rho)
dat_dst.createVariable('lon', np.dtype('float32').char, ('lon',))
dat_dst.variables['lon'][:] = lon_array
dat_dst.createDimension('ocean_time', len(ocean_time))
dat_dst.createVariable('ocean_time', np.dtype('float32').char, ('ocean_time',))
dat_dst.variables['ocean_time'][:] = ocean_time
dat_dst.createVariable('rho', np.dtype('float32').char, ('ocean_time','depths','lon'))
dat_dst.variables['rho'][:] = rho
dat_dst.close()
print('DONE: saved rho to data file ', dst_path)
sys.stdout.flush()
