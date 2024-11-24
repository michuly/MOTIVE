"""
calculate density using Fortran tools
"""
from simulation_parameters import *
from imports_file import *
from R_tools_new_michal import gridDict, zlevs, rho_eos

"""
get two plots:
1. u at 0N 140W, depth vs. time
2. u at 104W, temporal average, detph vs. latitude"""
### get history file names
min_num, max_num = 141743-24*1, 141743+24*1
his_files, tot_depths, time_dim = get_concatenate_parameters(min_num, max_num, pattern_his_file=pattern_his_sigma)
depths = tot_depths
### save an empty psd file ###
dst_path_v = os.path.join(data_path_his, "rho.nc")
print('Saving rho into data file:', dst_path_v)

with Dataset(os.path.join(grd_path, grd_name)) as dat_grd:
    if to_slice:
        lon_array = dat_grd.variables['lon_rho'][lat_ind, min_xi_rho:max_xi_rho+1]
        lat_array = dat_grd.variables['lat_rho'][min_eta_rho:max_eta_rho+1, lon_ind]
    else:
        lon_array = dat_grd.variables['lon_rho'][lat_ind, :]
        lat_array = dat_grd.variables['lat_rho'][:, lon_ind]
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

    for j in range(dat_his.dimensions['time'].size):
        print(i, end=" ")
        z_r, z_w = zlevs(grd, dat_his, itime=j)

        print('Uploading variables: temp and salinity  from:', i, ind_time, (ind_time+time_step), his_file)
        sys.stdout.flush()
        dat_his = Dataset(his_file, 'r')
        if to_slice:  # Shape: time, depth, y, x?e
            temp = dat_his.variables['temp'][::time_jump, :, lat_ind_1N, min_xi_rho:max_xi_rho+1]
            slt = dat_his.variables['temp'][::time_jump, :, lat_ind_1N, min_xi_rho:max_xi_rho+1]
        else:
            temp = dat_his.variables['temp'][::time_jump, :, lat_ind_1N, :]
            slt = dat_his.variables['temp'][::time_jump, :, lat_ind_1N, :]


        print('Calculating density...')
        sys.stdout.flush()
        rho = rho_eos(T=temp, S=slt, z_r=z_r, z_w=z_w, rho0=dat_his.rho0)

        print('Inetpolating rho onto depths...')
        sys.stdout.flush()
        rho_mat[ind_time:(ind_time + time_step), :, :] = rho
        ocean_time[ind_time:(ind_time+time_step)] = dat_his.variables['ocean_time'][:]
        dat_his.close()
        ind_time = ind_time + time_step


print('Check dimensions: ', lon_array.shape, len_xi_rho, rho.shape)
sys.stdout.flush()

n_chunks = rho.shape[0] // 24
rho = rho[:n_chunks * 24, :, :]
rho=rho.reshape(-1, 24, rho.shape[1], rho.shape[2]).mean(axis=1)
ocean_time = ocean_time[:n_chunks * 24][::24]
print('Check dimensions: ', rho.shape, ocean_time.shape)

print('Saving 1N meridional velocity...')
sys.stdout.flush()
# if not os.path.exists(dst_path):
dat_dst = Dataset(dst_path_v, 'w')
dat_dst.createDimension('depths', len(tot_depths))
dat_dst.createVariable('depths', np.dtype('float32').char, ('depths',))
dat_dst.variables['depths'][:] = tot_depths
dat_dst.createDimension('lon', len_xi_rho)
dat_dst.createVariable('lon', np.dtype('float32').char, ('lon',))
dat_dst.variables['lon'][:] = lon_array
dat_dst.createDimension('lat', len_xi_rho)
dat_dst.createVariable('lat', np.dtype('float32').char, ('lat',))
dat_dst.variables['lat'][:] = lat_array
dat_dst.createDimension('ocean_time', len(ocean_time))
dat_dst.createVariable('ocean_time', np.dtype('float32').char, ('ocean_time',))
dat_dst.variables['ocean_time'][:] = ocean_time
dat_dst.createVariable('rho', np.dtype('float32').char, ('ocean_time','depths','lat','lon'))
dat_dst.variables['rho'][:] = rho
dat_dst.close()
print('DONE: saved rho to data file ', dst_path_v)
sys.stdout.flush()
