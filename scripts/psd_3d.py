from simulation_parameters import *
from imports_file import *
from tools.get_depths import get_depths_run

# from R_tools_new_michal import zlevs, gridDict, Forder

### get history file names
his_files, tot_depths, time_dim = get_concatenate_parameters(depths ,min_num, max_num)

if time_jump > 1:
    time_step = int(np.floor(time_dim / time_jump))
else:
    time_step = time_dim
time_size = time_step * len(his_files)
print("Time parameters: ", time_size, time_dim, time_step, time_jump)

kh = freq_for_fft(len_xi_u, 2e3, N2=len_eta_v, D2=2e3) # D is given in meters
freq = freq_for_fft(time_size, time_jump) # D is given in meters

### save an empty psd file ###
dst_path = os.path.join(data_path_psd, "psd3d_xi_%d_%d_eta_%d_%d.nc" % (min_xi_u, max_xi_u, min_eta_v, max_eta_v))
print('Saving PSD into data file:', dst_path)
# if not os.path.exists(dst_path):
dat_dst = Dataset(dst_path, 'w')
dat_dst.createDimension('depths', len(tot_depths))
dat_dst.createVariable('depths', np.dtype('float32').char, ('depths',))
dat_dst.variables['depths'][:] = tot_depths
dat_dst.createDimension('kh', len(kh))
dat_dst.createVariable('kh', np.dtype('float32').char, ('kh',))
dat_dst.createDimension('freq', len(freq))
dat_dst.createVariable('freq', np.dtype('float32').char, ('freq',))
dat_dst.createVariable('psd', np.dtype('float32').char, ('depths','freq','kh'))
dat_dst.close()

if get_depths_run(sys.argv) is not None: # depths from outside bash script
    depths = get_depths_run(sys.argv)
if depths is None: # if depths is not given
    depths = tot_depths

### concatenate time to one series ###
for depth in depths:
    depth_ind = np.where(tot_depths == depth)[0][0]
    u = np.zeros((time_size, len_eta_v, len_xi_u))
    v = np.zeros((time_size, len_eta_v, len_xi_u))

    ind_time = 0
    for i in range(len(his_files)):
        his_file = his_files[i]
        print('Uploading variables: u and v from:', i, ind_time, ind_time+time_step, depth_ind, depth, his_file)
        sys.stdout.flush()
        dat_his = Dataset(his_file, 'r')
        try:
            if to_slice: # Shape: time, depth, y, x?e
                u_tmp=dat_his.variables['u'][::time_jump,depth_ind,min_eta_rho:max_eta_rho, min_xi_u:max_xi_u]
                v_tmp=dat_his.variables['v'][::time_jump,depth_ind,min_eta_v:max_eta_v, min_xi_rho:max_xi_rho]
            else:
                u_tmp=dat_his.variables['u'][::time_jump,depth_ind,:,:] # might be too slow with "::time_jump"
                v_tmp=dat_his.variables['v'][::time_jump,depth_ind,:,:]
            print('Changing coordinates from rho to u/v...')
            u_tmp = 0.5 * (u_tmp[:, 1:, :] + u_tmp[:, -1:, :])
            v_tmp = 0.5 * (v_tmp[:, :, 1:] + v_tmp[:, :, -1:])
            u[ind_time:(ind_time + time_step), :, :] = v_tmp
            v[ind_time:(ind_time + time_step), :, :] = u_tmp

        except ValueError:
            raise ValueError("Custom Error message: Make sure history file fits the slicing demands. "
                         "e.g. time_dim < time_jump, xi_dim <len_xi")
        dat_his.close()
        ind_time += time_step

        ### calculating PSD ###
    print('Calculating PSD... ')
    sys.stdout.flush()
    u = np.float32(u)
    u_tf = np.fft.fftn(u)/np.prod(u.shape)
    u_psd = np.real(u_tf * np.conjugate(u_tf))
    v = np.float32(v)
    v_tf = np.fft.fftn(v)/np.prod(v.shape)
    v_psd = np.real(v_tf * np.conjugate(v_tf))
    psd = u_psd+v_psd

    print('Calculating radial profile of psd... ')
    kh_array, psd_h = radial_profile(psd[:len(freq),:,:], len_eta_v, 2e3, len_xi_u, 2e3)
    print('Compare Kh dimensions: ', len(kh), len(kh_array))
    print('Compare freq dimensions: ', len(freq), int(np.floor(time_size/2+1)))

    print('Saving psd to dataset...')
    sys.stdout.flush()
    dat_dst = Dataset(dst_path, 'a')
    dat_dst.variables['psd'][depth_ind, :, :] = psd_h
    dat_dst.variables['kh'][:] = kh_array
    dat_dst.variables['freq'][:] = freq
    dat_dst.close()

print('DONE: saved psd to data file ', dst_path)