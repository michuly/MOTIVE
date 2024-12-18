from simulation_parameters import *
from imports_file import *

# from R_tools_new_michal import zlevs, gridDict, Forder
if socket.gethostname()=='southern' or socket.gethostname()=='atlantic.tau.ac.il':
    to_slice = True
    time_jump = 3
    min_eta_rho, max_eta_rho = 137, 360
    min_eta_v, max_eta_v = 137, 359
    min_xi_rho, max_xi_rho = 0, 2002
    min_xi_u, max_xi_u = 0, 2001
    if to_slice:
        len_eta_rho = max_eta_rho - min_eta_rho
        len_xi_rho = max_xi_rho - min_xi_rho
        len_eta_v = max_eta_v - min_eta_v
        len_xi_u = max_xi_u - min_xi_u

### get history file names
his_files, tot_depths, time_dim = get_concatenate_parameters(min_num, max_num)

if time_jump > 1:
    time_step = int(np.floor(time_dim / time_jump))
else:
    time_step = time_dim
time_size = time_step * len(his_files)
print("Time parameters: ", time_size, time_dim, time_step, time_jump)
kx = np.fft.fftfreq(len_xi_u, 2e3) # D is given in meters
freq = np.fft.fftfreq(time_size, time_jump) # D is given in meters
print('Check spatial dimensions: len_xi_u %d, min_xi %d, max_xi %d, kx %d, freq %d' % (len_xi_u, min_xi_u, max_xi_u,
                                                                                       len(kx), len(freq)))

### save an empty psd file ###
# dst_path_imag = os.path.join(data_path_psd, "fft_imag_zonal_freq_xi_%d_%d_eta_%d_%d.nc" % (min_xi_u, max_xi_u, min_eta_v, max_eta_v))
# dst_path_real = os.path.join(data_path_psd, "fft_real_zonal_freq_xi_%d_%d_eta_%d_%d.nc" % (min_xi_u, max_xi_u, min_eta_v, max_eta_v))
if to_slice:
    dst_path = os.path.join(data_path_psd, "psd_freq_kx_xi_%d_%d_eta_%d_%d.nc" % (min_xi_u, max_xi_u, min_eta_v, max_eta_v))
else:
    dst_path = os.path.join(data_path_psd, "psd_freq_kx.nc")

# print('Saving PSD into data file:', dst_path_imag)
# print('Saving PSD into data file:', dst_path_real)
print('Saving PSD into data file:', dst_path)
# for dst_path in [dst_path_real, dst_path_imag]:
# if not os.path.exists(dst_path):
dat_dst = Dataset(dst_path, 'w')
dat_dst.createDimension('depths', len(tot_depths))
dat_dst.createVariable('depths', np.dtype('float32').char, ('depths',))
dat_dst.variables['depths'][:] = tot_depths
dat_dst.createDimension('kx', len(kx))
dat_dst.createVariable('kx', np.dtype('float32').char, ('kx',))
dat_dst.createDimension('freq', len(freq))
dat_dst.createVariable('freq', np.dtype('float32').char, ('freq',))
dat_dst.createVariable('psd_u', np.dtype('float32').char, ('depths','freq','kx'))
dat_dst.createVariable('psd_v', np.dtype('float32').char, ('depths','freq','kx'))
dat_dst.close()

if get_depths_run(sys.argv, tot_depths) is not None: # depths from outside bash script
    depths = get_depths_run(sys.argv, tot_depths)
if depths is None: # if depths is not given
    depths = tot_depths

### concatenate time to one series ###
for depth in depths:
    depth_ind = np.where(tot_depths == depth)[0][0]
    v = np.zeros((time_size, len_eta_v, len_xi_u))
    u = np.zeros((time_size, len_eta_v, len_xi_u))

    ind_time = 0
    for i in range(len(his_files)):
        his_file = his_files[i]
        print('Uploading variables: u,v from:', i, ind_time, ind_time+time_step, depth_ind, depth, his_file)
        sys.stdout.flush()
        dat_his = Dataset(his_file, 'r')
        try:
            if to_slice: # Shape: time, depth, y, x?e
                v_tmp=dat_his.variables['v'][::time_jump,depth_ind,min_eta_v:max_eta_v, min_xi_rho:max_xi_rho]
                u_tmp=dat_his.variables['u'][::time_jump,depth_ind,min_eta_rho:max_eta_rho, min_xi_u:max_xi_u]
            else:
                v_tmp=dat_his.variables['v'][::time_jump,depth_ind,:,:] # might be too slow with "::time_jump"
                u_tmp=dat_his.variables['u'][::time_jump,depth_ind,:,:] # might be too slow with "::time_jump"
            print('Changing coordinates from rho to v...')
            v[ind_time:(ind_time + time_step), :, :] = 0.5 * (v_tmp[:, :, 1:] + v_tmp[:, :, -1:])
            u[ind_time:(ind_time + time_step), :, :] = 0.5 * (u_tmp[:, 1:, :] + u_tmp[:, -1:, :])

        except ValueError:
            raise ValueError("Custom Error message: Make sure history file fits the slicing demands. "
                         "e.g. time_dim < time_jump, xi_dim <len_xi")
        dat_his.close()
        ind_time += time_step

        ### calculating PSD ###
    print('Calculating PSD... ')
    sys.stdout.flush()
    v = np.float32(v)
    u = np.float32(u)
    u_tf = np.fft.fft2(u, axes=(0,2))/len_xi_u/time_size
    v_tf = np.fft.fft2(v, axes=(0,2))/len_xi_u/time_size
    psd_u = np.real(v_tf * np.conjugate(v_tf))
    psd_v = np.real(v_tf * np.conjugate(v_tf))

    print('Saving psd to dataset...')
    print('Check dimensions: ', psd_u.shape, psd_v.shape, len(kx), len(freq), len(tot_depths), 'to slice: ', to_slice)
    sys.stdout.flush()
    dat_dst = Dataset(dst_path, 'a')
    print('Check dimensions: netcdf shape', dat_dst.variables['psd_u'].shape, dat_dst.variables['psd_v'].shape)
    dat_dst.variables['psd_u'][depth_ind, :, :] = psd_u.mean(axis=1)
    dat_dst.variables['psd_v'][depth_ind, :, :] = psd_v.mean(axis=1)
    dat_dst.variables['kx'][:] = kx
    dat_dst.variables['freq'][:] = freq
    dat_dst.close()

    # print('Saving psd to dataset...')
    # sys.stdout.flush()
    # dat_dst = Dataset(dst_path_real, 'a')
    # dat_dst.variables['psd'][depth_ind, :, :] = np.real(u_tf).mean(axis=1)
    # dat_dst.variables['kx'][:] = kx
    # dat_dst.variables['freq'][:] = freq
    # dat_dst.close()

    # sys.stdout.flush()
    # dat_dst = Dataset(dst_path_imag, 'a')
    # dat_dst.variables['psd'][depth_ind, :, :] = np.imag(u_tf).mean(axis=1)
    # dat_dst.variables['kx'][:] = kx
    # dat_dst.variables['freq'][:] = freq
    # dat_dst.close()

print('DONE: saved psd to data file ', dst_path)
# print('DONE: saved psd to data file ', dst_path_real)
# print('DONE: saved psd to data file ', dst_path_imag)