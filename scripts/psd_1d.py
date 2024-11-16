from simulation_parameters import *
from imports_file import *
from concatenate_files import get_concatenate_parameters

# from R_tools_new_michal import zlevs, gridDict, Forder

### get history file names ###PYTHONPATH=/analysis/michalshaham/PythonProjects/MOTIVE/ python /analysis/michalshaham/PythonProjects/MOTIVE/tools/psd_1d.py
his_files, tot_depths, time_dim = get_concatenate_parameters(depths ,min_num, max_num)
if depths is None: # incase you do not want to calculate all depths
    depths=tot_depths

if time_jump > 1:
    time_step = int(np.floor(time_dim / time_jump))
else:
    time_step = time_dim
time_size = time_step * len(his_files)
print("Time parameters: ", time_size, time_dim, time_step, time_jump)

### save an empty psd file ###
dst_path = os.path.join(data_path_psd, "psd1d_xi_%d_%d_eta_%d_%d.nc" % (min_xi_u, max_xi_u, min_eta_v, max_eta_v))
print('Saving PSD into data file:', dst_path)
# if not os.path.exists(dst_path):
dat_dst = Dataset(dst_path, 'w')
dat_dst.createDimension('depths', len(depths))
dat_dst.createVariable('depths', np.dtype('float32').char, ('depths',))
dat_dst.variables['depths'][:] = depths
dat_dst.createDimension('freq', int(time_size / 2))
dat_dst.createVariable('freq', np.dtype('float32').char, ('freq',))
dat_dst.createVariable('psd', np.dtype('float32').char, ('depths','freq'))
dat_dst.close()


### concatenate time to one series ###
for depth_ind, depth in enumerate(depths):
    u = np.zeros((time_size, len_eta_rho, len_xi_u))
    v = np.zeros((time_size, len_eta_v, len_xi_rho))

    ind_time = 0
    for i in range(len(his_files)):
        his_file = his_files[i]
        print('Uploading variables: u and v from:', i, ind_time, ind_time+time_step, depth, his_file)
        sys.stdout.flush()
        dat_his = Dataset(his_file, 'r')
        try:
            if to_slice: # Shape: time, depth, y, x?e
                u[ind_time:(ind_time+time_step),:,:]=dat_his.variables['u'][::time_jump,depth_ind,min_eta_rho:max_eta_rho, min_xi_u:max_xi_u]
                v[ind_time:(ind_time+time_step),:,:]=dat_his.variables['v'][::time_jump,depth_ind,min_eta_v:max_eta_v, min_xi_rho:max_xi_rho]
            else:
                u[ind_time:(ind_time+time_step),:,:]=dat_his.variables['u'][::time_jump,depth_ind,:,:] # might be too slow with "::time_jump"
                v[ind_time:(ind_time+time_step),:,:]=dat_his.variables['v'][::time_jump,depth_ind,:,:]
        except ValueError:
            raise ValueError("Custom Error message: Make sure history file fits the slicing demands. "
                             "e.g. time_dim < time_jump, xi_dim <len_xi")
        ind_time+=time_step
        dat_his.close()

    ### calculating PSD ###
    print('Calculating PSD... ')
    sys.stdout.flush()
    u = np.float32(u)
    u_tf = np.fft.fft(u, axis=0)/time_size
    u_psd = np.mean(np.real(u_tf * np.conjugate(u_tf)), axis=(1, 2))
    v = np.float32(v)
    v_tf = np.fft.fft(v, axis=0)/time_size
    v_psd = np.mean(np.real(v_tf * np.conjugate(v_tf)), axis=(1, 2))
    freq=np.fft.fftfreq(time_size, 1)[:int(time_size / 2)]
    psd = u_psd[:int(time_size/2)]+v_psd[:int(time_size/2)]
    # plt.plot(f[:int(8881/2)],(u2_tf_tot[:int(8881/2)]+v2_tf_tot[:int(8881/2)])/452/681)

    print('Saving psd to dataset...')
    sys.stdout.flush()
    dat_dst = Dataset(dst_path, 'a')
    dat_dst.variables['psd'][depth_ind, :] = psd
    dat_dst.close()

dat_dst = Dataset(dst_path, 'a')
dat_dst.variables['freq'][:] = freq
dat_dst.close()

### plot psd ###
plt.plot(freq, psd)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('freq 1/hr')
plt.ylabel('PSD')
plt.title('Surface Horizontal PSD')
plt.grid(True)
plt.axvline(1/12,0,1e8, linestyle='--', c='k')
plt.axvline(1/24,0,1e8, linestyle='--', c='k')
plt.axvline(1/48,0,1e8, linestyle='--', c='k')
f_cor = 2 * 2 * np.pi / 24 * np.sin(np.deg2rad(35)) / 2 / np.pi
plt.axvline(f_cor,0,1e8, linestyle='--', c='k')
# plt.legend(['All freq forcing', 'Low freq forcing', '48$hr^{-1}$, 24$hr^{-1}$, $f_{cor}$, 12$hr^{-1}$'])
plt.show()

print('DONE: saved psd to data file ', dst_path)


