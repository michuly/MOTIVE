from simulation_data import *
from imports_file import *

# from R_tools_new_michal import zlevs, gridDict, Forder

### get history file names ###PYTHONPATH=/analysis/michalshaham/PythonProjects/MOTIVE/ python /analysis/michalshaham/PythonProjects/MOTIVE/tools/psd_1d.py
min_num, max_num = 141035, 142463  # minimum and maximum dates of files to be analyzed
nums, his_files = get_file_list(data_path, pattern_his, num_len=6)
if max_num!=0:
    his_files = [his_files[i] for i in range(len(his_files)) if (nums[i] >= min_num and nums[i] <= max_num)]
print('Example for history file: ', his_files[-1])

### set time parameters ###
with Dataset(his_files[0], 'r') as dat_his:
    # print("What is up, what is down:")
    # print(dat_his.variables['u'][:, 0, :, :].mean(), dat_his.variables['u'][:, -1, :, :].mean())
    print("What are the shapes:")
    print(len_xi_rho, len_xi_u, len_eta_rho, len_eta_v)
    print(dat_his.variables['u'].shape, dat_his.variables['v'].shape)
    # print(dat_his.variables['depth'][0])
    time_dim = dat_his.dimensions['time'].size
    if depths is None:
        depths = dat_his.variables['depth'][:]
        depths = depths[depths>-800]
if time_jump>1:
    time_step = int(np.floor(time_dim / time_jump))
else:
    time_step = time_dim
time_size = time_step * len(his_files)
print('Time parameters: ', time_dim, time_jump, time_step, time_size)

### save an empty psd file ###
dst_path = os.path.join(data_path_psd1d, "psd1_test.nc")
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
    u_psd = np.zeros(time_size)
    v_psd = np.zeros(time_size)
    u = np.zeros((time_size, len_eta_rho, len_xi_u))
    v = np.zeros((time_size, len_eta_v, len_xi_rho))

    ind_time = 0
    for i in range(len(his_files)):
        his_file = his_files[i]
        print('Uploading variables: u and v from:', i, ind_time, ind_time+time_step, depth, his_file)
        dat_his = Dataset(his_file, 'r')
        if to_slice: # Shape: time, depth, y, x?
            u[ind_time:(ind_time+time_step),:,:]=dat_his.variables['u'][::time_jump,depth_ind,min_eta_rho:max_eta_rho, min_xi_u:max_xi_u]
            v[ind_time:(ind_time+time_step),:,:]=dat_his.variables['v'][::time_jump,depth_ind,min_eta_v:max_eta_v, min_xi_rho:max_xi_rho]
        else:
            u[ind_time:(ind_time+time_step),:,:]=dat_his.variables['u'][::time_jump,depth_ind,:,:] # might be too slow with "::time_jump"
            v[ind_time:(ind_time+time_step),:,:]=dat_his.variables['v'][::time_jump,depth_ind,:,:]
        ind_time+=time_step
        dat_his.close()

    ### calculating PSD ###
    print('Calculating PSD... ')
    u_psd = np.zeros(time_size)
    v_psd = np.zeros(time_size)
    u = np.float32(u)
    u_tf = np.fft.fft(u, axis=0)
    u_psd += np.sum(np.real(u_tf * np.conjugate(u_tf)), axis=(1, 2))
    v = np.float32(v)
    v_tf = np.fft.fft(v, axis=0)
    v_psd += np.sum(np.real(v_tf * np.conjugate(v_tf)), axis=(1, 2))
    freq=np.fft.fftfreq(time_size, 1)[:int(time_size / 2)]
    psd = u_psd[:int(time_size/2)]/len_xi_u/len_eta_rho+v_psd[:int(time_size/2)]/len_xi_rho/len_eta_v
    # plt.plot(f[:int(8881/2)],(u2_tf_tot[:int(8881/2)]+v2_tf_tot[:int(8881/2)])/452/681)

    print('Saving psd to dataset...')
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


