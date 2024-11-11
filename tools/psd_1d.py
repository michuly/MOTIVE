import sys
sys.path.append('/analysis/michalshaham/CrocoTools/Python_Kau/')
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
# import R_tools_new_goth as tN
from tools.get_file_list import get_file_list
from R_tools_new_michal import zlevs, gridDict, Forder

pattern_his = 'OUTPUT/his/EMed3km_his.?????.nc'
grd_name="INPUT/EMed3km_grd.nc"

def plot_psd(letter='A', ustr='u', vstr='v'):
    path = '/atlantic3/michalshaham/EMedCroco3km_%s/' % letter
    nums, his_files = get_file_list(path, pattern_his, num_len=5)
    his_files=[his_files[i] for i in range(len(his_files)) if (nums[i] >= 2920 and nums[i] <= 3256)]
    # his_files=[his_files[i] for i in range(len(his_files)) if (nums[i] >= 3276 and nums[i] <= 3756)]
    print(his_files[-1])
    with nc.Dataset(his_files[0], 'r') as dat_his:
        print(dat_his.variables['u'][:,0,:,:].mean(), dat_his.variables['u'][:,-1,:,:].mean())
        # print(dat_his.variables['depth'][0])
        time_dim = dat_his.dimensions['time']
        time_size = time_dim.size * len(his_files)
        time_size_file = time_dim.size

    psd_plot=True
    if psd_plot:
        # u_psd = np.zeros(int(np.ceil(time_dim_size/2)))
        u_psd = np.zeros(time_size)
        v_psd = np.zeros(time_size)

        enum = 0
        u = np.zeros((time_size, 452, 681))
        v = np.zeros((time_size, 451, 682))
        for i in range(int(time_size/time_size_file)):
            his_file = his_files[i]
            num=12
            print(i, enum, enum+num, ustr, vstr, his_file)
            dat_his = nc.Dataset(his_file, 'r')
            if 'str' in ustr:
                u[enum:(enum+num),:,:]=dat_his.variables[ustr][:,:,:]
                v[enum:(enum+num),:,:]=dat_his.variables[vstr][:,:,:]
            else:
                u[enum:(enum+num),:,:]=dat_his.variables[ustr][:,79,:,:]
                v[enum:(enum+num),:,:]=dat_his.variables[vstr][:,79,:,:]
            enum+=num
            dat_his.close()
        u = np.float32(u)
        u_tf = np.fft.fft(u, axis=0)
        u_psd += np.sum(np.real(u_tf * np.conjugate(u_tf)), axis=(1, 2))
        v = np.float32(v)
        v_tf = np.fft.fft(v, axis=0)
        v_psd += np.sum(np.real(v_tf * np.conjugate(v_tf)), axis=(1, 2))
        f=np.fft.fftfreq(time_size, 6)
        psd = u_psd[:int(time_size/2)]/452/681+v_psd[:int(time_size/2)]/451/682
        plt.plot(f[:int(time_size/2)],psd)
        # plt.plot(f[:int(8881/2)],(u2_tf_tot[:int(8881/2)]+v2_tf_tot[:int(8881/2)])/452/681)

        return u, v

u1, v1 = plot_psd('C', ustr='u', vstr='v')
u2, v2 = plot_psd('C', ustr='sustr', vstr='svstr')
u3, v3 = plot_psd('D', ustr='u', vstr='v')
u4, v4 = plot_psd('D', ustr='sustr', vstr='svstr')
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


tpas_plot=False
def tpas_plot(tim):
    dat_his = nc.Dataset(his_files[tim], 'r')
    hbl=np.mean(dat_his.variables['hbl'][:,:,:], axis=0)
    tpas=np.mean(dat_his.variables['tpas'][:,:,:,:], axis=0)
    grd = gridDict(path, grd_name, ij=None)
    (z_r, z_w) = zlevs(grd, dat_his, itime=0)
    dat_his.close()
    z_r=Forder(z_r)
    mask = Forder(grd['mask_rho'])[np.newaxis, :, :]

    plt.figure()
    plt.hist(z_r.flatten(), density=True, bins=200, range=[-500, 0], weights=tpas.flatten())
    plt.xlabel('Depth')
    plt.ylabel('Mean Concentration')
    plt.title('Passive tracers time=%ddays' % tim)
    plt.grid(True)
    plt.show()

    plt.figure()
    hbl2 = -hbl.flatten()
    hbl2 = hbl2[mask.flatten() == 1]
    plt.hist(hbl2.flatten(), density=True, bins=200, range=[-200, 0])
    plt.xlabel('hbl')
    plt.ylabel('concentration')
    plt.title('hbl at time=%ddays' % tim)
    plt.grid(True)
    plt.show()

    # bin_mean, bin_edges, binnumber = spstats.binned_statistic(z_r.flatten(), tpas.flatten(), statistic='mean', bins=140, range=[-140, 0])
    # plt.figure()
    # plt.hlines(bin_mean, bin_edges[:-1], bin_edges[1:], colors='g', lw=5, label='binned statistic of data')
    # plt.plot((bin_edges[:-1]+bin_edges[1:])/2, bin_mean)
    # plt.xlabel('Depth')
    # plt.ylabel('Mean Concentration')
    # plt.title('Mean Concentration vs Depth')
    # plt.grid(True)
    # plt.show()


