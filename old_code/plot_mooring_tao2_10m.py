import sys
sys.path.append('/analysis/michalshaham/CrocoTools/Python_Kau/')
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from Motive.simulatiion_moorings import get_simulation_mooring, get_tao, get_tao_temp

######## get_real_mooring ###########
depth_ind =4
dt_real = 60*60  # sec
dt_real_T = 10*60  # sec
TAO_file = '/atlantic3/michalshaham/EPAC2km/TAO_140W.nc'
Nt_real, depths, tao2_data_real = get_tao(TAO_file, depth_ind)
TAO_file = '/atlantic3/michalshaham/EPAC2km/TAO_140W_SST.nc'
Nt_real_T, tao2_T_real = get_tao_temp(TAO_file)


######## get_simulation_mooring ###########
dt = 10*60  # sec
depth_ind =7
path_mooring = '/atlantic3/michalshaham/EPAC2km/OUTPUT/EXT2'
mooring_name = 'z_tao2'
Nt, tao2_data = get_simulation_mooring(path_mooring, mooring_name, depth_ind)

path_mooring = '/atlantic3/michalshaham/PACHUG/mooring_data2'
Nt_low, tao2_data_low = get_simulation_mooring(path_mooring, mooring_name, depth_ind)

######## Calculate the power spectra ###########
# Calculate power spectral density using Welch's method
fs = 1 / dt
fs_real = 1 / dt_real
nperseg = int(14 * 24 * 60 * 60 / dt)  # 2 weeks
noverlap=nperseg*1/2
nperseg_real = int(14 * 24 * 60 * 60 / dt_real)  # 2 weeks
noverlap_real=nperseg_real*1/2

# tao2_data = data_processing(tao2_data, axis=1)
# tao2_data_low = data_processing(tao2_data_low, axis=1)
# tao2_data_real = data_processing(tao2_data_real, axis=1)
# tao2_T_real = data_processing(tao2_T_real, axis=1)

print(tao2_data.shape, tao2_data_low.shape, tao2_data_real.shape, tao2_T_real.shape)

real_ind = np.max(np.where(np.isnan(tao2_data_real[1,:])))
frequencies3, PSD_v_tao2 = welch(tao2_data[1,:], fs=fs, nperseg=nperseg, noverlap=noverlap, axis=0)
frequencies3, PSD_u_tao2 = welch(tao2_data[0,:], fs=fs, nperseg=nperseg, noverlap=noverlap, axis=0)
frequencies_low, PSD_v_tao2_low = welch(tao2_data_low[1,:], fs=fs, nperseg=nperseg, noverlap=noverlap, axis=0)
frequencies_low, PSD_u_tao2_low = welch(tao2_data_low[0,:], fs=fs, nperseg=nperseg, noverlap=noverlap, axis=0)
frequencies_real, PSD_v_tao2_real = welch(tao2_data_real[1,real_ind+1:], fs=fs_real, nperseg=nperseg_real, noverlap=noverlap_real, axis=0)
frequencies_real, PSD_u_tao2_real = welch(tao2_data_real[0,real_ind+1:], fs=fs_real, nperseg=nperseg_real, noverlap=noverlap_real, axis=0)


fs_real_T = 1 / dt_real_T
nperseg = int(14 * 24 * 60 * 60 / dt)  # 2 weeks
noverlap=nperseg*1/2
nperseg_real = int(14 * 24 * 60 * 60 / dt_real_T)  # 2 weeks
noverlap_real=nperseg_real*1/2

frequencies3_T, PSD_T_tao2 = welch(tao2_data[2,3866:], fs=fs, nperseg=nperseg, noverlap=noverlap, axis=0)
frequencies_low_T, PSD_T_tao2_low = welch(tao2_data_low[2,3866:], fs=fs, nperseg=nperseg, noverlap=noverlap, axis=0)
frequencies_real_T, PSD_T_tao2_real = welch(tao2_T_real[0], fs=fs_real_T, nperseg=nperseg_real, noverlap=noverlap_real, axis=0)

######## Plot the power spectra ###########
# plt.figure(figsize=(10, 6))
plotting=True
if plotting:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvline(x=1/(24*60*60), color='k', linestyle='--', label='diurnal')
    ax.axvline(x=1/(24*60*60/2), color='k', linestyle='--', label='semi-diurnal')
    # ax.axvline(x=1/(24*60*60*10), color='k', linestyle='--', label='10dys')

    # ax.loglog(frequencies3, PSD_u_tao2+PSD_v_tao2, label='tao2')
    # ax.semilogy(frequencies1, PSD_v_motive1, label='motive1')
    # ax.semilogy(frequencies2, PSD_v_motive2, label='motive2')
    ax.loglog(frequencies3, PSD_u_tao2+PSD_v_tao2, label='tao2 (2km)')
    ax.loglog(frequencies_low, PSD_u_tao2_low+PSD_v_tao2_low, label='tao2 (6km)')
    ax.loglog(frequencies_real, PSD_u_tao2_real+PSD_v_tao2_real, label='tao2 (real)')
    # ax.loglog(freq, psd, label='tao2 - real')

    n = 11
    eps = 1 / np.sqrt(n)
    min = 1 / (1 + 2 * eps)
    max = 1 / (1 - 2 * eps)
    ax.plot([2e-6, 2e-6], [min * 1e2, max * 1e2], 'k')
    ax.set_ylim((1e-2,5e5))
    ax.set_xlim((None, 5e-4))
    ax.set_title('TAO2 PSD, z=-120m')
    # ax.set_xlabel('Time (Hr)')
    ax.set_xlabel('cycles per sec')
    ax.set_ylabel('PSD [(m/s)^2/cps]')
    ax.legend()
    ax.grid(True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvline(x=1/(24*60*60), color='k', linestyle='--', label='diurnal')
    ax.axvline(x=1/(24*60*60/2), color='k', linestyle='--', label='semi-diurnal')
    # ax.axvline(x=1/(24*60*60*10), color='k', linestyle='--', label='10dys')

    # ax.loglog(frequencies3, PSD_u_tao2+PSD_v_tao2, label='tao2')
    # ax.semilogy(frequencies1, PSD_v_motive1, label='motive1')
    # ax.semilogy(frequencies2, PSD_v_motive2, label='motive2')
    ax.loglog(frequencies3_T, PSD_T_tao2, label='tao2 (2km)')
    ax.loglog(frequencies_low_T, PSD_T_tao2_low, label='tao2 (6km)')
    ax.loglog(frequencies_real_T, PSD_T_tao2_real, label='tao2 (real)')
    # ax.loglog(freq, psd, label='tao2 - real')
    ax.plot([2e-6, 2e-6], [min * 1e2, max * 1e2], 'k')
    ax.set_ylim((1e-2,1e5))
    ax.set_title('Temperature PSD, Surface')
    # ax.set_xlabel('Time (Hr)')
    ax.set_xlabel('cycles per sec')
    ax.set_ylabel('PSD [C^2/cps]')
    ax.legend()
    ax.grid(True)


    plot_time = False
    if plot_time:

        # # Plot the time series for u
        # plt.figure(figsize=(10, 6))
        # plt.plot(np.arange(Nt) * dt / 60 / 60, tao2_data[0,:][0, :], label='tao2 (2km)')
        # plt.plot(np.arange(Nt_low) * dt / 60 / 60, tao2_data_low[0, :], label='tao2 (6km)')
        # plt.plot(np.arange(Nt_real_T) * dt / 60 / 60, tao2_data_real[0, :], label='tao2')
        #
        # plt.title('U time series, s_lev=%d' % depth_ind)
        # plt.xlabel('Time (hr)')
        # plt.ylabel('Vel [m/s]')
        # plt.legend()
        # plt.grid(True)

        # Plot the time series for v
        # plt.figure(figsize=(10, 6))
        # plt.plot(np.arange(Nt) * dt / 60 / 60, motive1_data[1, :], label='motive1')
        # plt.plot(np.arange(Nt) * dt / 60 / 60, motive2_data[1, :], label='motive2')
        # plt.plot(np.arange(Nt_low) * dt / 60 / 60, tao2_data_low[1, :], label='tao2')
        #
        # plt.title('V time series, s_lev=%d' % depth_ind)
        # plt.xlabel('Time (hr)')
        # plt.ylabel('Vel [m/s]')
        # plt.legend()
        # plt.grid(True)

        plt.figure(figsize=(10, 6))
        plt.plot((np.arange(Nt) * dt / 60 / 60)[3866:], tao2_data[2, 3866:], label='tao2 (2km)')
        plt.plot((np.arange(Nt_low) * dt / 60 / 60)[3866:], tao2_data_low[2, 3866:], label='tao2 (6km)')
        plt.plot((2320200+np.arange(Nt_real_T) * dt_real_T) / 60 / 60, tao2_T_real[0, :], label='tao2')

        plt.title('Temp time series, surface')
        plt.xlabel('Time (hr)')
        plt.ylabel('Temp [C]')
        plt.legend()
        plt.grid(True)

    plt.show()
