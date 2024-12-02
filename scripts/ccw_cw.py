from scipy.fft import fft, fftfreq
from simulation_parameters import *
from imports_file import *


def calculate_psd(velocity_u, velocity_v, dt):
    """
    Calculate the power spectral density (PSD) for counterclockwise (CCW) and clockwise (CW)
    rotary components of a 4D velocity field (time, depth, latitude, longitude).

    Parameters:
        velocity_u (numpy.ndarray): 4D array of eastward velocity components (t, z, y, x).
        velocity_v (numpy.ndarray): 4D array of northward velocity components (t, z, y, x).
        dt (float): Time step in seconds.

    Returns:
        psd_ccw (numpy.ndarray): 4D array of PSD for CCW components (freq, z, y, x).
        psd_cw (numpy.ndarray): 4D array of PSD for CW components (freq, z, y, x).
        frequencies (numpy.ndarray): Array of frequency values corresponding to PSD.
    """
    # Dimensions of the data
    t_dim, z_dim, y_dim = velocity_u.shape

    # Combine u and v into a complex velocity field
    complex_velocity = velocity_u + 1j * velocity_v

    # Frequency array
    frequencies = fftfreq(t_dim, dt)[:t_dim // 2]

    print('Ffting...')
    sys.stdout.flush()
    fft_values = fft(complex_velocity, axis=0)/t_dim

    dst_path = os.path.join(data_path_psd, "psd_ccw_cw.nc")
    print('Saving PSD into data file:', dst_path)
    dat_dst = Dataset(dst_path, 'w')
    dat_dst.createDimension('depths', len(tot_depths))
    dat_dst.createVariable('depths', np.dtype('float32').char, ('depths',))
    dat_dst.variables['depths'][:] = tot_depths
    dat_dst.createDimension('freq', len(frequencies))
    dat_dst.createVariable('freq', np.dtype('float32').char, ('freq',))
    dat_dst.variables['freq'][:] = frequencies
    dat_dst.createDimension('lat', len(lat_array))
    dat_dst.createVariable('lat', np.dtype('float32').char, ('lat',))
    dat_dst.variables['lat'][:] = lat_array
    dat_dst.createVariable('psd_ccw', np.dtype('float32').char, ('freq', 'depths', 'lat'))
    dat_dst.createVariable('psd_cw', np.dtype('float32').char, ('freq', 'depths', 'lat'))

    print('Check dimensions: ', tot_depths.shape, lat_array.shape, frequencies.shape, fft_values.shape)
    sys.stdout.flush()
    dat_dst.variables['psd_ccw'][:] = np.abs(fft_values[:t_dim // 2,:,:]) ** 2
    dat_dst.variables['psd_cw'][:] = np.abs(fft_values[-(t_dim // 2):,:,:]) ** 2

    dat_dst.close()

    return psd_ccw, psd_cw, frequencies


def plot_psd(frequencies, psd_ccw, psd_cw, z_idx, y_idx, x_idx):
    """
    Plot the PSD for CCW and CW components at a specific depth and spatial location.

    Parameters:
        frequencies (numpy.ndarray): Array of frequency values corresponding to PSD.
        psd_ccw (numpy.ndarray): 4D array of PSD for CCW components (freq, z, y, x).
        psd_cw (numpy.ndarray): 4D array of PSD for CW components (freq, z, y, x).
        z_idx (int): Depth index for plotting.
        y_idx (int): Latitude index for plotting.
        x_idx (int): Longitude index for plotting.
    """
    plt.figure(figsize=(12, 6))

    # Plot CCW PSD
    plt.subplot(1, 2, 1)
    plt.plot(frequencies, psd_ccw[:, z_idx, y_idx, x_idx], label="CCW PSD")
    plt.title("Counterclockwise (CCW) PSD")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.grid(True)

    # Plot CW PSD
    plt.subplot(1, 2, 2)
    plt.plot(frequencies, psd_cw[:, z_idx, y_idx, x_idx], label="CW PSD", color='r')
    plt.title("Clockwise (CW) PSD")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":

    """
    get two plots:
    1. u at 0N 140W, depth vs. time
    2. u at 104W, temporal average, detph vs. latitude"""
    ### get history file names
    min_num, max_num = 141743 - 24 * 1, 141743 + 24 * 1
    # min_num, max_num = 141035, 143111
    his_files, tot_depths, time_dim = get_concatenate_parameters(min_num, max_num)
    depths = tot_depths
    ### save an empty psd file ###

    with Dataset(os.path.join(grd_path, grd_name)) as dat_grd:
        print('Options: ', dat_grd.variables.keys())
        lat_array = dat_grd.variables['lat_psi'][:, lon_ind]

    ### concatenate time to one series ###
    time_step = time_dim
    time_size = time_step * len(his_files)
    print("Time parameters: ", time_size, time_dim, time_step)
    ind_time = 0
    v = np.zeros((time_size, 88, len_eta_v))
    v.fill(np.nan)
    u = np.zeros((time_size, 88, len_eta_v))
    u.fill(np.nan)
    ocean_time = np.zeros(time_size)
    ocean_time.fill(np.nan)
    for i in range(len(his_files)):
        his_file = his_files[i]
        print('Uploading variables: v  from:', i, ind_time, (ind_time + time_step), his_file)
        sys.stdout.flush()
        dat_his = Dataset(his_file, 'r')
        u_tmp = dat_his.variables['u'][:, :, :, lon_ind]
        print('Check dimensions: ', tot_depths.shape, ocean_time.shape, lat_array.shape, u_tmp.shape, len_eta_v, v.shape, u.shape)
        sys.stdout.flush()
        u[ind_time:(ind_time + time_step), :, :] = 0.5 * (u_tmp[:, :, 1:] + u_tmp[:, :, -1:])
        v[ind_time:(ind_time + time_step), :, :] = dat_his.variables['v'][:, :, :, lon_ind]
        ocean_time[ind_time:(ind_time + time_step)] = dat_his.variables['ocean_time'][:]
        dat_his.close()
        ind_time = ind_time + time_step

    # Calculate PSD
    sys.stdout.flush()
    psd_ccw, psd_cw, frequencies = calculate_psd(u, v, dt=1)

    # Plot PSD at a specific depth and spatial location
    plot_psd(frequencies, psd_ccw, psd_cw)
