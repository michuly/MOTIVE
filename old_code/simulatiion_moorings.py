import sys
sys.path.append('/analysis/michalshaham/CrocoTools/Python_Kau/')
import numpy as np
from tools.get_file_list import get_file_list
from tools.find_lon_lat import find_lon_lat
import netCDF4 as nc
from scipy.signal import savgol_filter
from R_tools_new_michal import gridDict, wrtNcfile_z_lev, mooring_zlevs
# from scipy.signal import filtfilt, butter, decimate

"""
a. do spectrum analysis on the time dimension V
b. build a large matrix of all times. V
c. interpolate it to z_level - through python or linux tool?
pros and cons:
1. linux tool in one time, but determines the z_lev ahead
2. python will calculate it everytime, but the z_lev can be chosen
d.     for each time there is a different lenth of time series, but all files will have the same time length at that time 
    step. I need to check if dt is different, or just Nt.
e. plot where are the moorings
f. do it for real moorings
g. plot time series V
h. history file have time step of 1hr, so how z_lev are to be used? do z_lev change alot in 1hr?
i. build z_lev for all moorings
"""

dt_0= 10 * 60  # in sec
path_his = '/southern/rbarkan/data/EPAC2km/OUTPUT/HIS/'
path_grd = '/southern/rbarkan/data/EPAC2km'
path_mooring = '/southern/rbarkan/data/EPAC2km/OUTPUT/EXT/'
path_mooring_zlev = '/atlantic3/michalshaham/EPAC2km/OUTPUT/EXT'
pattern_his = 'EPAC2km_his.??????.nc'
grd_name="Epac2km_grd.nc"
# pattern_tao2 = 'tao2.*.nc'
pattern_mooring = '%s.*.nc'
pattern_mooring_zlev = 'z_%s.*.nc'
mooring = dict(tao1=(-110,0), tao3=(-170,0),tao2=(-140,0),motive1=(-140,0.5),motive2=(-140,1.75),motive3=(-140,3))

#########################################
# mooring data processing
#######################################3

def data_processing(original_data, axis=1):

    return original_data

    # Assuming 'data' is your dataset
    window_size = 5  # Must be an odd number
    degree = 2  # Degree of the polynomial to fit
    smoothed_data = savgol_filter(original_data, window_size, degree, axis=axis)

    # smoothing
    # window_size = 8  # Adjust this according to your needs
    # smoothed_data = np.apply_along_axis(lambda x: np.convolve(x, np.ones(window_size) / window_size, mode='same'),
    #                                     axis=axis, arr=original_data)
    # # Design a low-pass Butterworth filter
    # b, a = butter(N=6, Wn=0.2, btype='low')
    # # Apply the filter along the time dimension
    # filtered_data = np.apply_along_axis(lambda x: filtfilt(b, a, x), axis=axis, arr=smoothed_data)

    return smoothed_data


######## Build the data matrices ###########

def get_simulation_mooring(path_mooring, mooring_name, depth_ind):
    nums, mooring_files = get_file_list(path_mooring, pattern_mooring % mooring_name)

    with nc.Dataset(mooring_files[0], 'r') as dat_mooring:
        Nt = dat_mooring.dimensions['time'].size
        t_len = (len(mooring_files)+1) * Nt

    mooring_data = np.zeros((3,t_len))
    mooring_data.fill(np.nan)
    print(mooring_data.shape)
    i, N_tot = 0, 0
    for mooring_file in mooring_files:
        """
        for each time there is a different lenth of time series, but all files will have the same time length at that time 
        step. I need to check if dt is different, or just Nt.
        """
        dat_mooring = nc.Dataset(mooring_file, 'r')
        Nt = dat_mooring.dimensions['time'].size
        if len(dat_mooring.dimensions)==2:
            mooring_data[0, i:i+Nt]=dat_mooring.variables['u'][:, depth_ind]
            mooring_data[1, i:i+Nt]=dat_mooring.variables['v'][:, depth_ind]
            mooring_data[2, i:i+Nt]=dat_mooring.variables['temp'][:, depth_ind]
        elif len(dat_mooring.dimensions)==3:
            mooring_data[0, i:i+Nt]=dat_mooring.variables['u'][:, depth_ind, 0]
            mooring_data[1, i:i+Nt]=dat_mooring.variables['v'][:, depth_ind, 0]
            mooring_data[2, i:i+Nt]=dat_mooring.variables['temp'][:, depth_ind, 0]
        dat_mooring.close()

        i = i+Nt
        N_tot+=Nt

    mooring_data= mooring_data[:, :N_tot]
    return N_tot, mooring_data


##############################333
# interpolate mooring
#################################3

def interpolate_mooring(path_mooring, mooring_name, path_grd, grd_name):
    global grd, hc, Cs_r, Cs_w, zeta
    grd = gridDict(path_grd, grd_name, ij=None)
    nums, his_files = get_file_list(path_his, pattern_his)
    lon_i, lat_i = find_lon_lat(path_grd, grd_name, mooring[mooring_name])
    with nc.Dataset(his_files[0], 'r') as nch:
        hc = nch.hc
        Cs_r = nch.Cs_r
        Cs_w = nch.Cs_w

    z_interp = [0, -1, -2, -10, -25, -45, -80, -120]
    # nums, mooring_files = get_file_list(path_mooring, '%s.*.nc' % mooring_name, num_pattern=r'\b(\d{14})\b')
    nums, mooring_files = get_file_list(path_mooring, '%s.*.nc' % mooring_name, num_pattern=r'\b(\d{6})\b')
    for num, mooring_file in zip(nums, mooring_files):
        print(mooring_file)
        dat_mooring = nc.Dataset(mooring_file, 'r')
        u = dat_mooring.variables['u'][:, :, 0]
        v = dat_mooring.variables['v'][:, :, 0]
        temp = dat_mooring.variables['temp'][:, :, 0]
        Nt = dat_mooring.dimensions['time'].size
        Ns = len(z_interp)
        u_interp = np.empty((Nt, Ns))
        u_interp.fill(np.nan)
        v_interp = np.empty((Nt, Ns))
        v_interp.fill(np.nan)
        temp_interp = np.empty((Nt, Ns))
        temp_interp.fill(np.nan)
        for i in range(Nt):
            print(i, ' ', end='')
            if i%10==0:
                print()
            zeta = dat_mooring.variables['zeta'][:]
            (z_r, z_w) = mooring_zlevs(grd['h'][lon_i:lon_i + 1, lat_i:lat_i + 1], zeta[0:1, :, None], hc, Cs_r, Cs_w)
            u_interp[i, :] = np.interp(z_interp, z_r[0, 0, :], u[i, :])
            v_interp[i, :] = np.interp(z_interp, z_r[0, 0, :], v[i, :])
            temp_interp[i, :] = np.interp(z_interp, z_r[0, 0, :], temp[i, :])

        dat_mooring.close()
        dr=path_mooring.replace('/southern/rbarkan/data/','/atlantic3/michalshaham/')+'2'
        out_file='z_%s.%d.nc' % (mooring_name, int(num))
        print('\n', dr, out_file)
        nco = wrtNcfile_z_lev(dr=dr, outfile=out_file, vardict={'u': u_interp, 'v': v_interp, 'temp': temp_interp, 'depth': z_interp})


################################
# get real mooring
###############################

def get_tao(TAO_file, depth_ind=0):
    # TAO_file = '/atlantic3/michalshaham/EPAC2km/TAO_140W.nc'
    with nc.Dataset(TAO_file, 'r') as dat_tao:
        times = dat_tao.dimensions['time'].size
        depths = dat_tao.dimensions['depth'].size
        data_tao = dat_tao.variables['my_data'][depth_ind*2:(depth_ind+1)*2]/100 #  for units
        print(data_tao.shape)

    return times, depths, data_tao


def get_tao_temp(TAO_file):
    # TAO_file = '/atlantic3/michalshaham/EPAC2km/TAO_140W.nc'
    with nc.Dataset(TAO_file, 'r') as dat_tao:
        times = dat_tao.dimensions['time'].size
        data_tao = dat_tao.variables['my_data'][:] #  for units
        print(data_tao.shape)

    return times, data_tao


####################################
if __name__ == "__main__":
    # path_his = '/southern/rbarkan/data/PACHUG/YY2016M02'
    # path_grd = '/southern/rbarkan/data/PACHUG'
    # path_mooring = '/southern/rbarkan/data/PACHUG/mooring_data'
    # path_mooring_zlev = '/atlantic3/michalshaham/PACHUG/mooring_data'
    # pattern_his = 'pachug_rst.20160229235500.nc'
    # grd_name = "pachug_grd.nc"

    path_his = '/southern/rbarkan/data/EPAC2km/OUTPUT/HIS/'
    path_grd = '/southern/rbarkan/data/EPAC2km'
    path_mooring = '/southern/rbarkan/data/EPAC2km/OUTPUT/EXT'
    path_mooring_zlev = '/atlantic3/michalshaham/EPAC2km/OUTPUT/EXT'
    pattern_his = 'EPAC2km_his.??????.nc'
    grd_name = "Epac2km_grd.nc"

    mooring_name = 'tao2'
    interpolate_mooring(path_mooring, mooring_name, path_grd, grd_name)
