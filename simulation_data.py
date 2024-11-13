import os
import socket

print('Host name: ', socket.gethostname())
"""
lon index for -140.000000: 778
lat index for 0.000000: 249
"""

if socket.gethostname()=='southern' or socket.gethostname()=='atlantic.tau.ac.il':
    data_path = "/southern/rbarkan/data/EPAC2km/OUTPUT/HIS/"
    grid_path = "/southern/rbarkan/data/EPAC2km/"
    grd_name = "Epac2km_grd.nc"
    pattern_his = "z_EPAC2km_his.*.nc"
    data_path_psd1d = "/southern/michalshaham/Data/MOTIVE/psd"

    to_slice=True
    len_time = 12
    time_jump=6
    depths=[-1,-40,-175]
    min_eta_rho, max_eta_rho = 223,1334
    min_eta_v, max_eta_v = 223,1333
    min_xi_rho, max_xi_rho = 137, 360
    min_xi_u, max_xi_u = 137, 359
    len_eta_rho = max_eta_rho - min_eta_rho + 1
    len_xi_rho = max_xi_rho - min_xi_rho + 1
    len_eta_v = max_eta_v - min_eta_v + 1
    len_xi_u = max_xi_u - min_xi_u + 1

elif socket.gethostname()=='Michals-MacBook-Pro.local':
    data_path = "/Users/michal/Data/MOTIVE/"
    grid_path = "/Users/michal/Data/MOTIVE/"
    grd_name = "Epac2km_grd_lon_lat.nc"
    grd_name_sampled = "Epac2km_grd_lon_lat_sampled.nc"
    pattern_his = "z_EPAC2km_vel.*.nc"
    pattern_his_1t = "z_EPAC2km_vel_1t.*.nc"
    data_path_psd1d = "/Users/michal/Data/MOTIVE/psd_1d"

    to_slice=False
    len_time=12
    time_jump=2
    depths=None
    min_eta_rho,max_eta_rho=137,360
    min_eta_v,max_eta_v=137,359
    min_xi_rho,max_xi_rho=612,945
    min_xi_u,max_xi_u=612,944
    len_eta_rho=max_eta_rho-min_eta_rho+1
    len_xi_rho=max_xi_rho-min_xi_rho+1
    len_eta_v=max_eta_v-min_eta_v+1
    len_xi_u=max_xi_u-min_xi_u+1
