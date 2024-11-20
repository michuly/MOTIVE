import os
import socket
import numpy as np

print('Host name: ', socket.gethostname())
"""
lon index for -140.000000: 778
lat index for 0.000000: 249
"""

if socket.gethostname()=='southern' or socket.gethostname()=='atlantic.tau.ac.il':
    data_path = "/southern/rbarkan/data/EPAC2km/OUTPUT/HIS/"
    grd_path = "/southern/rbarkan/data/EPAC2km/"
    grd_name = "Epac2km_grd.nc"
    pattern_his = "z_EPAC2km_his.*.nc"
    data_path_psd = "/southern/michalshaham/Data/MOTIVE/psd"
    data_path_his = "/southern/michalshaham/Data/MOTIVE/his"
    min_num, max_num = 141095, 143111
# minimum and maximum dates of files to be analyzed

    to_slice=False
    len_time = 12
    time_jump = 1
    depths = None
    min_eta_rho, max_eta_rho = 137, 360
    min_eta_v, max_eta_v = 137, 359
    min_xi_rho, max_xi_rho = 223,1334
    min_xi_u, max_xi_u = 223,1333
    if to_slice:
        len_eta_rho = max_eta_rho - min_eta_rho
        len_xi_rho = max_xi_rho - min_xi_rho
        len_eta_v = max_eta_v - min_eta_v
        len_xi_u = max_xi_u - min_xi_u
    else:
        len_eta_rho = 722
        len_xi_rho = 2002
        len_eta_v = 721
        len_xi_u = 2001
    lon_ind = 788 # 140W
    lat_ind = 249 # 0N
    lat_ind_1N = 304 # 1N

elif socket.gethostname()=='Michals-MacBook-Pro.local':
    data_path = "/Users/michal/Data/MOTIVE/"
    grd_path = "/Users/michal/Data/MOTIVE/"
    grd_name_tot = "Epac2km_grd_lon_lat.nc"
    grd_name = "Epac2km_grd_lon_lat_sampled.nc"
    pattern_his = "z_EPAC2km_vel.*.nc"
    pattern_his_1t = "z_EPAC2km_vel_1t.*.nc"
    data_path_psd = "/Users/michal/Data/MOTIVE/psd"
    data_path_his = "/Users/michal/Data/MOTIVE/his"
    min_num, max_num = 141095, 141811  # minimum and maximum dates of files to be analyzed

    to_slice=False
    len_time=12
    time_jump=1
    depths=[-1,-40,-200]
    min_eta_rho,max_eta_rho=137,360
    min_eta_v,max_eta_v=137,359
    min_xi_rho,max_xi_rho=612,945
    min_xi_u,max_xi_u=612,944
    if to_slice:
        len_eta_rho = max_eta_rho - min_eta_rho
        len_xi_rho = max_xi_rho - min_xi_rho
        len_eta_v = max_eta_v - min_eta_v
        len_xi_u = max_xi_u - min_xi_u
    else:
        len_eta_rho = max_eta_rho - min_eta_rho + 1
        len_xi_rho = max_xi_rho - min_xi_rho + 1
        len_eta_v = max_eta_v - min_eta_v + 1
        len_xi_u = max_xi_u - min_xi_u + 1

    lon_ind = 788 - min_xi_u # 140W
    lat_ind = 249 - min_eta_rho # 0N
    lat_ind_1N = 304 - min_eta_rho # 1N


tot_depths = np.array([-0, -1, -2, -4, -7, -10, -15, -20, -25, -30, -40, -50, -60, -70,
-80, -90, -100, -110, -120, -130, -140, -150, -175, -200, -225, -250,
-275, -300, -325, -350, -375, -400, -425, -450, -475, -500, -550, -600,
-650, -700, -750, -800, -875, -950, -1025, -1100, -1175, -1250, -1325,
-1400, -1475, -1550, -1625, -1700, -1775, -1850, -1925, -2000, -2075,
-2150, -2225, -2300, -2375, -2450, -2525, -2600, -2675, -2750, -2825,
-2900, -2975, -3050, -3125, -3200, -3275, -3350, -3425, -3500, -3600,
-3700, -3800, -3900, -4000, -4100, -4200, -4300, -4400, -4500])