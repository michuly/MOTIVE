from tools.get_file_list import get_file_list

path_his = '/southern/michalshaham/Data/EPAC2km/'
pattern_his = 'OUTPUT/HIS/z_EPAC2km_his.14????.nc'
pattern_his_w = 'OUTPUT/HIS/z_sampled_EPAC2km_his.14????.nc'
path_grd = '/southern/rbarkan/data/EPAC2km/'
grd_name='Epac2km_grd.nc'


#################################
import netCDF4 as nc
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

path = '/southern/rbarkan/data/EPAC2km/'
temp_name = 'OUTPUT/HIS/temp_feb-apr.nc'
vort_name = 'OUTPUT/HIS/vort_feb-APR.nc'
grd_name='Epac2km_grd.nc'
vid_len=24
vid_ind=0
# samples=[310,530,950,1250]
# lat_coor, lon_coor = 3.5, -134
samples=[260,510,1280,1530]
lat_coor, lon_coor = 2, -129

vort_file=os.path.join(path, vort_name)

with nc.Dataset(os.path.join(path, grd_name)) as dat_grd:
    lon_vort = dat_grd.variables['lon_rho'][:]
    lat_vort = dat_grd.variables['lat_rho'][:]
    ind_lat = np.argmin(np.abs(dat_grd.variables['lat_rho'][:, 0] - lat_coor))
    ind_lon = np.argmin(np.abs(dat_grd.variables['lon_rho'][0, :] - (lon_coor)))


with nc.Dataset(vort_file, 'r') as dat_temp:
    time_dim = dat_temp.dimensions['time']
    time_size = time_dim.size
    eta_rho_size = time_dim.size
    xi_rho_size = time_dim.size

#############################################
# building the data matrix
# create the matrix for animation
print(vort_file)
dat_vort=nc.Dataset(vort_file, 'r')

# Define the masked value (replace -99999 with the actual masked value)
masked_value = -99999
date=[]

print('Loading data...')
# masked_array = ma.masked_values(dat_vort.variables['rvort'][vid_ind * vid_len:(vid_ind + 1) * vid_len, 0, :, :], masked_value)
# vort=ma.filled(masked_array, np.nan)
print('Done loading data...')

date=dat_vort.variables['ocean_time'][:]
dat_vort.close()


##########################################

# collecting the dimensions
with nc.Dataset(os.path.join(path_grd, grd_name)) as dat_grd:
    lon_w = dat_grd.variables['lon_rho'][ind_lat, samples[2]:samples[3] + 1]
    lat_w = dat_grd.variables['lat_rho'][samples[0]:samples[1] + 1, ind_lon]

nums, his_files = get_file_list(path_grd, pattern_his_w)
with nc.Dataset(his_files[0], 'r') as dat_his:
    time_dim = dat_his.dimensions['time']
    time_step = time_dim.size
    time_size = time_step * len(nums)
    z_rho_size= dat_his.dimensions['depth'].size
    z_rho = dat_his.variables['depth'][:]
    eta_rho_dim = dat_his.dimensions['eta_rho']
    eta_rho = np.arange(eta_rho_dim.size)
    xi_rho_dim = dat_his.dimensions['xi_rho']
    xi_rho = np.arange(xi_rho_dim.size)

    w_lat_cross = np.zeros((time_size, z_rho_size, xi_rho_dim.size))
    w_lon_cross = np.zeros((time_size, z_rho_size, eta_rho_dim.size))


#####################################
# building the data matrix

# Define the masked value (replace -99999 with the actual masked value)
masked_value = -99999
t_ind=0
# for his_file in his_files:
#     print(his_file)
#     dat_his = nc.Dataset(his_file, 'r')
#
#     for j in range(dat_his.dimensions['time'].size):
#         masked_array = ma.masked_values(dat_his.variables['w'][j, :, ind_4N-samples[0], :], masked_value)
#         w_lat_cross[t_ind, :, :]=ma.filled(masked_array, np.nan)
#         masked_array = ma.masked_values(dat_his.variables['w'][j, :, :, ind_135W-samples[2]], masked_value)
#         w_lon_cross[t_ind, :, :]=ma.filled(masked_array, np.nan)
#
#         t_ind +=1
#         # date.append(int(dat_his.variables['ocean_time'][j]))
#
#     dat_his.close()

#######################################
# Create a figure and axis

from datetime import datetime, timedelta
from mpl_toolkits.axes_grid1 import make_axes_locatable

ind=0
start_date = datetime(2000, 1, 1)
target_date = start_date + timedelta(seconds=date[ind]+608)
target_date_string = target_date.strftime("%Y-%m-%d %H:%M:%S")
print("Target Date:", target_date_string)

vmin_vort = -2e-5
vmax_vort = 2e-5
vmin_u=-1.5
vmax_u=1.5

fig = plt.figure(figsize=(10, 8))

a=5
gs0 = fig.add_gridspec(2, 2*a+1)
ax0 = fig.add_subplot(gs0[0:1, 0:17])
ax1 = fig.add_subplot(gs0[1:2, 0:a])
ax2 = fig.add_subplot(gs0[1:2, a:2*a])
cax1 = fig.add_subplot(gs0[1:2, 2*a:2*a+1])
gs0.update(wspace=1.5, hspace=None)

# ax0 = fig.add_subplot(2, 1, 1)
# ax1 = fig.add_subplot(2, 2, 3)
# ax2 = fig.add_subplot(2, 2, 4)

dat_vort=nc.Dataset(vort_file, 'r')
im0 = ax0.pcolor(lon_vort, lat_vort, dat_vort.variables['rvort'][0, 0, :, :], cmap='seismic', vmin=vmin_vort, vmax=vmax_vort)
dat_vort.close()
ax0.set_title(f'date: {target_date_string}', pad=14)
ax0.set_aspect('equal')
ax0.set_xlabel('lon', fontsize=12)
ax0.set_ylabel('lat', fontsize=12)
from matplotlib.patches import Rectangle
x1,x2,y1,y2=lon_vort[ind_lat, samples[2]],lon_vort[ind_lat, samples[3]],lat_vort[samples[0],ind_lon],lat_vort[samples[1], ind_lon]
ax0.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, alpha=1, facecolor='none', edgecolor='k', lw=2))
ax0.plot([lon_coor] * len(lat_w), lat_w, 'k', ls=':')
ax0.plot(lon_w, [lat_coor] * len(lon_w), 'k', ls='--')
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="4%", pad=0.4)
cbar = plt.colorbar(im0, cax=cax)
cbar.set_label('vort [1/s]', fontsize=14)

dat_his = nc.Dataset(his_files[0], 'r')
im1 = ax1.pcolor(lat_w, z_rho, dat_his.variables['w'][0, :, :, ind_lon - samples[2]], cmap='seismic', vmin=-0.006, vmax=0.006)
ax1.set_ylabel('depth [m]', fontsize=12)
ax1.set_xlabel('lat', fontsize=12)
ax1.plot(lat_w,[-570]*len(lat_w),'k',ls=':',lw=4)
# divider = make_axes_locatable(ax0)
# cax = divider.append_axes("right", size="4%", pad=0.4)
# cbar = plt.colorbar(im1, cax=cax)
# cbar.set_label('w [m/s]', fontsize=12)

im2 = ax2.pcolor(lon_w, z_rho, dat_his.variables['w'][0, :, ind_lat - samples[0], :], cmap='seismic', vmin=-0.006, vmax=0.006)
dat_his.close()
ax2.set_xlabel('lon', fontsize=12)
cbar = plt.colorbar(im2, cax1)
cbar.set_label('w [m/s]', fontsize=14)
ax2.set_yticks([])
ax2.plot(lon_w,[-570]*len(lon_w),'k',ls='--', lw=4)

plt.show()
######################################
# Define the update function for the animation

def update(frame):
    print(frame)
    ax0.clear()
    ax1.clear()
    ax2.clear()

    target_date = start_date + timedelta(seconds=date[frame]+608)
    target_date_string = target_date.strftime("%Y-%m-%d %H:%M:%S")

    dat_vort = nc.Dataset(vort_file, 'r')
    im0 = ax0.pcolor(lon_vort, lat_vort, dat_vort.variables['rvort'][frame, 0, :, :], cmap='seismic', vmin=vmin_vort, vmax=vmax_vort)
    dat_vort.close()

    # im0 = ax0.pcolor(lon, lat, vort[frame, :, :], cmap='seismic', vmin=vmin_vort, vmax=vmax_vort)
    ax0.set_title(f'date: {target_date_string}', pad=14)
    ax0.set_aspect('equal')
    ax0.set_xlabel('lon', fontsize=12)
    ax0.set_ylabel('lat', fontsize=12)
    # divider = make_axes_locatable(ax0)
    # cax = divider.append_axes("right", size="4%", pad=0.4)
    # cbar = plt.colorbar(im0, cax=cax)
    # cbar.set_label('vort [1/s]', fontsize=14)
    ax0.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, alpha=1, facecolor='none', edgecolor='k', lw=2))
    ax0.plot([lon_coor] * len(lat_w), lat_w, 'k', ls=':')
    ax0.plot(lon_w, [lat_coor] * len(lon_w), 'k', ls='--')

    # Plot the matrix for the current time step
    dat_his = nc.Dataset(his_files[int(np.floor(frame/time_step))], 'r')
    im1 = ax1.pcolor(lat_w, z_rho, dat_his.variables['w'][frame % time_step, :, :, ind_lon - samples[2]], cmap='seismic',
                     vmin=-0.006, vmax=0.006)
    ax1.plot(lat_w, [-570] * len(lat_w), 'k', ls=':', lw=4)
    ax1.set_ylabel('depth [m]', fontsize=12)
    ax1.set_xlabel('lat', fontsize=12)

    im2 = ax2.pcolor(lon_w, z_rho, dat_his.variables['w'][frame % time_step, :, ind_lat - samples[0], :], cmap='seismic',
                     vmin=-0.006, vmax=0.006)
    dat_his.close()
    ax2.set_xlabel('lon', fontsize=12)
    ax2.set_yticks([])
    ax2.plot(lon_w, [-570] * len(lon_w), 'k', ls='--', lw=4)

    return im0, im1, im2


###########################################s
# Create the animation
print('Video length: ', time_size)
animation = FuncAnimation(fig, update, frames=time_size, interval=100, blit=False)
# animation = FuncAnimation(fig, update, frames=4, interval=100, blit=False)  # interval = Delay between frames in milliseconds.

# Save the animation as a GIF file
animation.save('cross_w_animation_slow.gif', writer='pillow')

# Show the animation (optional)
plt.show()





