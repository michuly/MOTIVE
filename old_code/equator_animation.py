import netCDF4 as nc # https://bobbyhadz.com/blog/python-note-this-error-originates-from-subprocess
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import ticker

import numpy.ma as ma
from tools.get_file_list import get_file_list

path = '/southern/rbarkan/data/EPAC2km/'
pattern_his = 'OUTPUT/HIS/z_EPAC2km_his.14????.nc'
grd_name='Epac2km_grd.nc'
path_vort_zlev = '/atlantic3/michalshaham/EPAC2km/OUTPUT/EXT'


#################################
# lines of code needed for cross plotting vorticity a partial data set called z_EPAC2km_vort_ver

# vort_ver_name = 'z_EPAC2km_vort_ver.14????.nc'
# nums, vort_ver_files = get_file_list(path_vort_zlev, vort_ver_name)
#
# for vort_ver_file in vort_ver_files:
#     dat_vort_ver = nc.Dataset(vort_ver_file, 'r')
#     masked_array = ma.masked_values(dat_vort_ver.variables['vort'][:, 2, :, j].T, masked_value)
#     vort_cross[t_ind, :, :] = ma.filled(masked_array, np.nan)


##########################################
# collecting the dimensions
with nc.Dataset(os.path.join(path, grd_name)) as dat_grd:
    ind_equator = np.argmin(np.abs(dat_grd.variables['lat_rho'][:, 0]))
    lon_u = dat_grd.variables['lon_psi'][ind_equator, :]
    lon_vort = dat_grd.variables['lon_rho'][ind_equator, :]

plot_depth = 40
nums, his_files = get_file_list(path, pattern_his)
with nc.Dataset(his_files[0], 'r') as dat_his:
    time_dim = dat_his.dimensions['time']
    time_dim_size = time_dim.size * len(nums)
    # z_rho_size= dat_his.dimensions['depth']
    z_rho_size = plot_depth
    z_rho = dat_his.variables['depth'][:plot_depth]
    eta_rho_dim = dat_his.dimensions['eta_rho']
    eta_rho = np.arange(eta_rho_dim.size)
    xi_rho_dim = dat_his.dimensions['xi_rho']
    xi_rho = np.arange(xi_rho_dim.size)
    xi_u_dim = dat_his.dimensions['xi_u']
    xi_u = np.arange(xi_u_dim.size)

    u_cross = np.zeros((time_dim_size, z_rho_size, xi_u_dim.size))
    vort_cross = np.zeros((time_dim_size, z_rho_size, xi_rho_dim.size))


#####################################
# building the data matrix

# Define the masked value (replace -99999 with the actual masked value)
masked_value = -99999
t_ind=0
date=[]
for his_file in his_files:
    print(his_file)
    dat_his = nc.Dataset(his_file, 'r')

    for j in range(dat_his.dimensions['time'].size):
        masked_array = ma.masked_values(dat_his.variables['vort'][j, :plot_depth, ind_equator, :], masked_value)
        vort_cross[t_ind, :, :]=ma.filled(masked_array, np.nan)
        masked_array = ma.masked_values(dat_his.variables['u'][j, :plot_depth, ind_equator, :], masked_value)
        u_cross[t_ind, :, :]=ma.filled(masked_array, np.nan)

        t_ind +=1
        date.append(int(dat_his.variables['ocean_time'][j]))

    dat_his.close()

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

fig, axes = plt.subplots(2,1, figsize=(10,7))
im1 = axes[0].pcolor(lon_u, z_rho, u_cross[0, :, :], cmap='seismic', vmin=vmin_u, vmax=vmax_u)
axes[0].set_title(f'date: {target_date_string}', pad=14)
# axes[0].set_aspect('equal')
axes[0].set_ylabel('depth [m]', fontsize=12)
divider = make_axes_locatable(axes[0])
cax = divider.append_axes("right", size="4%", pad=0.4)
cbar = plt.colorbar(im1, cax=cax)
cbar.set_label('zonal vel [m/s]', fontsize=12)
cbar.locator = ticker.FixedLocator([-1.5,-1,-0.5,0,0.5,1,1.5])

im2 = axes[1].pcolor(lon_vort, z_rho, vort_cross[0, :, :], cmap='seismic', vmin=vmin_vort, vmax=vmax_vort)
# axes[1].set_aspect('equal')
axes[1].set_xlabel('lon', fontsize=12)
axes[1].set_ylabel('depth [m]', fontsize=12)
divider = make_axes_locatable(axes[1])
cax = divider.append_axes("right", size="4%", pad=0.4)
cbar = plt.colorbar(im2, cax=cax)
cbar.set_label('vort [1/s]', fontsize=14)

# plt.show()


######################################
# Define the update function for the animation

def update(frame):
    print(frame)
    axes[0].clear()
    axes[1].clear()

    target_date = start_date + timedelta(seconds=date[frame]+608)
    target_date_string = target_date.strftime("%Y-%m-%d %H:%M:%S")

    # Plot the matrix for the current time step
    im1 = axes[0].pcolor(lon_u, z_rho, u_cross[frame, :, :], cmap='seismic', vmin=vmin_u, vmax=vmax_u)
    axes[0].set_title(f'date: {target_date_string}', pad=10)
    # axes[0].set_aspect('equal')
    axes[0].set_ylabel('depth [m]', fontsize=12)

    im2 = axes[1].pcolor(lon_vort, z_rho, vort_cross[frame, :, :], cmap='seismic', vmin=vmin_vort, vmax=vmax_vort)
    # axes[1].set_aspect('equal')
    axes[1].set_xlabel('lon', fontsize=12)
    axes[1].set_ylabel('depth [m]', fontsize=12)

    return im1, im2


###########################################s
# Create the animation

animation = FuncAnimation(fig, update, frames=time_dim_size, interval=50, blit=False)
# animation = FuncAnimation(fig, update, frames=2, interval=100, blit=False)  # interval = Delay between frames in milliseconds.

# Save the animation as a GIF file
animation.save('equator_animation.gif', writer='pillow')

# Show the animation (optional)
plt.show()





