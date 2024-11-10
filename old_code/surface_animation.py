import netCDF4 as nc
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy.ma as ma

path = '/southern/rbarkan/data/EPAC2km/'
temp_name = 'OUTPUT/HIS/temp_feb-apr.nc'
vort_name = 'OUTPUT/HIS/vort_feb-APR.nc'
grd_name='Epac2km_grd.nc'
vid_len=531
vid_ind=0

temp_file=os.path.join(path, temp_name)
vort_file=os.path.join(path, vort_name)

with nc.Dataset(os.path.join(path, grd_name)) as dat_grd:
    lon = dat_grd.variables['lon_rho'][:]
    lat = dat_grd.variables['lat_rho'][:]


with nc.Dataset(temp_file, 'r') as dat_temp:
    time_dim = dat_temp.dimensions['time']
    time_size = time_dim.size
    eta_rho_size = time_dim.size
    xi_rho_size = time_dim.size

print(time_size)
#############################################
# building the data matrix
# create the matrix for animation
print(temp_file)
print(vort_file)
dat_temp=nc.Dataset(temp_file, 'r')
dat_vort=nc.Dataset(vort_file, 'r')

# Define the masked value (replace -99999 with the actual masked value)
masked_value = -99999
date=[]

print('Loading data...')
masked_array = ma.masked_values(dat_vort.variables['rvort'][vid_ind * vid_len:(vid_ind + 1) * vid_len, 0, :, :], masked_value)
vort=ma.filled(masked_array, np.nan)
print('Still loading data...')
masked_array = ma.masked_values(dat_temp.variables['temp'][vid_ind * vid_len:(vid_ind + 1) * vid_len, 0, :, :], masked_value)
temp=ma.filled(masked_array, np.nan)
print('Done loading data...')

date=dat_vort.variables['ocean_time'][:]

dat_temp.close()
dat_vort.close()

##############################################3
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
vmin_temp1=5
vmin_temp2=26
vmax_temp=30.3

fig, axes = plt.subplots(2,1, figsize=(10,7))
im1 = axes[0].pcolor(lon, lat, temp[ind, :, :], cmap='seismic', vmin=vmin_temp2, vmax=vmax_temp)
axes[0].set_title(f'date: {target_date_string}', pad=14)
axes[0].set_aspect('equal')
axes[0].set_ylabel('lat', fontsize=12)
divider = make_axes_locatable(axes[0])
cax = divider.append_axes("right", size="4%", pad=0.4)
cbar = plt.colorbar(im1, cax=cax)
cbar.set_label('temp.', fontsize=14)

im2 = axes[1].pcolor(lon, lat, vort[ind, :, :], cmap='seismic', vmin=vmin_vort, vmax=vmax_vort)
axes[1].set_aspect('equal')
axes[1].set_xlabel('lon', fontsize=12)
axes[1].set_ylabel('lat', fontsize=12)
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
    im1 = axes[0].pcolor(lon, lat, temp[ind, :, :], cmap='seismic', vmin=vmin_temp2, vmax=vmax_temp)
    axes[0].set_title(f'date: {target_date_string}', pad=10)
    axes[0].set_aspect('equal')
    axes[0].set_ylabel('lat', fontsize=12)

    im2 = axes[1].pcolor(lon, lat, vort[ind, :, :], cmap='seismic', vmin=vmin_vort, vmax=vmax_vort)
    axes[1].set_aspect('equal')
    axes[1].set_xlabel('lon', fontsize=12)
    axes[1].set_ylabel('lat', fontsize=12)

    return im1, im2


############################################
# Create the animation
print('Saving animation...')
animation = FuncAnimation(fig, update, frames=vid_len, interval=50, blit=False)
# animation = FuncAnimation(fig, update, frames=100, interval=50, blit=False)

# Save the animation as a GIF file
animation.save('surface_full_%d.gif' % vid_ind, writer='pillow')

# Show the animation (optional)
# plt.show()





