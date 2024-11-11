import sys
sys.path.append('/analysis/michalshaham/CrocoTools/Python_Kau/')
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from tools.get_file_list import get_file_list
from R_tools_new_michal import vort, zlevs, colorbar_tight, gridDict, ncload, strain, Forder, linear_interp
import scipy.stats as spstats
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import ticker
import numpy.ma as ma


path = '/atlantic3/michalshaham/EMedCroco3km_D/'
grd_name="INPUT/EMed3km_grd.nc"
pattern_mld = 'Analysis/MLD/EMed3km_his_mld.*.nc'
pattern_his_z = 'OUTPUT/his/z_EMed3km_his.*.nc'
pattern_his = 'OUTPUT/his/EMed3km_his.*.nc'
start_str='03276'
end_str='03300'
output_file = '/atlantic3/michalshaham/EMedCroco3km_A/INPUT/EMed3km_his_low_pass_init_tpas.03276.nc'


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
    ind_equator = np.argmin(np.abs(dat_grd.variables['lat_rho'][:, 30]-34))
    lon = dat_grd.variables['lon_rho'][ind_equator, :]

plot_depth = 20
nums, z_his_files = get_file_list(path, pattern_his_z, digits=5)
z_his_files=[z_his_files[i] for i in range(len(z_his_files)) if nums[i] >= 3276]
nums, his_files = get_file_list(path, pattern_his, digits=5)
his_files=[his_files[i] for i in range(len(his_files)) if nums[i] >= 3276]

with nc.Dataset(z_his_files[0], 'r') as dat_his:
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

    tpas_cross = np.zeros((time_dim_size, z_rho_size, xi_rho_dim.size))


#####################################
# building the data matrix

# Define the masked value (replace -99999 with the actual masked value)
masked_value = -99999
t_ind=0
date=[]
for his_file in z_his_files:
    print(his_file)
    dat_his = nc.Dataset(his_file, 'r')

    for j in range(dat_his.dimensions['time'].size):
        masked_array = ma.masked_values(dat_his.variables['tpas'][j, :plot_depth, ind_equator, :], masked_value)
        tpas_cross[t_ind, :, :]=ma.filled(masked_array, np.nan)
        t_ind +=1
        date.append(int(dat_his.variables['time'][j]))

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

fig, axes = plt.subplots(2,1, figsize=(10,7))

dat_his = nc.Dataset(his_files[0])
hbl = dat_his.variables['hbl'][0,ind_equator,:]
grd = gridDict(path, grd_name, ij=None)
tpas = dat_his.variables['tpas'][0,:,:,:]
(z_r, z_w) = zlevs(grd, dat_his, itime=0)
dat_his.close()
z_r = Forder(z_r)
axes[1].hist(z_r.flatten(), bins=80, range=[-120, 0], weights=tpas.flatten(), density=True)
axes[1].set_xlabel('Depth')
axes[1].set_ylabel('Mean Concentration')
axes[1].set_title('Mean Concentration vs Depth')
axes[1].grid(True)

im2 = axes[0].pcolor(lon, z_rho, tpas_cross[0, :, :], cmap='YlGnBu', vmin=0, vmax=1)
# axes[1].set_aspect('equal')
axes[0].set_xlabel('lon', fontsize=12)
axes[0].set_ylabel('depth [m]', fontsize=12)
axes[0].plot(lon, -hbl, 'r', linewidth=1)
divider = make_axes_locatable(axes[0])
cax = divider.append_axes("right", size="4%", pad=0.4)
cbar = plt.colorbar(im2, cax=cax)
cbar.set_label('density', fontsize=14)

# plt.show()


######################################
# Define the update function for the animation

def update(frame):
    print(frame)
    axes[0].clear()
    axes[1].clear()

    # target_date = start_date + timedelta(seconds=date[frame]+608)
    # target_date_string = target_date.strftime("%Y-%m-%d %H:%M:%S")

    dat_his = nc.Dataset(his_files[int(frame / 4)])
    hbl = dat_his.variables['hbl'][np.mod(frame,4),ind_equator,:]
    tpas = dat_his.variables['tpas'][np.mod(frame,4),:,:,:]
    grd = gridDict(path, grd_name, ij=None)
    (z_r, z_w) = zlevs(grd, dat_his, itime=0)
    dat_his.close()
    z_r = Forder(z_r)
    axes[1].hist(z_r.flatten(), bins=80, range=[-120, 0], weights=tpas.flatten(), density=True)
    axes[1].set_xlabel('Depth')
    axes[1].set_ylabel('Mean Concentration')
    axes[1].set_title('Mean Concentration vs Depth')
    axes[1].grid(True)
    axes[1].grid(True)

    # Plot the matrix for the current time step
    z = 'seismic'
    im1 = axes[0].pcolor(lon, z_rho, tpas_cross[frame, :, :], cmap='YlGnBu', vmin=0, vmax=1)
    axes[0].set_title(f'hour: {frame}', pad=10)
    axes[0].plot(lon, -hbl, 'r', linewidth=1)
    # axes[0].set_aspect('equal')
    axes[0].set_ylabel('depth [m]', fontsize=12)


    return im1, im2


###########################################s
# Create the animation

# animation = FuncAnimation(fig, update, frames=time_dim_size, interval=50, blit=False)
# animation = FuncAnimation(fig, update, frames=280, interval=100, blit=False)  # interval = Delay between frames in milliseconds.

# Save the animation as a GIF file
# animation.save('equator_animation_B.gif', writer='pillow')

# Show the animation (optional)
plt.show()





