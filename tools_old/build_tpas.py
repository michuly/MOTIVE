import sys
sys.path.append('/analysis/michalshaham/CrocoTools/Python_Kau/')
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from R_tools_new_michal import zlevs, gridDict, Forder
import scipy.stats as spstats
import os


def tpas_1d(z_, boundary):

    # PLOT TPAS
    # bin_mean, bin_edges, binnumber = spstats.binned_statistic(z_r.flatten(), tpas[0,:,:,:].flatten(), statistic='mean', bins=140, range=[-140, 0])
    # plt.figure()
    # z=-np.linspace(0,150,1000)
    # hbl=-100
    # plt.plot(z, tpas_1d(z, hbl))
    # plt.xlabel('Depth')
    # plt.ylabel('Mean Concentration')
    # plt.title('tpas init func (hbl=100m)')
    # plt.grid(True)

    return 0.5 * (1 + np.tanh(3.6 - 8 * z_ / boundary))


def tpas_func(z, mld_, hbl_ ,mask):

    def tpas_tmp(z_, boundary):
        return np.where(np.logical_or(mask==0, boundary==0), 0, tpas_1d(z_, boundary))

    tpas_mld = tpas_tmp(z, mld_)
    # tpas = mask * tpas_mld
    tpas_hbl = tpas_tmp(z, -hbl_)
    cond=np.repeat(mld_ > -hbl_, 80, 0)
    tpas = np.where(cond, tpas_mld, tpas_hbl)
    tpas = mask * tpas
    return tpas

path = '/atlantic3/michalshaham/EMedCroco3km_D/'
grd_name="INPUT/EMed3km_grd.nc"
pattern_mld = 'Analysis/MLD/EMed3km_his_mld.*.nc'
pattern_his = 'OUTPUT/his/EMed3km_his.*.nc'
num_str='03276'
output_file = '/atlantic3/michalshaham/EMedCroco3km_A/INPUT/EMed3km_his_low_pass_init_tpas.03276.nc'

his_file=os.path.join(path, pattern_his).replace('*', num_str)
mld_file=os.path.join(path, pattern_mld).replace('*', num_str)
dat_his = nc.Dataset(his_file)
hbl=dat_his.variables['hbl'][:]
grd = gridDict(path, grd_name, ij=None)
(z_r, z_w) = zlevs(grd, dat_his, itime=0)
dat_his.close()
dat_mld = nc.Dataset(mld_file)
mld=dat_mld.variables['rho001'][:]
dat_mld.close()
z_r=Forder(z_r)

hbl[np.abs(hbl)<1e-8]=0
tpas_init=tpas_func(z_r, mld[0:1,:,:], hbl[0:1,:,:], Forder(grd['mask_rho'])[np.newaxis, :, :])
tpas_init=tpas_init[np.newaxis,:,:,:]

print('BUILDING TPAS...')
print('Input file:  ', his_file, '...')
print('AND file:    ', mld_file, '...')
print('Output file: ', output_file, '...')
nco=nc.Dataset(output_file, 'a')
if 'tpas' not in nco.variables.keys():
    nco.createVariable('tpas', np.dtype('float32').char, ('time', 's_rho', 'eta_rho', 'xi_rho'))
nco.variables['tpas'][:] = tpas_init
plt.figure()
plt.pcolor(nco.variables['tpas'][0,79,:,:])
plt.title('TPAS from ', output_file.split('michalshaham')[1])
nco.close()
plt.figure()
plt.pcolor(tpas_init[0,79,:,:])
plt.title('TPAS from ', his_file.split('michalshaham')[1])
plt.colorbar()

bin_mean, bin_edges, binnumber = spstats.binned_statistic(z_r.flatten(), tpas_init.flatten(), statistic='mean', bins=140, range=[-140, 0])
plt.figure()
plt.hlines(bin_mean, bin_edges[:-1], bin_edges[1:], colors='g', lw=5, label='binned statistic of data')
plt.plot((bin_edges[:-1]+bin_edges[1:])/2, bin_mean)
# plt.hist(z_r.flatten(), bins=80,range=[-120,0], weights=tpas.flatten(), density=True)
plt.xlabel('Depth')
plt.ylabel('Mean Concentration')
plt.title('Mean Concentration vs Depth')
plt.grid(True)
plt.show()

