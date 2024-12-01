"""
calculate density using Fortran tools
"""
import sys

from old_tools.R_tools_new_goth import Forder
from old_tools.R_tools_new_goth_nautilus import Forder

sys.path.append('/analysis/michalshaham/CrocoTools/Python_Kau/')
from simulation_parameters import *
from imports_file import *
from R_tools_new_michal import gridDict, zlevs, rho_eos, rho1_eos

"""
dimensions:
	time = 12 ;
	s_rho = 100 ;
	eta_rho = 2 ;
	xi_rho = 2002 ;
"""
### get history file names
if len(sys.argv) > 1:
    max_num = int(sys.argv[1])
    min_num = max_num
    print('the depth is: ', min_num)
else:
    min_num, max_num = 141743-24*40, 141743+24*40
his_files, _, time_dim = get_concatenate_parameters(min_num, max_num, pattern_his_file="sampled_EPAC2km_his.*.nc")
grd = gridDict(grd_path, grd_name_2N, ij=None)

### create new rho variable in netcdf ###
with Dataset(his_files[0], 'a') as dat_his:
    if 'rho' not in dat_his.variables.keys():
        dat_his.createVariable('rho', np.dtype('float32').char, ('time', 's_rho', 'eta_rho', 'xi_rho'))

for his_file in his_files:
    dat_his = Dataset(his_file, 'a')
    for i in range(dat_his.dimensions['time'].size):

        print('Uploading variables: temp and salinity from:', i, his_file)
        z_r, z_w = zlevs(grd, dat_his, itime=i)
        print("Check dimensions: ", z_r.shape, z_r[0,0,0], z_w.shape)
        sys.stdout.flush()
        temp = dat_his.variables['temp'][i, :, :, :]
        salt = dat_his.variables['salt'][i, :, :, :]
        print("Check dimensions: ", temp.shape, Forder(temp).shape, z_r.shape, z_w.shape)

        print('Calculating density...')
        sys.stdout.flush()
        rho = rho1_eos(T=Forder(temp), S=Forder(salt), z_r=z_r, z_w=z_w, rho0=dat_his.rho0)
        # rho = rho_eos(T=temp, S=salt, z_r=z_r.transpose(), z_w=z_w.transpose(), rho0=dat_his.rho0)
        print('Mean and std rho:', rho.mean(), rho.std())
        sys.stdout.flush()
        dat_his.variables['rho'][i,:,:,:] = rho

    dat_his.close()

sys.stdout.flush()
print('DONE: saved rho to data file ')
sys.stdout.flush()
