from simulation_parameters import *
from imports_file import *
from tools.get_depths import get_depths_run

"""
get two plots:
1. u at 0N 140W, depth vs. time
2. u at 104W, temporal average, detph vs. latitude"""
### get history file names
his_files, tot_depths, time_dim = get_concatenate_parameters(depths ,min_num, max_num)

### save an empty psd file ###
dst_path = os.path.join(data_path_his, "u_xi_%d_%d_eta_%d_%d.nc" % (min_xi_u, max_xi_u, min_eta_v, max_eta_v))
print('Saving PSD into data file:', dst_path)
# if not os.path.exists(dst_path):
dat_dst = Dataset(dst_path, 'w')
dat_dst.createDimension('depths', len(tot_depths))
dat_dst.createVariable('depths', np.dtype('float32').char, ('depths',))
dat_dst.variables['depths'][:] = tot_depths
dat_dst.createDimension('kh', len(kh))
dat_dst.createVariable('kh', np.dtype('float32').char, ('kh',))
dat_dst.createDimension('freq', len(freq))
dat_dst.createVariable('freq', np.dtype('float32').char, ('freq',))
dat_dst.createVariable('psd', np.dtype('float32').char, ('depths','freq','kh'))
dat_dst.close()