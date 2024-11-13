from netCDF4 import Dataset
import numpy as np
from simulation_data import *

dst_path = ''
dim_strs, dim_sizes, dim_values = [], [], [] # dimension names and values?
var_strs, var_dims, var_values = [], [], []

dat_dst = Dataset(dst_path, 'w')
for dim_str, dim_size, dim_value in zip(dim_strs, dim_sizes, dim_values):
    dat_dst.createDimension(dim_str, dim_size)
    if dim_value:
        dat_dst.createVariable(dim_str, np.dtype('float32').char, (dim_str,))
        dat_dst.variables[dim_str][:] = dim_value

for var_str, var_dim, var_value in zip(var_strs, var_dims, var_values):
    dat_dst.createVariable(var_str, np.dtype('float32').char, var_dim)
    dat_dst.variables[var_str][:] = var_value

dat_dst.close()