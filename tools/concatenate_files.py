from simulation_parameters import *
from tools.get_file_list import get_file_list
from netCDF4 import Dataset
import numpy as np

# from R_tools_new_michal import zlevs, gridDict, Forder
def get_concatenate_parameters(depths=None ,min_num=0, max_num=0):
    ### get history file names ###
    nums, his_files = get_file_list(data_path, pattern_his, num_len=6)
    print('Maximum and Minimum time pf file found: ', np.min(nums), np.max(nums))
    if min_num != 0:
        his_files = [his_files[i] for i in range(len(his_files)) if (nums[i] >= min_num)]
    if max_num != 0:
        his_files = [his_files[i] for i in range(len(his_files)) if (nums[i] <= max_num)]
    print('Example for history file: ', his_files[-1])
    ### set time parameters ###
    with Dataset(his_files[0], 'r') as dat_his:
        # print("What is up, what is down:")
        # print(dat_his.variables['u'][:, 0, :, :].mean(), dat_his.variables['u'][:, -1, :, :].mean())
        print("What are the shapes:")
        print(len_xi_rho, len_xi_u, len_eta_rho, len_eta_v)
        print(dat_his.variables['u'].shape, dat_his.variables['v'].shape)
        # print(dat_his.variables['depth'][0])
        time_dim = dat_his.dimensions['time'].size
        depths = dat_his.variables['depth'][:]
            # depths = depths[depths > -800]

    return his_files, depths, time_dim

