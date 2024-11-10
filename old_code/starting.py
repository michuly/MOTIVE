import sys
sys.path.append('/analysis/michalshaham/CrocoTools/Python_Kau/')
from R_tools_new_michal import *
from tools.get_file_list import get_file_list
from tools.get_ocean_time import *
from tools.find_lon_lat import find_lon_lat
import numpy as np
import os
import netCDF4 as nc
import matplotlib.pyplot as plt

depth_ind = 95  # bottom is 0 (?)

dt_0 = 10 * 60  # in sec
path_his = '/southern/rbarkan/data/EPAC2km/OUTPUT/HIS/'
path_grd = '/southern/rbarkan/data/EPAC2km'
path_mooring = '/southern/rbarkan/data/EPAC2km/OUTPUT/EXT/'
path_mooring_zlev = '/atlantic3/michalshaham/EPAC2km/OUTPUT/EXT'
pattern_his = 'EPAC2km_his.??????.nc'


# pattern_tao2 = 'tao2.*.nc'
pattern_mooring = '%s.*.nc'
pattern_mooring_zlev = 'z_%s.*.nc'
mooring = dict(tao1=(-110,0), tao3=(-170,0),tao2=(-140,0),motive1=(-140,0.5),motive2=(-140,1.75),motive3=(-140,3))