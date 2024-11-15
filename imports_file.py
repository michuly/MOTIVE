"""
paths and packages
"""
from netCDF4 import Dataset # https://bobbyhadz.com/blog/python-note-this-error-originates-from-subprocess
import numpy as np
import os
from tools.get_file_list import get_file_list
from tools.freq_for_fft import freq_for_fft
from tools.radial_profile import radial_profile
from tools.concatenate_files import get_concatenate_parameters
from tools.get_depths import get_depths_run

import matplotlib.pyplot as plt
# import R_tools_new_michal as tN
import glob
import re
import sys
