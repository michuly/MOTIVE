import sys
import numpy as np
from scripts.psd_1d import tot_depths


def get_depths_run(argv):
    if len(argv) > 1:
        max_depth = int(argv[1])
        min_depth = int(argv[2])
        print('Check max %d is higher than min %d' % (max_depth, min_depth))
        depths = tot_depths[np.where(tot_depths>min_depth) and np.where(tot_depths<=max_depth)]
        print('START RUNNING: DEPTHS ', depths)
        sys.stdout.flush()
        return depths
    else:
        return None