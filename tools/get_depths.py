import sys
import numpy as np


def get_depths_run(argv, tot_depths=None):
    if len(argv) > 1:
        max_depth = int(argv[1])
        min_depth = int(argv[2])
        print('Check max %d is higher than min %d' % (max_depth, min_depth))
        if tot_depths is not None:
            depths = tot_depths[(tot_depths>min_depth) and (tot_depths<=max_depth)]
        else:
            depths = np.arange(min_depth, max_depth+1)
        print('START RUNNING: DEPTHS ', depths)
        sys.stdout.flush()
        return depths
    else:
        return None