import sys


def get_depths_run(argv):
    if len(argv) > 1:
        min_depth = int(argv[1])
        max_depth = int(argv[2])
        depths = range(min_depth, max_depth + 1)
        print('START RUNNING: DEPTHS {}-{}'.format(min_depth, max_depth + 1))
        sys.stdout.flush()
        return depths
    else:
        return None