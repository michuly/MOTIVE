import glob
import re
from simulation_parameters import *

def get_file_list(path, pattern, nums=None, num_len=6):
    """
    this function return a list of file paths, that fits the "pattern" given.
    if nums is provided, the list will fit the list of numbers given.
    if nums is None, the function return the list of number of the file name.
    :return: nums, file_path
    """
    num_pattern = r'\b(\d{%d})\b' % num_len
    # print('An example for file pattern:')
    # print(os.path.join(path, pattern))
    file_list = glob.glob(os.path.join(path, pattern)) # Use glob to get a list of file names matching the pattern

    if nums is None:
        nums=[]
        for file_name in file_list:
            match = re.search(num_pattern, file_name) # Use regular expression to extract numbers from file names
            if match:
                nums.append(int(match.group(1)))
    else:
        file_list_tmp=[]
        for num in nums:
            file_name=os.path.join(path, pattern).replace('*', str(num))
            if file_name in file_list:
                file_list_tmp.append(file_name)
            else:
                print('File Not Found: ', file_name)

        # for file_name in file_list:
        #     match = re.search(pattern_tmp, file_name)
        #     if int(match.group(1)) in nums:
        #         file_list_tmp.append(file_name)
        file_list=file_list_tmp

    return sorted(nums), sorted(file_list)


### Tests
# path = '/atlantic3/michalshaham/EMedCrocoC/'
# pattern = 'OUTPUT/his/z_EMed3km_his.0*.nc'
# nums = [141743]
# print(get_file_list(data_path, pattern_his, nums))
# print(get_file_list(data_path, pattern_his))