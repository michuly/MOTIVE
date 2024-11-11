import os
import glob
import re


def get_file_list(path, pattern, nums=None, digits=6):
    """
    this function return a list of file paths, that fits the "pattern" given.
    if nums is provided, the list will fit the list of numbers given.
    if nums is None, the function return the list of number of the file name.
    :return: nums, file_path
    """
    num_pattern = r'\b(\d{%d})\b' % digits
    # Use glob to get a list of file names matching the pattern
    file_list = glob.glob(os.path.join(path, pattern))

    if nums is None:
        # Use regular expression to extract numbers from file names
        nums=[]
        for file_name in file_list:
            match = re.search(num_pattern, file_name)
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


# path = '/atlantic3/michalshaham/EMedCrocoC/'
# pattern = 'OUTPUT/his/z_EMed3km_his.0*.nc'
# nums = [2916, 2920, 2924, 2928, 2932, 2936, 2940, 2944, 2948, 2952, 2956]
# print(get_file_list(path, pattern, nums))