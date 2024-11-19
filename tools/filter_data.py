import numpy as np
from scipy.signal import butter, sosfiltfilt

def butter_sos2_filter(data, filter_width, dt, axis=0, filter_order=6):
    """
    data: signal to filter
    filter_width: the width of the filter in units of dt
    dt: step in time of the signal
    """

    fs = 1 / dt
    if type(filter_width)==tuple:
        _btype='bandpass'
        f_cutoff = 1 / np.array(filter_width)
        print('Filter used: time=%.2g,%.2g, freq=%.2g,%.2g' % (*filter_width*dt, *f_cutoff))
    elif type(filter_width) == int:
        _btype='lowpass'
        f_cutoff = 1 / filter_width
        print('Filter used: time=%.2g, freq=%.2g' % (filter_width*dt, f_cutoff))

    sos = butter(filter_order, f_cutoff, btype=_btype, output='sos', fs=fs)
    data_filt = sosfiltfilt(sos, data, axis=axis)
    return data_filt