from tools.data_analysis_tools import *
from tools.flow_analysis_tools import processing_and_fft, freq_for_fft, radial_profile

"""
this file has generic functions to calculate Energy Spectrum from a field flow of u and v. 
the functions can receive different axes for fft.
"""


def calculate_energy(u_tf, shape, dx_shape, normal_axes=(), mean_axes=(), radial_profile_axes=()):
    """
    psd = u*conj(u)
    the resulted energy is then averaged over dimensions not used in the fft.
    """

    print('calculating PSD... ')
    psd = np.abs(u_tf * np.conj(u_tf))
    del u_tf

    # normalizing and averaging,
    # !!! make sure that shape is the original (before slicing).
    psd = psd.mean(axis=mean_axes)  # averaging over axes not in fft
    delta_f = 1
    for ax in normal_axes:
        delta_f /= (shape[ax] * dx_shape[ax])  # calc normalizing factor df
    if radial_profile_axes:
        freq_h, psd_h = radial_profile(psd, shape, dx_shape, axes=radial_profile_axes)
        psd_h /= freq_h[1] * delta_f  # NORMALIZING FACTOR by 1/df
        return freq_h, psd_h
    else:
        psd /= delta_f  # NORMALIZING FACTOR by 1/df
        return psd


"""
this section calculates the power spectrum density (PSD) for the different dimensions - frequency (time), 
horizontal wavenumber (space), and frequency- horizontal wavenumber (time-space)
"""


def psd_1d(u, dx_shape=(1,), axis=0, filter_flag=0, filter_widths=(0,), filter_axes=(0,), sub_mean=False,
           hours_cut=0, window=False):
    """
    this functions takes a velocity matrix and calculates its energy spectrum Vs. frequency (time).
    """
    if window:
        window_axes = (axis,)
    else:
        window_axes = ()

    if sub_mean:
        minus_mean_axes = (axis,)
    else:
        minus_mean_axes = ()

    if hours_cut != 0:
        hours_cut = (hours_cut, 0, 0)
    else:
        hours_cut = ()

    u_tf, shape = processing_and_fft(u, axes=(axis,), slice_axes=(axis,), minus_mean_axes=minus_mean_axes,
                                     window_axes=window_axes, filter_flag=filter_flag, filter_widths=filter_widths,
                                     filter_axes=filter_axes, hours_cut=hours_cut)
    # del u
    mean_axes = tuple([_ax for _ax in range(np.ndim(u_tf)) if _ax not in (axis,)])
    psd = calculate_energy(u_tf, shape=shape, dx_shape=dx_shape, normal_axes=(axis,), mean_axes=mean_axes)
    freq = freq_for_fft(N1=shape[axis], D1=dx_shape[axis])

    if sub_mean:
        variance = np.sum((u-np.mean(u, 0)) ** 2, 0).mean() / shape[axis]
    else:
        variance = np.sum(u ** 2, 0).mean() / shape[axis]
    psd_sum = np.sum(psd) * freq[1]
    print('Raw data variance ' + str(variance))
    print('Spectrum variance ' + str(psd_sum))
    print('variance ratio %1.1e' % (psd_sum / variance))

    return freq, psd


def psd_2d(u, dx_shape=(1, 1, 1), axes=(1, 2), sub_mean=True, window=False, reflect_axes=(), reflect_flags=(),
           filter_widths=(), filter_axes=()):
    """
    this functions takes a velocity matrix and calculates its energy spectrum Vs. horizontal wavenumber (space).
    using 2d fft for each time step, then averaging over them. (gives the same result as 'psd_horizontal').
    """
    if window:
        window_axes = axes
    else:
        window_axes = ()
    if sub_mean:
        minus_mean_axes = axes
    else:
        minus_mean_axes = ()
    if reflect_axes:
        reflection_normalize = True
    else:
        reflect_axes = ()
        reflect_flags = ()
        reflection_normalize = False
    if axes==(1,2):
        slice_axes=()
    else:
        slice_axes=axes
    filter_flag = 1 if filter_axes else 0
    u_tf, shape = processing_and_fft(u, axes=axes, minus_mean_axes=minus_mean_axes, window_axes=window_axes,
                                     filter_flag=filter_flag, filter_widths=filter_widths, filter_axes=filter_axes,
                                     reflect_axes=reflect_axes, reflect_flags=reflect_flags,
                                     reflection_normalize=reflection_normalize, slice_axes=slice_axes)
    mean_axes = tuple([_ax for _ax in range(np.ndim(u)) if _ax not in axes])

    if axes==(1,2):
        radial_axes=(1,2)
        freq, psd = calculate_energy(u_tf, shape=shape, dx_shape=dx_shape, normal_axes=(), mean_axes=mean_axes,
                                         radial_profile_axes=radial_axes)  # does not fit 2d (mean)
    else:
        radial_axes=()
        psd = calculate_energy(u_tf, shape=shape, dx_shape=dx_shape, normal_axes=(), mean_axes=mean_axes,
                                         radial_profile_axes=radial_axes)  # does not fit 2d (mean)
        freq_t = freq_for_fft(shape[0], dx_shape[0])
        freq_h = freq_for_fft(shape[2], dx_shape[2])
        freq = (freq_t, freq_h)

    # del u
    del u_tf

    variance = np.sum((u - np.nanmean(u)) ** 2) / np.prod([shape[ax] for ax in axes])
    psd_sum = np.sum(psd) * freq[1]
    print('Raw data variance ' + str(variance))
    print('Spectrum variance ' + str(psd_sum))
    print('variance ratio %1.1e' % (psd_sum / variance))
    return freq, psd


def psd_horizontal(u, dx_shape, filter_widths=(), filter_axes=()):
    """
    this functions takes a velocity matrix and calculates its energy spectrum Vs. horizontal wavenumber (space).
    using 3d fft, then summing over the frequencies (time). (gives the same result as 'psd_2d').
    """
    # put the condition from 1d.
    axes = (0, 1, 2)
    time_ax = 0

    # !!! not using windowing over time axis, although fft-ing it.
    # !!! this is bacause I sum over it anyways, (might be wrong)
    filter_flag = 1 if filter_axes else 0
    u_tf, shape = processing_and_fft(u, axes=axes, slice_axes=(time_ax,), minus_mean_axes=(0, 1, 2), window_axes=(1, 2),
                                     filter_flag=filter_flag, filter_widths=filter_widths, filter_axes=filter_axes)
    del u
    print('not sure if need to normal t_axis')
    # psd = calculate_energy(u_tf, shape=shape, dx_shape=dx_shape, mean_axes=(0,), normal_axes=(0,))
    psd = calculate_energy(u_tf, shape=shape, dx_shape=dx_shape, mean_axes=(0,))
    del u_tf

    freq_h, psd_th = radial_profile(psd, shape=shape, dx_shape=dx_shape, axes=(1, 2))

    delta_f = 1 / (shape[time_ax] * dx_shape[time_ax])
    # un-normalizing the time axis, and integrating over it.
    return freq_h, np.sum(psd_th, axis=time_ax) * delta_f


def psd_3d(u, dx_shape, filter_widths=(), filter_axes=(), hours_cut=0, window=False, sub_mean=True, reflecting=False):
    """
    this functions takes a velocity matrix and calculates its energy spectrum Vs. frequency (time)
    and horizontal wavenumber (space). using 3d fft.
    """
    axes = (0, 1, 2)
    time_ax = 0
    if hours_cut != 0:
        hours_cut = (hours_cut, 0, 0)
    else:
        hours_cut = ()
    if window:
        window_axes = axes
    else:
        window_axes = ()
    if sub_mean:
        minus_mean_axes = axes
    else:
        minus_mean_axes = ()
    if reflecting:
        reflect_axes = (1, 2)
        flags = (-1, 0)
        reflection_normalize = True
    else:
        reflect_axes = ()
        flags = ()
        reflection_normalize = False

    filter_flag = 1 if filter_axes else 0
    u_tf, shape = processing_and_fft(u, axes=axes, slice_axes=(time_ax,), minus_mean_axes=minus_mean_axes,
                                     filter_flag=filter_flag, filter_widths=filter_widths, filter_axes=filter_axes,
                                     window_axes=window_axes, hours_cut=hours_cut, reflect_axes=reflect_axes,
                                     reflect_flags=flags, reflection_normalize=reflection_normalize)
    # del u
    freq_h, psd_th = calculate_energy(u_tf, shape=shape, dx_shape=dx_shape, normal_axes=(0,),
                                      radial_profile_axes=(1, 2))
    del u_tf

    freq_t = freq_for_fft(N1=shape[time_ax], D1=dx_shape[time_ax])

    variance = np.sum((u - np.nanmean(u)) ** 2) / np.prod([shape[ax] for ax in axes])
    psd_sum = np.sum(psd_th) * freq_h[1] * freq_t[1]
    print('Raw data variance ' + str(variance))
    print('Spectrum variance ' + str(psd_sum))
    print('variance ratio %1.1e' % (psd_sum / variance))

    return freq_t, freq_h, psd_th


def psd_enstrophy_1d(u, dx_shape, filter_widths=(), filter_axes=(), hours_cut=0, window=True, sub_mean=True,
                     reflect_axes=(), reflect_flags=()):
    """
    this functions takes a velocity matrix and calculates its energy spectrum Vs. frequency (time)
    and horizontal wavenumber (space). using 3d fft.
    """
    axes = (0, 1, 2)
    time_ax = 0
    if hours_cut != 0:
        hours_cut = (hours_cut, 0, 0)
    else:
        hours_cut = ()
    if window:
        window_axes = axes
    else:
        window_axes = ()
    if sub_mean:
        minus_mean_axes = axes
    else:
        minus_mean_axes = ()
    if reflect_axes:
        reflection_normalize = True
    else:
        reflect_axes = ()
        reflect_flags = ()
        reflection_normalize = False

    filter_flag = 1 if filter_axes else 0
    u_tf, shape = processing_and_fft(u, axes=axes, slice_axes=(time_ax,), minus_mean_axes=minus_mean_axes,
                                     filter_flag=filter_flag, filter_widths=filter_widths, filter_axes=filter_axes,
                                     window_axes=window_axes, hours_cut=hours_cut, reflect_axes=reflect_axes,
                                     reflect_flags=reflect_flags, reflection_normalize=reflection_normalize)
    del u
    psd = calculate_energy(u_tf, shape=shape, dx_shape=dx_shape, normal_axes=(0,))
    del u_tf

    freq_t = freq_for_fft(N1=shape[time_ax], D1=dx_shape[time_ax])
    print('FREQ', shape[time_ax], dx_shape[time_ax])

    _, n1, n2 = [shape[ax] for ax in axes]
    _, dx1, dx2 = [dx_shape[ax] for ax in axes]
    k1 = np.abs(np.fft.fftfreq(n1, dx1))
    k2 = np.abs(np.fft.fftfreq(n2, dx2))
    k1_mat, k2_mat = np.meshgrid(k1[1:], k2[1:], sparse=True, indexing='ij')
    kh2_mat = np.zeros((1, n1 - 1, n2 - 1))
    kh2_mat[0, :, :] = k1_mat ** 2 + k2_mat ** 2  # r = sqrt(ky^2+kx^2)

    psd = psd[:, 1:, 1:] / kh2_mat
    psd = np.mean(psd, (1, 2))

    return freq_t, psd
