import numpy as np

def radial_profile(data, n1, dx1, n2, dx2, verbose=True):
    """
    this function calculates the psd for horizontal wavenumber (kh), from psd for cartesian wavenumbers (kx and ky).
    compared to Roy's code.
    kh_array construction is different, and fits different n's (and so different df's).
    for the same df's - the same as Roy's code.
    """
    if verbose:
        print("calculating radial profile")
    # calculating and normalizing kh
    df_1, df_2 = 1 / (n1 * dx1), 1 / (n2 * dx2)
    # not sure if this should be mean, or mean of sqr. Roy code dell with same length.
    dk_h = np.sqrt(np.mean([df_1**2, df_2**2]))

    kx = np.abs(np.fft.fftfreq(n1, dx1))
    ky = np.abs(np.fft.fftfreq(n2, dx2))
    kx_mat, ky_mat = np.meshgrid(kx, ky, sparse=True, indexing='ij')
    kh_mat = np.hypot(kx_mat, ky_mat)  # r = sqrt(kx^2+ky^2)

    # using kh matrix to get kh vector with unique values:
    kh_array_normalized = (kh_mat/dk_h).round().astype(int).ravel()
    kh_array = np.unique(kh_array_normalized) * dk_h  # give kh right scale

    # calculating psd
    nt = np.shape(data)[0]
    # data_h_i = np.zeros((nt, len(kh_array)))
    data_h_r = np.zeros((nt, len(kh_array)))
    for t_ind in range(nt):
        # data_h_i[t_ind, :] = np.bincount(kh_array_normalized, weights=np.imag(data[t_ind, :, :]).ravel())
        data_h_r[t_ind, :] = np.bincount(kh_array_normalized, weights=np.real(data[t_ind, :, :]).ravel())
    data_h = data_h_r
    # print('imaginary data for SF:', np.sum(data_h_i))

    return kh_array, data_h