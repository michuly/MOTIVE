import numpy as np

def freq_for_fft(N1, D1, N2=None, D2=None):
    """
    using np.fft to calculate the freq array of a given axis.
    it cuts half of the array, due to the symmetrical nature of the fft.
    """
    if N2 is None and D2 is None:
        freq = np.fft.fftfreq(N1, d=D1)[0:int(np.floor(N1 / 2) + 1)]  # cut the freq array in the middle
        if N1 % 2 == 0:
            freq[-1] = -freq[-1]  # if N is even, the last freq is negative, and need to be positive.
        return freq
    else:
        df_1, df_2 = 1 / (N1 * D1), 1 / (N2 * D2)
        # not sure if this should be mean, or mean of sqr. Roy code dell with same length.
        dk_h = np.sqrt(np.mean([df_1**2, df_2**2]))

        kx = np.abs(np.fft.fftfreq(N1, D1))
        ky = np.abs(np.fft.fftfreq(N2, D2))
        kx_mat, ky_mat = np.meshgrid(kx, ky, sparse=True, indexing='ij')
        kh_mat = np.hypot(kx_mat, ky_mat)  # r = sqrt(kx^2+ky^2)
        kh_array_normalized = (kh_mat/dk_h).round().astype(int).ravel()
        kh_array = np.unique(kh_array_normalized) * dk_h  # give kh right scale

        # NORMALIZING FACTOR, find factor to get density.
        # a_cor = 2 * np.pi * kh_array * dk_h
        # a_cor[0]=1
        # data_h /= a_cor # normalizing df_h
        return kh_array