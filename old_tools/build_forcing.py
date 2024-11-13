import sys
sys.path.append('/analysis/michalshaham/CrocoTools/Python_Kau/')
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def build_forcing():
    # Butterworth filter parameters
    cutoff_frequency_days = 7
    sample_rate = 24  # samples per day
    cutoff_frequency = 1 / cutoff_frequency_days  # cutoff frequency in cycles per day

    # Normalize the cutoff frequency to the Nyquist frequency (half the sample rate)
    nyquist_frequency = 0.5 * sample_rate
    normalized_cutoff = cutoff_frequency / nyquist_frequency

    # Design the Butterworth low-pass filter
    order = 4
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    dat_output = nc.Dataset('/atlantic3/michalshaham/EMedCroco3km_F/INPUT/EMed3km_blk_2019_with5predays_low_pass.nc', 'a')
    dat_input = nc.Dataset('/atlantic3/michalshaham/EMedCroco3km_F/INPUT/EMed3km_blk_2019_with5predays.nc', 'r')

    y_ind = list(map(int, np.floor(np.linspace(0, 452, 45))))
    x_ind = list(map(int, np.floor(np.linspace(0, 682, 68))))
    for i in range(len(y_ind) - 1):
        for j in range(len(x_ind) - 1):
            y1, y2, x1, x2 = y_ind[i], y_ind[i + 1], x_ind[j], x_ind[j + 1]
            print(y_ind[i], y_ind[i + 1], x_ind[j], x_ind[j + 1])
            y2_v = y2 if y2<452 else 451
            x2_u = x2 if x2<682 else 681
            v = np.float32(dat_input.variables['vwnd'][:, y1:y2_v, x1:x2])
            dat_output.variables['vwnd'][:, y1:y2_v, x1:x2] = filtfilt(b, a, v, axis=0)
            # v = filtfilt(b, a, v)
            u = np.float32(dat_input.variables['uwnd'][:, y1:y2, x1:x2_u])
            dat_output.variables['uwnd'][:, y1:y2, x1:x2_u] = filtfilt(b, a, u, axis=0)
            # u = filtfilt(b, a, u)
    dat_input.close()
    dat_output.close()
    print('FINISHED writing')


def add_wspd():
    # dat = nc.Dataset('/atlantic3/michalshaham/EMedCroco3km_F/INPUT/EMed3km_blk_2019_with5predays_low_pass.nc', 'a')
    dat = nc.Dataset('/atlantic3/michalshaham/EMedCroco3km_F/INPUT/EMed3km_blk_2019_with5predays.nc', 'r')
    uwnd = dat['uwnd']
    vwnd = dat['vwnd']
    wspd = dat['wspd']
    err=[]
    for i in range(450,550):
        for j in range(90,120):
            if i==0:
                iu1, iu2 = 0, 0
            elif i==681:
                iu1, iu2 = 680, 680
            else:
                iu1, iu2 = i, i-1

            if j==0:
                jv1, jv2 = 0, 0
            elif j==451:
                jv1, jv2 = 450, 450
            else:
                jv1, jv2 = j, j-1

            print(i,j,iu1,iu2,jv1,jv2)
            # wspd[:,j,i]=np.sqrt(0.25*(uwnd[:,j,iu1]+uwnd[:,j,iu2])**2+0.25*(vwnd[:,jv1,i]+vwnd[:,jv2,i])**2)
            a=wspd[1000,j,i]
            b=np.sqrt(0.25*(uwnd[1000,j,iu1]+uwnd[1000,j,iu2])**2+0.25*(vwnd[1000,jv1,i]+vwnd[1000,jv2,i])**2)
            err.append(np.abs((b-a)/np.mean([a,b])))
            print(a,b,err[-1])
    print(np.mean(err))

    dat.close()
    print('FINISHED writing')


def plot_psd():

    file_paths=['/atlantic3/michalshaham/EMedCroco3km_F/INPUT/EMed3km_blk_2019_with5predays.nc',
               '/atlantic3/michalshaham/EMedCroco3km_F/INPUT/EMed3km_blk_2019_with5predays_low_pass.nc']

    for file_path in file_paths:
        dat=nc.Dataset(file_path)
        f=np.fft.fftfreq(8881,1)
        v2_tf_tot = np.zeros(8881)
        y_ind = list(map(int, np.floor(np.linspace(0, 451, 45))))
        x_ind = list(map(int, np.floor(np.linspace(0, 681, 68))))
        y_ind = [90,100,110,120]
        x_ind = [450,460,470,480,490,500]
        for i in range(len(y_ind) - 1):
            for j in range(len(x_ind) - 1):
                print(y_ind[i], y_ind[i + 1], x_ind[j], x_ind[j + 1])
                w_tf = np.float32(dat.variables['wspd'][:8881, y_ind[i]:y_ind[i + 1], x_ind[j]:x_ind[j + 1]])
                w_tf = np.fft.fft(w_tf, axis=0)
                v2_tf_tot += np.sum(np.real(w_tf*np.conjugate(w_tf)), axis=(1,2))
                # v_tf = np.float32(dat.variables['vwnd'][:8881, y_ind[i]:y_ind[i + 1], x_ind[j]:x_ind[j + 1]])
                # v_tf = np.fft.fft(v_tf, axis=0)
                # v2_tf_tot += np.sum(np.real(v_tf * np.conjugate(v_tf)), axis=(1, 2))
                # u_tf = np.float32(dat.variables['uwnd'][:8881, y_ind[i]:y_ind[i + 1], x_ind[j]:x_ind[j + 1]])
                # u_tf = np.fft.fft(u_tf, axis=0)
                # v2_tf_tot += np.sum(np.real(u_tf*np.conjugate(u_tf)), axis=(1,2))

        dat.close()
        plt.plot(f[:int(8881/2)],v2_tf_tot[:int(8881/2)]/452/681)

    plt.xscale('log')
    plt.yscale('log')
    plt.axvline(1/12,0,1e8, linestyle='--', c='k')
    plt.axvline(1/24,0,1e8, linestyle='--', c='k')
    plt.axvline(1/48,0,1e8, linestyle='--', c='k')
    plt.xlabel('freq 1/hr')
    plt.ylabel('PSD')
    plt.title('PSD low passed wind')
    plt.grid(True)
    plt.legend(['All freq forcing', 'Low freq forcing', '48$hr^{-1}$, 24$hr^{-1}$, 12$hr^{-1}$'])


if __name__ == '__main__':
    # plot_psd()
    add_wspd()