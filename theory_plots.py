import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap, TSNE
from scipy.interpolate import LSQUnivariateSpline
from scipy.signal import (
        periodogram, welch, spectrogram
)

import pycwt as wavelet
from pycwt.helpers import find

from helper_functions import (
        plot_scaleogram, get_simulated_data,
        estimate_pdf, get_watershed_labels,
        get_contours, plot_watershed_heat,
        assign_labels,
)
                            
from simulated_data import load_simulated

import cv2
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi


def underlying(t):
    """ Underlying time series """
    return np.sin(2*np.pi*t) + 1/4*np.cos(6*np.pi*t) + t / 3


def non_stationary_func(t):
    """
    Example function which has frequencies on multiple scales,
    which changes over time
    """
    w1, w2, w3 = 2.4, 4.7, 17
    w4, w5, w6 = 0.7, 1.4, 6.4
    return np.where(t < 30, 3*np.sin(w1*2*np.pi*t) + 2*np.sin(w2*2*np.pi*t) + np.sin(w3*2*np.pi*t),
               np.sin(w4*2*np.pi*t) + 2*np.sin(w5*2*np.pi*t) + 3*np.sin(w6*2*np.pi*t))


def get_theory_data():
    """
    Create simple simulated time series data to exemplify
    the various methods.
    """
    n, t0, tn = 500, 0, 4
    t = np.linspace(t0, tn, n).reshape(-1, 1)
    y = underlying(t[:,0]) + np.random.normal(0, 0.3, n)

    return y, t


def get_non_stationary_data():
    """
    Create simulated non-stationary data on multiple 
    frequency scales.
    """

    t = np.arange(60*120 + 1) / 120
    n = len(t)
    y = non_stationary_func(t) + np.random.normal(0, 0.3, n)

    return y, t
    

def plot_non_stationary_time_series():
    """ Example plot of a non-stationary time series """ 

    y, t = get_theory_data()

    plt.figure(figsize = (12, 5))
    plt.plot(t, y, c = "k", label = "Observed series")
    plt.xlabel(r'$t$')
    plt.savefig("./figures/time_series_example.pdf", bbox_inches = "tight")
    # plt.show()


def plot_linear_trend():
    """ 
    Plot the linear trend in non-stationary time series
    found by linear regression 
    """

    y, t = get_theory_data()

    reg = LinearRegression().fit(t, y) 

    plt.figure(figsize = (12, 5))
    plt.plot(t, y, c = "k", label = "Observed series")
    # plt.plot(t, underlying(t), c = "r", label = "Underlying function")
    plt.plot(t, reg.predict(t), c = "b", label = "Linear trend")
    plt.xlabel(r'$t$')
    plt.savefig("./figures/time_series_example_with_trend.pdf", bbox_inches = "tight")
    # plt.show()


def plot_cubic_splines():
    """ 
    Apply cubic spline regression using 
    least squares on the data
    """

    y, t = get_theory_data()
    
    deg = 3
    t = np.reshape(t, t.size)  # Need to flatten the array
    n, t0, tn = len(t), t[0], t[-1]
    m1, m2 = 10, 50
    knots10 = np.linspace(t0, tn, n // m1)[1:-1,]
    knots50 = np.linspace(t0, tn, n // m2)[1:-1,]

    spl10 = LSQUnivariateSpline(t, y, knots10, k = deg)
    spl50 = LSQUnivariateSpline(t, y, knots50, k = deg)
    plt.figure(figsize = (12, 5))
    plt.plot(t, y, c = "k", alpha = 0.8,
             label = "Observed series")
    plt.plot(t, spl10(t), c = "r", lw = 3,
             label = "Cubic spline: " + r'$n =$' + str(n) + r'$, m =$' + str(n // m1))
    plt.plot(t, spl50(t), c = "g", lw = 3,
             label = "Cubic spline: " + r'$n =$' + str(n) + r'$, m =$' + str(n // m2))
    plt.xlabel(r'$t$')
    plt.legend()
    plt.savefig("./figures/cubic_splines.pdf", bbox_inches="tight")
    # plt.show()


def plot_periodogram():
    """
    Plot the periodogram, i.e., the estimated power spectral density 
    """

    y, t = get_theory_data()

    fs = 1/(t[1] - t[0])
    f, Pxx_den = periodogram(y ,fs, window = "boxcar",
                             detrend = "linear", scaling = "density")

    plt.figure(figsize = (12, 5))
    plt.semilogy(f, Pxx_den)
    plt.vlines(x = [1, 3], ymin = 1e-5, ymax = 1e1, 
               colors = "red", linestyles = "dashed", alpha = 0.5)
    plt.xlim([0, 20])
    plt.ylim([1e-5, 1e1])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power spectrum")
    # plt.savefig("./figures/periodogram_basic.pdf", bbox_inches="tight")
    # plt.show()


def plot_smoothed_periodogram():
    """
    Plot the smoothed periodogram, i.e., the estimated 
    power spectral density using a spectral window.
    """

    y, t = get_theory_data()

    fs = 1/(t[1] - t[0])
    f, Pxx_den = welch(y ,fs, window = "bartlett", noverlap = 0,
                             nperseg = 256, detrend = "linear",  
                             scaling = "density")

    plt.figure(figsize = (12, 5))
    plt.semilogy(f, Pxx_den)
    plt.vlines(x = [1, 3], ymin = 1e-5, ymax = 1e1, 
               colors = "red", linestyles = "dashed", alpha = 0.5)
    plt.xlim([0, 20])
    plt.ylim([1e-5, 1e1])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power spectrum")
    plt.savefig("./figures/periodogram_smoothed.pdf", bbox_inches="tight")
    # plt.show()


def plot_windows():
    """
    Plot the hann window, both the lag window
    and the spectral window.
    """
    M = 5 
    k = np.linspace(-6, 6, 500)
    w = np.linspace(-np.pi - 0.5, np.pi + 0.5, 500)
    def w_r(w):
        """ Rectangular spectral window """
        return 1/(2*np.pi)*np.sin(w*(M+0.5))/np.sin(w/2)
    def w_t(w):
        """ Hann spectral window """
        return (0.25*w_r(w - np.pi/M) + (1 - 0.5)*w_r(w) +
                0.25*w_r(w + np.pi/M))
    def w_h(k):
        """ Hann lag window """
        return np.where(np.abs(k) <= M, 0.5*(1 + np.cos(np.pi*k/M)), 0)
        

    plt.figure(figsize = (12, 5))
    ax1 = plt.subplot(1, 2, 1)
    plt.plot(k, w_h(k), c = "k", label = r'$W_n^H(k)$')
    plt.xlabel(r'$k$')
    plt.legend()

    ax2 = plt.subplot(1, 2, 2)
    plt.plot(w, w_t(w), c = "k", label = r'$\mathcal{W}_n^H(\omega)$')
    plt.xlabel(r'$\omega$')
    plt.legend()
    plt.savefig("./figures/windows.pdf", bbox_inches = "tight")
    # plt.show()


def spectrogram_example():
    """
    Illustrate the difference in the periodogram, STFT and CWT
    """

    y, t = get_non_stationary_data()

    # Compute periodogram
    fs = 1/(t[1] - t[0])
    f, Pxx_den = periodogram(y ,fs, window = "hann",
                             scaling = "density")
    plt.figure(figsize = (12, 5))
    plt.semilogy(f, Pxx_den)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power spectrum")
    # plt.show()

    # Compute short-time Fourier transform and plot spectrogram
    fo, time, Sxx = spectrogram(y, fs, window = "hann")
    plt.pcolormesh(time, fo, Sxx)
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time")
    # plt.show()
   

def wavelet_example():
    """
    Example to understand and play with CWT parameters
    """

    y, t = get_non_stationary_data()

    # Normalize
    std = np.std(y)
    y = (y - np.mean(y)) / std
    # Save length
    n = len(y)
    # Sampling time
    dt = t[1] - t[0]
    # Number of frequency scales
    num_scales = 18
    # Frequency upper bound
    w_max = 20
    # Frequency lower bound
    w_min = 0.5

    # Starting scale (smallest scale)
    s0 = 1/w_max
    # Largest scale 
    sn = 1/w_min
    # Number of scales minus one
    J =  num_scales - 1
    # Number of sub-octaves per octave
    dj = 1/J*np.log2(sn/s0)
    # Mother wavelet
    mother = wavelet.Morlet(6)

    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(
            y, dt, dj, s0, J, mother)
    # wave has shape: (num_scales, n)
    iwave = wavelet.icwt(wave, scales, dt, dj, mother)


    power = np.abs(wave)**2
    fft_power = np.abs(fft)**2
    period = 1/freqs

    # Divide power by the scales to be able to compare them Liu 2007
    # power /= scales[:, None]
    
    
    plt.figure(figsize = (12, 5))
    plot_scaleogram(power, t, freqs, scales)
    plt.savefig("./figures/scaleogram_example.pdf")
    # plt.show()
    
    

def watershed_example():
    """
    Applies kernel denisity estimation
    followed by watershed segmentation
    on simulated data.
    Plots exemplifies the groupings.
    """

    df = load_simulated()
    bc = df["bc"]
    data = bc.embedded_train
    border = 30
    bw = 0.2
    pixels = 500j 
    kde, grid = estimate_pdf(data, bw, border, pixels)
    labels = get_watershed_labels(kde)
    contours = get_contours(kde, labels)
    data_labels = assign_labels(data, labels, grid) 

    outside = np.ones(kde.shape)
    outside[kde == 0] = 0

    xmax, ymax = np.max(data, axis = 0) + border
    xmin, ymin = np.min(data, axis = 0) - border

    lab = df["labels"][::int(bc.capture_framerate / bc.ds_rate)]

    plt.figure(figsize = (12, 12))
    ax1 = plt.subplot(221)
    plt.imshow(np.zeros(contours.shape), 
               alpha = np.zeros(contours.shape),
               extent = [xmin, xmax, ymin, ymax]) 
    plt.scatter(data[:,0], data[:,1], c = lab + 1,
                cmap = "Paired", marker = ".")
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.xticks([])
    plt.yticks([])
    ax2 = plt.subplot(222, sharex = ax1, sharey = ax1)
    im1 = ax2.imshow(kde, cmap = "coolwarm", alpha = outside,
               extent = [xmin, xmax, ymin, ymax]) 
    plt.grid(None)
    plt.xticks([])
    plt.yticks([])
    ax3 = plt.subplot(223)
    plot_watershed_heat(data, kde, contours, border)
    plt.xticks([])
    plt.yticks([])
    ax4 = plt.subplot(224)
    plt.imshow(np.zeros(contours.shape), alpha = contours,
               extent = [xmin, xmax, ymin, ymax]) 
    plt.scatter(data[:,0], data[:,1], c = lab + 1,
                cmap = "Paired", marker = ".")
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(wspace = 0.1, hspace = 0)
    plt.savefig("./figures/clustering_example.pdf",
                bbox_inches = "tight")
    # plt.show()

       


def main():
    seaborn.set_theme()
    np.random.seed(28)

    # plot_non_stationary_time_series()
    # plot_linear_trend()
    # plot_cubic_splines()
    # plot_periodogram()
    # plot_smoothed_periodogram()
    # plot_windows()

    # spectrogram_example()
    # wavelet_example() 
    # watershed_example()


if __name__ == "__main__":
    main()
