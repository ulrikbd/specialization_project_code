import numpy as np
import scipy
import math
import vg
import pickle

from scipy.interpolate import LSQUnivariateSpline
from scipy.stats import gaussian_kde
from scipy import ndimage as ndi

import matplotlib.pyplot as plt
import numpy.random as rd

from skimage.segmentation import watershed
from skimage.feature import peak_local_max

import cv2 as cv
import imutils

import pycwt as wavelet
from pycwt.helpers import find

from sklearn.manifold import TSNE


def estimate_pdf(array, bw, border, pixels):
    """
    Estimate the probability density
    of a 2D array.
    Applies a kernel density estimator
    with given bandwidth

    Parameters:
        array (np.ndarray(float)): Array to be 
            used for the estimation
        bw (float): Bandwidth used in the kde algorithm
        border (float): Padding around data to make
            watershed segmentation easier
        pixels (int): Pixelation along each axis
            to compute the pdf on

    Returns:
        kde (np.ndarray): Pdf estimated by kernel
            density estimation on a grid defined
            by xmin, ymin, dx, dy
        grid (list(np.ndarray)): List containing 
            information (X, Y) about the grid where 
            kde were applied
    """


    kernel = gaussian_kde(array.T)

    xmax, ymax = np.max(array, axis = 0) + border
    xmin, ymin = np.min(array, axis = 0) - border
    X, Y = np.mgrid[xmin:xmax:pixels, ymin:ymax:pixels]
    positions = np.vstack([X.ravel(), Y.ravel()])
    kde = np.reshape(kernel(positions).T, X.shape)
    kde = np.rot90(kde)
    # Truncate low values to zero, such that we get an
    # outer border
    kde[np.abs(kde) < 1e-5] = 0
    

    return kde, [X, Y]


def get_watershed_labels(image):
    """
    Performs watershed segmentation on
    an image.

    Parameters:
        image (ndarray): nxm image array where
            high values are far from the border.

    Returns:
        labels (np.ndarray): Label array
            given by the watershed segmentation
            algorithm
    """

    max_coords = peak_local_max(image)
    local_maxima = np.zeros_like(image, dtype = bool)
    local_maxima[tuple(max_coords.T)] = True
    markers = ndi.label(local_maxima)[0]
    labels = watershed(-image, markers, mask = image)

    return labels


def get_contours(image, labels):
    """
    Obtain the contours for an image,
    here a watershed segmentation.

    Parameters:
        image (ndarray): nxm image array where
            high values are far from the border.
        labels (ndarray): nxm array with the 
            labels found by watershed segmentation

    Returns:
        countours (ndarray): nxm array which is
            zero everywhere except at countours,
            where it is one.
    """

    unique_labels = np.unique(labels)
    # Create space for the contours
    contours = np.zeros(image.shape)

    
    # Iterate over the clusters
    for i in range(1, len(unique_labels)):
        # Create image only showing the single cluster
        mask = np.zeros(image.shape, dtype = "uint8")
        mask[labels == unique_labels[i]] = 255

        # Find the contours
        cnts = cv.findContours(mask.copy(),
                                cv.RETR_LIST,
                                cv.CHAIN_APPROX_NONE)
        cnts = imutils.grab_contours(cnts)[0]
        
        for pair in cnts:
            contours[pair[0][0], pair[0][1]] = 1

    return contours.T



def assign_labels(data, labels, grid):
    """
    Classify data points to labels given by
    a watershed segmentation.

    Parameters:
        data (np.ndarray): Collection of 
            data points (t-SNE embeddings)
            to be classified.
        labels (np.ndarray): Label array
            given by the watershed segmentation
            algorithm
        grid (list(np.ndarry)): X and Y coordinates
            to which the kernel density estimation was
            applied.

    Returns:
        data_labels (np.ndarray): Array
            containing the assigned labels
    """

    # Rotate the watershed image 270 degrees 
    # to match the embedding coordinates
    labels = np.rot90(labels, 3)

    X = grid[0]
    Y = grid[1]

    # Create storage for the labels
    data_labels = np.zeros(len(data))

    # Iterate over every data point
    for i in range(len(data)):
        # Find the closest grid point
        xi = (np.abs(X[:,0] - data[i,0])).argmin()
        yi = (np.abs(Y[0,:] - data[i,1])).argmin()
        
        # Assign label
        data_labels[i] = int(labels[xi, yi])

    return data_labels

        
def plot_watershed_heat(data, image, contours, border):
    """
    Plots the segmented heatmat of the pdf obtained
    via kernel density estimation of the t-SNE
    embedding.

    Parameters:
        data (ndarray):
    """


    xmax, ymax = np.max(data, axis = 0) + border
    xmin, ymin = np.min(data, axis = 0) - border

    outside = np.ones(image.shape)
    outside[image == 0] = 0

    plt.imshow(image, cmap = "coolwarm", alpha = outside,
               extent = [xmin, xmax, ymin, ymax]) 
    plt.imshow(np.zeros(contours.shape), alpha = contours,
               extent = [xmin, xmax, ymin, ymax]) 
    plt.grid(None)



def spline_regression(y, dim, freq, knot_freq):
    """
    Performs spline regression on a time series y(t).
    The internal knots are centered to get close to 
    equal distance to the endpoints.

    Parameters:
        y (np.ndarray): The time series values
        dim (int): Order of the spline
        freq (float): Frequency of the time points.
        knot_freq (float): Chosen frequency of the
            internal knots.

    Returns:
        (np.ndarray): The regression curve at time 
            points t
    """

    t = np.arange(len(y))
    
    # Calculate the interior knots
    space = int(freq / knot_freq)
    rem = len(t) % space
    knots = np.arange(freq + rem // 2, len(t), space)
    
    spl = LSQUnivariateSpline(t, y, knots, k = dim)
    
    return spl(t)




def pickle_relevant_features(original_filepath, new_filepath):
    """
    Pickle the relevant features contained in the data
    found in the given filepath

    Parameters:
        filepath (string): path to the original pickled data
    """

    # Get the pickled data for one rat
    with open(original_filepath, "rb") as file:
        data = pickle.load(file)
    
    raw_features = extract_relevant_features(data)

    with open(new_filepath, "wb") as file:
        pickle.dump(raw_features, file)


def extract_relevant_features(data):
    """
    Collect the relevant time series which will be used 
    as raw features in the analysis.
    Features extracted:
     - Egocentric head actions relative to body in 3D:
        roll (X), pitch (Y), azimuth (Z)
     - Speed in the XY plane
     - Back angles: pitch (Y), azimuth (Z)
     - Sorted point data??


    Parameters:
        data (dict): All the provided data on one rat

    Returns:
        relevant_features (numpy.ndarray): 
            The relevant data
    """
    
    n_features = 7
    speeds = np.array(data["speeds"][:,2])
    ego3_rotm = np.array(data["ego3_rotm"])

    n_time_points = len(speeds)

    ego3q = np.zeros((n_time_points, 3))
    for i in range(n_time_points):
        ego3q[i,:] = rot2expmap(ego3_rotm[i,:,:])
    
    relevant_features = np.zeros((n_time_points, n_features))
    relevant_features[:,:3] = ego3q
    relevant_features[:,3] = speeds
    relevant_features[:,4:6] = data["back_ang"]
    relevant_features[:,6] = data["sorted_point_data"][:,4,2]
    
    return relevant_features


def rot2expmap(rot_mat):
    """
    Converts rotation matrix to quaternions
    Stolen from github
    """

    expmap = np.zeros(3)
    if np.sum(np.isfinite(rot_mat)) < 9:
        expmap[:] = np.nan
    else:
        d = rot_mat - np.transpose(rot_mat)
        if scipy.linalg.norm(d) > 0.01:
            r0 = np.zeros(3)
            r0[0] = -d[1, 2]
            r0[1] = d[0, 2]
            r0[2] = -d[0, 1]
            sintheta = scipy.linalg.norm(r0) / 2.
            costheta = (np.trace(rot_mat) - 1.) / 2.
            theta = math.atan2(sintheta, costheta)
            r0 = r0 / scipy.linalg.norm(r0)
        else:
            eigval, eigvec = scipy.linalg.eig(rot_mat)
            eigval = np.real(eigval)
            r_idx = np.argmin(np.abs(eigval - 1))
            r0 = np.real(eigvec[:, r_idx])
            theta = vg.angle(r0, np.dot(rot_mat, r0))

        theta = np.fmod(theta + 2*math.pi, 2*math.pi) # Remainder after dividsion (modulo operation)
        if theta > math.pi:
            theta = 2*math.pi - theta
            r0 = -r0
        expmap = r0*theta

    return expmap



def plot_scaleogram(power, t, freqs, scales, ax):
    """
    Plots the spectrogram of the power density
    achived using a continuous wavelet transform

    Parameters:
        power (np.ndarray): Estimated power for all 
            time points across multiple scales
    """
     
    # Find appropriate contour levels
    min_level = np.log2(power).min() / 2
    max_level = np.max([np.log2(power).max(), 1])
    level_step = (max_level - min_level) / 9
    levels = np.arange(min_level, max_level + level_step, level_step)

    cnf = plt.contourf(t, np.log2(freqs), np.log2(power),
                 levels = levels, extend="both",
                 cmap=plt.cm.viridis)
    # cbar = plt.colorbar(cnf)
    # cbar.set_ticks(levels)
    # cbar.set_ticklabels(np.char.mod("%.1e", 2.**levels))
    # y_ticks = 2**np.arange(np.ceil(np.log2(freqs.min())),
                           # np.ceil(np.log2(freqs).max()))
    # ax.set_yticks(np.log2(y_ticks))
    # ax.set_yticklabels(y_ticks)
    # ax.set_xlabel("Time [" + r'$s$' + "]")
    # ax.set_ylabel("Frequency [Hz]")


def generate_simulated_data():
    """
    Generate simulated data to test the implementation
    choices. Each distinct behaviour is characterised
    by a set of features, each a superpostion of 
    sine waves corresponding frequencies and amplitudes.
    """

    rd.seed(666)

    # Number of distinct behaviours
    n_b = 10
    # Number of features
    n_f = 5
    # Number of sine-waves per feature
    n_s = 4
    # Length of recorded time series, in seconds
    t_max = 600
    # Capture framerate
    fr = 120
    # Example data
    data = np.zeros(shape = (fr * t_max, n_f))
    # Store labels
    labels = np.zeros(len(data))
    # Time points
    t = np.arange(t_max * fr) / fr
    # Expected length of a behaviour, in seconds
    length = 3
    # Lower frequency bound, Hz
    w_lower = 0.5
    # Upper frequency bound, Hz
    w_upper = 20
    # Mu amplitude parameter
    a_mu = 1 
    # Sigma amplitude parameter
    a_sig = 0.5
    # Determine the corresponding frequencies
    w = w_lower + (w_upper - w_lower) * rd.rand(n_b, n_f, n_s)
    # Determine the corresponding amplitudes
    a = rd.lognormal(mean = a_mu, sigma = a_sig, size = (n_b, n_f, n_s))
    # Gaussian noise added to the features
    noise_std = 0.2

    # Define feature generating function given amplitudes and frequencies
    def feature(freq, ampl, t):
        val = 0
        for i in range(len(freq)):
            val += ampl[i]*np.sin(2*np.pi*freq[i]*t)
        return val + rd.normal(0, noise_std)


    ## Simulate data
    # Choose behavioural changing times
    t_change = np.sort(rd.randint(0, len(data), int(len(data) / length / fr)))
    t_change = np.append(t_change, len(data))
    t_change = np.insert(t_change, 0, 0)
    # Choose behaviour labels
    behaviours = rd.randint(0, n_b, len(t_change))

    # Iterate through behaviours
    for i in range(1, len(t_change)):
        beh = behaviours[i]
        t_c = t_change[i]
        t_vals = t[t_change[i - 1]:t_change[i]]
        labels[t_change[i - 1]:t_change[i]] = np.ones(len(t_vals))*beh

        # Iterate over features
        for j in range(n_f):
            freq = w[beh, j, :]
            ampl = a[beh, j, :]
            temp = lambda time: feature(freq, ampl, time)
            data[t_change[i - 1]:t_change[i], j] = temp(t_vals)
            
    return data, labels, t_change, behaviours, w, t
        

def get_simulated_data():
    """
    Retrieve dictionary of simulated data
    """

    with open("./extracted_data/simulated_data.pkl", "rb") as file:
        df = pickle.load(file)

    return df


def pickle_simulated_data():
    """
    Generate simulated data and pickle it
    for easy retrieval elsewhere.
    """
    data, labels, t_change, behaviours, w, t = generate_simulated_data()
    df = {
            "data": data,
            "labels": labels,
            "t_change": t_change,
            "behaviours": behaviours,
            "freqencies": w,
            "time": t,
    }

    with open("./extracted_data/simulated_data.pkl", "wb") as file:
        pickle.dump(df, file)


def scale_power_spectrum(bc, sqrt = True, standardize = True):
    """
    Finds the power spectrum and scales it 
    corresponding to the variable scale:

    Arguments:
        bc (pipeline): Pipeline containing the data
        sqrt (bool): To take square root of the
            spectrum or not
        standardize (bool): To standardize the 
            spectrum or not
    Returns:
        embedding: t-SNE embedding found
    """
    bc.power = []
    bc.features = []

    # Iterate over animals
    for d in range(len(bc.data)):
        # Create storage for new feature vector
        x_d = np.zeros(shape = (len(bc.data[d]),
                                bc.n_features*
                                (bc.num_freq + 1)))
        # Create storage for power spectrum
        power_d = np.zeros(shape = (bc.n_features,
                                    bc.num_freq,
                                    len(bc.data[d])))

        # Iterate over raw features
        for i in range(bc.n_features):
            # Apply cwt
            wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(
                    bc.data[d][:,i], bc.dt, bc.dj, 1/bc.max_freq,
                    bc.num_freq - 1, bc.mother)
            # Store the frequencies and scales
            if scales.all() != bc.scales.all():
                bc.scales = scales
                bc.freqs = freqs
            # Compute wavelet power spectrum
            power_i = np.abs(wave)**2
            # Normalize over scales (Liu et. al. 07)
            power_i /= scales[:, None]
            # Store power
            power_d[i] = power_i
            if sqrt:
                # Take the square root to ...
                power_i = np.sqrt(power_i)
            
            # Center and rescale trend
            trend = bc.trend[d][:,i]
            trend_std = np.std(trend)
            trend = (trend - np.mean(trend)) / trend_std


            # Store new features
            x_d[:,i] = trend
            x_d[:,bc.n_features + i*bc.num_freq:bc.n_features +(i + 1)*
                bc.num_freq] = power_i.T
    
        bc.features.append(x_d)
        bc.power.append(power_d)

    if standardize:
        bc.standardize_features()
    bc.pca()
    bc.tsne()

    return bc.embedded_train


def plot_methodology(bc):
    """
    Create plot showing the full methodology of
    a trained pipeline.
    """ 
    
    # Feature to plot
    feat = 1
    # Animal
    animal = 0
    # Time interval 
    time = [60*bc.capture_framerate, 120*bc.capture_framerate]
    # Data, 
    ts = bc.data[animal][time[0]:time[1],feat]
    detrend = bc.data_detrended[animal][time[0]:time[1],feat]
    trend = bc.trend[animal][time[0]:time[1],feat]
    # Scalogram
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(
            bc.data[animal][:,feat], bc.dt, bc.dj, 1/bc.max_freq,
            bc.num_freq - 1, bc.mother)
    power = np.abs(wave)**2

    # Raw time series
    ax1 = plt.subplot(421)
    plt.plot(ts, c = "k", lw = 1)
    plt.plot(trend, c = "r", lw = 1)
    # Detrending
    ax2 = plt.subplot(423)
    plt.plot(detrend, c = "b", lw = 1)
    # Scaleogram
    ax3 = plt.subplot(222)
    t1 = 60*bc.capture_framerate
    t2 = 70*bc.capture_framerate
    t = np.arange(len(bc.data[animal]))/bc.capture_framerate
    plot_scaleogram(power[:,t1:t2], t[t1:t2], freqs, scales, ax3)
    # t-SNE embedding
    ax4 = plt.subplot(223)
    plt.scatter(bc.embedded[:,0], bc.embedded[:,1], s = 1)
    # Heatmap + segmentation
    ax5 = plt.subplot(224)
    contours = get_contours(bc.kde, bc.ws_labels)
    plot_watershed_heat(bc.embedded, bc.kde, contours,
                        bc.border - 15)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.subplots_adjust(hspace = 0.1, wspace = 0.1)


def describe_pipeline(bc):
    """
    Description of a trained clustering
    pipeline. Print descriptive statistics
    and plot relevant results.
    """

    print(f"Number of animals: {len(bc.data)}")
    print(f"Number of features: {bc.n_features}")
    total_timepoints = np.sum([len(d) for d in bc.raw_features])
    nonan_timepoints = np.sum([len(d) for d in bc.data])
    print(f"Number of input time points: {total_timepoints}")
    print(f"Non-NaN time points: {nonan_timepoints}")
    print(f"Number of NaN-timepoints: {total_timepoints - nonan_timepoints}")
    print(f"Average length of series: {nonan_timepoints/len(bc.data)/bc.capture_framerate} seconds")
    print(f"Capture framerate: {bc.capture_framerate} Hz")
    print(f"Training points for t-SNE: {len(bc.embedded_train)}")
    print(f"Downsampling rate: {bc.ds_rate} Hz")
    print(f"Number of found behaviors: {len(np.unique(bc.ws_labels))}")
    print(f"Number of principal components used: {bc.n_pca}")


def perplexity_tuning(bc):
    """
    Tries a series of perplexity values on 
    an already trained pipeline.
    Plots the result.
    """
    
    # Perplexity values to be tested
    perp = [5, 30, 50, 100, 200, 500]

    # Iterate over chosen perplexities
    for i in range(len(perp)):
        bc.perp = perp[i]
        bc.tsne()
        bc.pre_embedding()
        bc.kernel_density_estimation(300j)
        bc.watershed_segmentation()

        contours = get_contours(bc.kde, bc.ws_labels)
        
        plt.subplot(2, 3, i + 1)
        plt.title("Perplexity = " + str(perp[i]))
        plot_watershed_heat(bc.embedded, bc.kde,
                            contours, bc.border)

def main():
    pickle_simulated_data()


if __name__ == "__main__":
    main()
