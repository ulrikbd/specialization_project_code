import numpy as np
import pickle as pkl

from helper_functions import (
        pickle_relevant_features, spline_regression,
        plot_scaleogram, estimate_pdf, 
        get_watershed_labels, assign_labels,
)

import pycwt as wavelet
from pycwt.helpers import find

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE

from scipy.spatial.distance import cdist


class BehavioralClustering():
    """
    Wrapper class for clustering distinct behaviours
    in rats from recorded postural dynamics.

    Attributes:
        train_file_names (List(str)): Name of raw data 
            to be used in the clustering
        original_file_path (str): Location of the training
            files.
        extracted_file_path (str): Location of the pickled 
            files containing only relevant features.
        raw_features (List(np.ndarray)): The features relevant to 
            the analysis, straight from the input files.
        knot_frequency (float): Interior spacing of the knots
            used in the spline regression
        spline_dim (int): Dimension of the regression spline
            used in detrending the data
        trend (list(np.ndarray)): The trend found in the 
            time series by spline regression
        data_detrended (list(np.ndarray)): The detrended
            time series.
        feature_names (list(str)): Names of the features
            used
        n_features (int): Number of different recorded
            features, i.e., number of movements recorded
            per animal.
        capture_framerate (int): Frequency (Hz) of the 
            recorded postural time series.
        dt (float): Time interval in the time series
        used_indices (list(np.ndarray)): Indices of non-NaN values
        num_freq (int): Number of dyadically spaced frequency
            scales used in the CWT
        min_freq (float): Minumum frequency scale used in 
            the CWT
        max_freq (float): Maximum frequency scale used in 
            the CWT
        dj (float): The number of sub-octaves per octave,
            logarithmic spacing for the scales used
            in the CWT
        mother (str): Mother wavelet used in the CWT
        power (list(np.ndarray)): The computed power
            spectrum from the cwt
        features (list(np.ndarray): Features extracted 
            from the time frequency analysis
        scales (np.ndarray): Stores the cwt scales
        freqs (np.ndarray): Stores the cwt frequencies
        var_pca (np.ndarray): Percentage ratios of the 
            explained variance in the principal component
            analysis
        n_pca (int): Number of principal components
            explaining more than 95% of the data
        fit_pca (np.ndarray): The transformed (reduced) 
            features using principal component analysis
        ds_rate (int): Downsampling frequency in [Hz],
            used as t-SNE memory complexity is O(n^2)
        perp (float): Perplexity parameter used in t-SNE
        embedded_train (np.ndarray): Two dimensional embedding
            obtained by t-SNE on the downsampled data
        embedded (np.ndarray): The low dimensional embedding
            of all points
        tsne_ind (np.ndarray): Indices of time points used
            when finding the t-SNE embedding
        kde (np.ndarray): Kernel density estimation of
            the t-SNE embedding
        border (int): Border around the t-SNE embedding
            for improved watershed segmentation, and 
            visualization
        grid (list(np.ndarray)): Grid on which the 
            kernel density estimation is applied
        bw (float/str): Bandwidth method used in the kernel
            density estimation
        ws_labels (np.ndarray): Assigned clusters
            found by watershed segmentation
        beh_labels (np.ndarry): Final classification of
            the non-nan time points
    """

    def __init__(self):
        self.train_file_names = []
        self.original_file_path = None
        self.extracted_file_path = None
        self.raw_features = [] 
        self.knot_frequency = 0.5
        self.spline_dim = 3
        self.trend = []
        self.data_detrended = []
        self.feature_names = ["exp0", "exp1", "exp2",
                              "speed2", "BackPitch",
                              "BackAzimuth", "NeckElevation"]
        self.n_features = len(self.feature_names)
        self.capture_framerate = 120 
        self.dt = 1/self.capture_framerate
        self.used_indices = []
        self.data = []
        self.num_freq = 18
        self.min_freq = 0.5
        self.max_freq = 20
        self.dj = 1/(self.num_freq - 1)*np.log2(
                self.max_freq/self.min_freq)
        self.mother = "morlet"
        self.power = []
        self.features = []
        self.scales = np.zeros(self.num_freq)
        self.freqs = np.zeros(self.num_freq)
        self.var_pca = None
        self.n_pca = None
        self.fit_pca = None
        self.ds_rate = 2
        self.perp = 30
        self.embedded_train = None
        self.embedded = None
        self.kde = None
        self.border = None
        self.grid = None
        self.bw = "scott" 
        self.ws_labels = None
        self.beh_labels = None


    def remove_nan(self):
        """
        Removes each time point where one or more value
        in the time series contain a NaN value.
        Stores the new time series data in 
        an attribute, together with the used row indices.
        """
                
        # Iterate over animals
        for i in range(len(self.raw_features)):
            # Find the indices of usable rows
            self.used_indices.append(~np.isnan(
                self.raw_features[i]).any(axis =1))
            # Filter the data and add to an attribute
            self.data.append(self.raw_features[i][self.used_indices[i]])



    def detrend(self):
        """
        Detrends the time series individually
        using spline regression, and stores the
        trend and detrended data as attributes
        """
       
       # Iterate over animals
        for d in range(len(self.data)):
            # Perform spline regression on each time series
            trend = [spline_regression(y, self.spline_dim,
                        self.capture_framerate,
                        self.knot_frequency) for y in self.data[d].T]
            self.trend.append(np.array(trend).T)
            self.data_detrended.append(self.data[d] - self.trend[d])
            # Normalize data
            std = np.std(self.data_detrended[-1], axis = 0)
            self.data_detrended[-1] = self.data_detrended[-1] / std


    def time_frequency_analysis(self):
        """
        Perform time frequency analysis
        using the continuous wavelet transform on the 
        detrended time series data. The power spectrum
        is computed using the pycwt python package,
        and divided by the scales. Taking the square root,
        centering and rescaling by the trend standard 
        deviation, the extended time series are concatenated
        with the normalized trend data creating new
        features. We store the power matrix, scales and 
        frequencies to be used for plotting scaleograms.
        """ 

        # Iterate over animals
        for d in range(len(self.data)):
            # Create storage for new feature vector
            x_d = np.zeros(shape = (len(self.data[d]),
                                    self.n_features*
                                    (self.num_freq + 1)))

            # Iterate over raw features
            for i in range(self.n_features):
                # Apply cwt
                wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(
                        self.data[d][:,i], self.dt, self.dj, 1/self.max_freq,
                        self.num_freq - 1, self.mother)
                # Store the frequencies and scales
                if scales.all() != self.scales.all():
                    self.scales = scales
                    self.freqs = freqs
                # Compute wavelet power spectrum
                power_i = np.abs(wave)**2
                # Normalize over scales (Liu et. al. 07)
                power_i /= scales[:, None]
                # Take the square root to ...
                power_i = np.sqrt(power_i)
                
                # Center and rescale trend
                trend = self.trend[d][:,i]
                trend_std = np.std(trend)
                trend = (trend - np.mean(trend)) / trend_std

                # Center and rescale power spectrum 
                power_i = (power_i - np.mean(power_i)) / trend_std    
                # Store new features
                x_d[:,i] = trend
                x_d[:,self.n_features + i*self.num_freq:self.n_features +(i + 1)*
                    self.num_freq] = power_i.T
        
            self.features.append(x_d)

    
    def standardize_features(self):
        """
        Standardizes the features extracted via CWT such that they can
        be used in PCA, i.e., with mean zero and variance one.
        """
        
        # Iterate over animals
        for i in range(len(self.data)):
            self.features[i] = ((self.features[i] - np.mean(self.features[i],
                                                            axis = 0)) /
                                np.std(self.features[i], axis = 0))

        
    def pca(self):
        """
        Compute the principal components explaining 95% of
        the variance in features extracted via CWT.
        Stores the new reduced features as an attribute.
        """

        # Concatenate features
        features = np.concatenate(self.features, axis = 0)
        
        # Find principal components
        pca = PCA()
        pca.fit(features)

        # Find number of features explaning 95% of the variance
        self.var_pca = pca.explained_variance_ratio_
        self.n_pca = np.argmax(np.cumsum(self.var_pca) > 0.95) + 1

        # Apply the transformation using the sufficient
        # number of principal components
        pca = PCA(n_components = self.n_pca)
        self.fit_pca = pca.fit_transform(features)
          

    def tsne(self):
        """
        Embed the principal component scores onto a 
        two dimensional manifold using t-SNE.
        Previous to the dimensionality reduction 
        we downsample to reduce memory complexity in
        the algorithm.
        """

        # Downsample the principal component scores
        self.tsne_ind = np.arange(0, len(self.fit_pca),
                        int(self.capture_framerate / self.ds_rate))
        train = self.fit_pca[self.tsne_ind,:] 

        # Perform t-SNE
        self.embedded_train = TSNE(n_components = 2,
                                perplexity = self.perp,
                                init = "pca").fit_transform(train)
         

    def pre_embedding(self):
        """
        Embeds all data points into the t-SNE plane
        by choosing the components of the nearest 
        neighbor (in the training set) by euclidean
        distance in the PCA space.
        """ 

        # Principal component scores for 
        # points used to find t-SNE embedding.
        pca_train = self.fit_pca[self.tsne_ind,:]

        # Create storage for embeddings
        self.embedded = np.zeros(
                shape = (len(self.fit_pca), 2))

        # Iterate over all time points
        for i in range(len(self.fit_pca)):
            # Find closest time point in PCA space
            dist = cdist(self.fit_pca[i,:][np.newaxis,:],
                         pca_train)

            # Choose embedding corresponding to this
            # time point
            self.embedded[i] = self.embedded_train[dist.argmin()]

    
    def kernel_density_estimation(self, pixels):
        """
        Perform kernel density estimation on the t-SNE 
        embedding, estimating the pdf using Gaussian kernels.

        Arguments:
            pixels (int): Pixelation aloach each axis which
                the pdf is computed on
        """

        self.border = np.max(np.abs(self.embedded))/5

        # Outer border set for better visualizations
        self.kde, self.grid = estimate_pdf(
                self.embedded, self.bw, self.border, pixels)


    def watershed_segmentation(self):
        """
        Perform watershed segmentation on the 
        kernel density estimation pdf.
        """

        self.ws_labels = get_watershed_labels(self.kde)   


    def classify(self):
        """
        Assigns a behavioral action label to 
        all (non-nan) time points in the data set.
        """

        # Classify the embedded points
        self.beh_labels = assign_labels(self.embedded, 
                                   self.ws_labels,
                                   self.grid)
         

    def set_original_file_path(self, original_file_path):
        self.original_file_path = original_file_path


    def set_extracted_file_path(self, extracted_file_path):
        self.extracted_file_path = extracted_file_path


    def pickle_relevant_features(self, file_names):
        """
        Extract the relevant raw features from the original
        data, pickle it, and dump it in the location
        specified in the extraced_file_path attribute

        Parameters:
            file_names (List(str)): File names of original data
        """
        for name in file_names:
            pickle_relevant_features(self.original_file_path + name,
                                     self.extracted_file_path +
                                     name.split(".")[0] + 
                                     "_extracted.pkl")


    def load_relevant_features(self, train_files):
        """
        Loads the pickled raw relevant features into the 
        instance attribute to be used for training.
        Stores number of features as attribute.

        Parameters:
            train_files (List(str)): Names of training files
        """

        self.train_file_names = train_files
        for filename in train_files:
            with open(self.extracted_file_path + 
                      filename.split(".")[0] + "_extracted.pkl",
                      "rb") as file:
                self.raw_features.append(pkl.load(file))

        # Store the number of features
        self.n_features = self.raw_features[0].shape[1]


     

    
