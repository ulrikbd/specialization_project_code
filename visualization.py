import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn

from helper_functions import *

from behavioral_clustering import BehavioralClustering

from example import load_example


def describe(array):
    """
    Show descriptive statistics for an numpy.ndarray
    """
    n_nan = np.sum(np.isnan(array))
    print(f'Length: {len(array)}, #NaN: {n_nan}, %NaN: {n_nan / len(array) * 100:.2f}%')
    print(f'Min: {np.nanmin(array):.2f}, Max: {np.nanmax(array):.2f}, '
          f'Mean: {np.nanmean(array):.2f}, Median: {np.nanmedian(array):.2f}')


def show_feature_information(relevant_features):
    """
    Print out useful information about the features
    """
    for i in range(relevant_features.shape[1]):
        print(f'Feature {i + 1}:')
        describe(relevant_features[:,i])
        print()

    
def get_extracted_filename(original_name):
    """
    Returns filename which specifies the relevant features
    are extracted.
    """
    return original_name.split(".")[0] + "_extracted.pkl"


def plot_detrending(bc):
    """
    Visualizes the detrending performed on the 
    actual data.
    """
    

    for animal_ind in range(len(bc.data)):
        for feat_ind in range(bc.n_features):
            y = bc.data[animal_ind][:, feat_ind]
            x = bc.data_detrended[animal_ind][:, feat_ind]
            s = bc.trend[animal_ind][:, feat_ind]
            t = np.arange(len(y)) / bc.capture_framerate

            plt.figure(figsize = (12, 10))
            ax1 = plt.subplot(2, 1, 1)
            plt.xticks(np.arange(0, t[-1], 60))
            plt.plot(t, y, c = "k", lw = 0.7,  label = r'$y(t)$')
            plt.plot(t, s, c = "r", label = r'$s_3(t)$')
            plt.tick_params('x', labelbottom = False)
            plt.legend(loc = "lower right", prop = {"size": 15},
                       frameon = False, framealpha = 0)

            ax2 = plt.subplot(2, 1, 2, sharex = ax1)
            plt.plot(t, x, c = "b", lw = 0.6, label = r'$x(t)$')
            plt.xlabel("Time [" + r'$s$' + "]")
            plt.subplots_adjust(hspace = 0.03)
            plt.legend(loc = "upper right", prop = {"size": 15},
                       frameon = False, framealpha = 0)
            path = "./figures/detrending/"
            path += ("detrending_animal_" + str(animal_ind) +
                    "_" + bc.feature_names[feat_ind] +
                    ".pdf")
            plt.savefig(path, bbox_inches = "tight")
            # plt.show()


def plot_scaleograms(bc):
    """
    Plot scaleograms for all the computed wavelet 
    transforms, i.e., for all animals and features.
    """ 

    for animal_ind in range(len(bc.data)):
        for feat_ind in range(bc.n_features):
            power = bc.power[animal_ind][feat_ind] 
            t = np.arange(len(bc.data[animal_ind])) / bc.capture_framerate

            # Plot the first 10 seconds in the third minute
            min_ind = 180*bc.capture_framerate
            max_ind = min_ind + 10*bc.capture_framerate + 1
            
            plt.figure(figsize = (12, 5))
            plot_scaleogram(power[:,min_ind:max_ind], t[min_ind:max_ind],
                            bc.freqs, bc.scales)
            path = "./figures/scaleograms/"
            path += ("scaleogram_animal_" + str(animal_ind) +
                    "_" + bc.feature_names[feat_ind] +
                    ".pdf")
            plt.savefig(path, bbox_inches = "tight")
            

def plot_tsne(bc):
    """
    Show a scatter plot of the t-SNE embedding on the 
    downsampled points
    """


    plt.figure(figsize = (12, 5))
    plt.scatter(bc.embedded[:,0], bc.embedded[:, 1],
                marker = ".")
    plt.show()


            
def plot_kde(bc):
    """
    Plot the kernel density estimation of the tsne embedding
    """


    plt.figure(figsize = (12, 5))
    plt.imshow(bc.kde, cmap = "coolwarm")
    # plt.show()


def plot_watershed(bc):
    """
    Plots the watershed segmentation of the kernel
    kernel density estimation.
    """

    
    contours = get_contours(bc.kde, bc.ws_labels)

    outside = np.ones(bc.kde.shape)
    outside[bc.kde == 0] = 0

    xmax, ymax = np.max(bc.embedded, axis = 0) + bc.border
    xmin, ymin = np.min(bc.embedded, axis = 0) - bc.border

    plt.figure(figsize = (12, 5))
    plot_watershed_heat(bc.embedded, bc.kde, contours,
                        bc.border)
    # plt.show()
     


def main():
    seaborn.set_theme()
    bc = load_example()
    plot_detrending(bc)
    plot_scaleograms(bc)
    plot_tsne(bc)
    plot_kde(bc)
    plot_watershed(bc)


if __name__ == "__main__":
    main()

