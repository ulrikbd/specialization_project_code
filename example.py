import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import seaborn

from behavioral_clustering import BehavioralClustering
from helper_functions import (
        plot_scaleogram, assign_labels,
        scale_power_spectrum, estimate_pdf,
        get_contours, get_watershed_labels,
        plot_watershed_heat, describe_pipeline,
        perplexity_tuning,
)

from scipy.spatial.distance import cdist


def get_example_pipeline():
    filenames = [
        "26148_020520_bank0_s1_light.pkl",
        "26148_030520_bank0_s2_light.pkl",
    ]

    bc = BehavioralClustering()
    bc.set_original_file_path("./dataset/data_files/")
    bc.set_extracted_file_path("./extracted_data/")
    
    bc.load_relevant_features(filenames)
    bc.remove_nan()
    bc.detrend()
    bc.time_frequency_analysis()
    bc.pca()
    bc.tsne()
    bc.pre_embedding()
    bc.kernel_density_estimation(500j)
    bc.watershed_segmentation()
    bc.classify()
    return bc


def pickle_example():
    """
    Pickle example model to be easily retrieved 
    """
    path = "./models/example.pkl"
    bc = get_example_pipeline()

    with open(path, "wb") as file:
        pkl.dump(bc, file)


def load_example():
    """
    Loads the pickled example model
    """
    path = "./models/example.pkl"
    
    with open(path, "rb") as file:
        bc = pkl.load(file)
    
    return bc


def test_scaling_example():
    """
    Try the clustering algorithm with various 
    scaling choices for the wavelet spectrum.
    1. Square root + standardization
    2. Square root + no standardization
    """

    ## 1     
    bc = load_example() 
    emb_1 = scale_power_spectrum(
            bc, sqrt = True, standardize = True)
    ## 2 
    bc = load_example()
    emb_2 = scale_power_spectrum(
            bc, sqrt = True, standardize = False)


    plt.figure(figsize = (12, 5))
    plt.subplot(121)
    plt.scatter(emb_1[:,0], emb_1[:,1], s = 2)
    plt.title("Square root w/standarization")
    plt.subplot(122)
    plt.scatter(emb_2[:,0], emb_2[:,1], s = 2)
    plt.title("Square root wo/standarization")
    plt.savefig("./figures/test_scaling_example.pdf", 
                bbox_inches = "tight")
    plt.show()


def test_pre_embedding():
    """
    Test effects of pre-embedding all points 
    into t-SNE plane by euclidean distance in PCA
    space.
    """

    bc = load_example()

    
    # Apply kernel density estimation on all points
    kde, grid = estimate_pdf(bc.embedded_train, bc.bw, bc.border,
                             500j)

    ws_labels = get_watershed_labels(kde)
    contours = get_contours(kde, ws_labels)
    bc_contours = get_contours(bc.kde, bc.ws_labels)
        
    plt.figure(figsize = (12, 10))
    plt.subplot(221)
    plt.scatter(bc.embedded_train[:,0], bc.embedded_train[:,1], s = 2)
    plt.title("Training data, " + str(len(bc.tsne_ind)) + " points")
    plt.subplot(222)
    plt.scatter(bc.embedded[:,0], bc.embedded[:,1], s = 2)
    plt.title("Embedded data, " + str(len(bc.fit_pca)) + " points")
    plt.subplot(223)
    plot_watershed_heat(bc.embedded_train, kde,
                        contours, bc.border)
    plt.title("Segmentation wo/pre embedding") 
    plt.subplot(224)
    plot_watershed_heat(bc.embedded, bc.kde,
                        bc_contours, bc.border)
    plt.title("Segmentation w/pre embedding") 
    plt.savefig("./figures/test_pre_embed.pdf",bbox_inches = "tight")
    plt.show()


def main():
    seaborn.set_theme()
    pickle_example()
    bc = load_example()
    print(bc.border)
    # test_scaling_example()
    # test_pre_embedding()
    describe_pipeline(bc)
    fig = plt.figure()
    contours = get_contours(bc.kde, bc.ws_labels)
    plot_watershed_heat(bc.embedded, bc.kde, contours, bc.border)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()

    # plt.figure(figsize = (12, 8))
    # perplexity_tuning(bc)
    # plt.show()


    

if __name__ == "__main__":
    main()


