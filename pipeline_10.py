import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

from behavioral_clustering import BehavioralClustering
from helper_functions import (
        plot_scaleogram, assign_labels,
        get_contours, plot_watershed_heat,
        get_watershed_labels, estimate_pdf,
        plot_methodology,
)

from scipy.spatial.distance import cdist


def get_pipeline():
    cwd = os.getcwd()
    filenames = [path.split("/")[-1] for path in
                 os.listdir(cwd + "/dataset/data_files/")]

    bc = BehavioralClustering()
    bc.set_original_file_path("./dataset/data_files/")
    bc.set_extracted_file_path("./extracted_data/")

    bc.load_relevant_features(filenames)
    # 10 first animals, else RAM overload
    bc.raw_features = bc.raw_features[:10]
    bc.remove_nan()
    print("Detrend")
    bc.detrend()
    print("Time frequency analysis")
    bc.time_frequency_analysis()
    print("Pca")
    bc.pca()
    bc.ds_rate = 1
    print("tsne")
    bc.tsne()
    print("preembedding")
    bc.pre_embedding()
    print("kde")
    bc.kernel_density_estimation(500j)
    print("watershed")
    bc.watershed_segmentation()
    bc.classify()

    return bc


def pickle_pipeline():
    """
    Pickle full model to be easily retrieved 
    """
    path = "./models/pipeline_10.pkl"
    bc = get_pipeline()

    with open(path, "wb") as file:
        pkl.dump(bc, file)


def load_pipeline():
    """
    Loads the pickled full model
    """
    path = "./models/pipeline_10.pkl"
    
    with open(path, "rb") as file:
        bc = pkl.load(file)
    
    return bc


def perplexity_tuning_full():
    """
    Perform the clustering on the full pipeline
    varying the perplexity parameter i t-SNE.
    """

    # Perplexity values to be tested
    perp = [1, 5, 30, 50, 200, 500]

    bc = load_pipeline()
    bc.bw = 0.2

    plt.figure(figsize = (12, 8))
    # Iterate over chosen perplexities
    for i in range(len(perp)):
        bc.perp = perp[i]
        print("Perplexity:", perp[i])
        bc.tsne()
        bc.pre_embedding()
        bc.kernel_density_estimation(500j)
        bc.watershed_segmentation()

        contours = get_contours(bc.kde, bc.ws_labels)
        
        plt.subplot(2, 3, i + 1)
        plt.title("Perplexity = " + str(perp[i]))
        plot_watershed_heat(bc.embedded, bc.kde,
                            contours, bc.border)
    plt.savefig("./figures/perplexity_tuning_full_pipeline.pdf", 
                bbox_inches = "tight")


def bandwidth_tuning():
    """
    Plot the resulting clusters for several values of
    the bandwidth parameter used in the kernel 
    density estimation.
    """

    bc = load_pipeline()

    bandwidths = np.logspace(-5, -1, 10)

    for bandwidth in bandwidths:
        bc.bw = bandwidth

        bc.tsne()
        bc.pre_embedding()
        bc.kernel_density_estimation(500j)
        bc.watershed_segmentation()

        contours = get_contours(bc.kde, bc.ws_labels)
        
        plt.figure()
        plt.title("Bandwidth = " + str(bandwidth))
        plot_watershed_heat(bc.embedded, bc.kde,
                            contours, bc.border)
        plt.savefig("./figures/bandwidth_tuning/clustering_bw_" + str(bc.bw) + ".pdf", 
                bbox_inches = "tight")


def test_embedding():
    """
    Try embedding ALL points to the t-SNE plane found by the downsampled features.
    Only then do we apply the KDE + Watershed segmentation
    """

    bc = load_pipeline()

    # Principal component scores for 
    # points used to find t-SNE embedding.
    pca_train = bc.fit_pca[bc.tsne_ind,:]

    # Create storage for embeddings
    emb_total = np.zeros(shape = (len(bc.fit_pca), 2))

    # Iterate over all time points
    for i in range(len(bc.fit_pca)):
        # Find closest time point in PCA space
        dist = cdist(bc.fit_pca[i,:][np.newaxis,:],
                     pca_train)

        # Choose embedding corresponding to this
        # time point
        emb_total[i] = bc.embedded[dist.argmin()]

    
    # Apply kernel density estimation on all points
    kde, grid = estimate_pdf(emb_total, bc.bw, bc.border,
                             500j)

    ws_labels = get_watershed_labels(kde)
    contours = get_contours(kde, ws_labels)
    bc_contours = get_contours(bc.kde, bc.ws_labels)
        
    plt.figure(figsize = (12, 5))
    plt.subplot(121)
    plot_watershed_heat(bc.embedded, bc.kde,
                        bc_contours, bc.border)
    plt.subplot(122)
    plot_watershed_heat(emb_total, kde,
                        contours, bc.border)
    plt.savefig("./figures/full_embedding.pdf",bbox_inches = "tight")
    plt.show()


def main():
    sns.set_theme()
    # pickle_pipeline()
    bc = load_pipeline()

    # perplexity_tuning_full()
    # bandwidth_tuning()
    # test_embedding()
    
    plt.figure(figsize = (12, 10))
    plot_methodology(bc)
    plt.savefig("./figures/methodology_10.pdf",
                bbox_inches = "tight")
    # plt.show()
    

if __name__ == "__main__":
    main()


