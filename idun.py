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
        describe_pipeline, perplexity_tuning,
)

from scipy.spatial.distance import cdist



def load_data_to_pipeline():
    """
    Create a pipeline which includes the input data,
    but no analysis is performed
    """
    cwd = os.getcwd()
    filenames = [path.split("/")[-1] for path in
                 os.listdir(cwd + "/dataset/data_files/")]

    bc = BehavioralClustering()
    bc.set_original_file_path("./dataset/data_files/")
    bc.set_extracted_file_path("./extracted_data/")

    bc.load_relevant_features(filenames)
    
    return bc


def pickle_idun_pipeline(bc):
    """
    Pickle model to be easily retrieved,
    and finished by the IDUN hpc cluster
    """

    path = "/cluster/work/ulrikbd/specialization_project/code/models/full_pipeline.pkl"

    with open(path, "wb") as file:
        pkl.dump(bc, file)


def load_pipeline():
    """
    Loads the pickled full model
    """
    path = "/cluster/work/ulrikbd/specialization_project/code/models/full_pipeline.pkl"
    
    with open(path, "rb") as file:
        bc = pkl.load(file)
    
    return bc


def train_model(bc):
    """
    Trains the full model
    """

    # bc.remove_nan()
    # bc.detrend()
    # bc.time_frequency_analysis()
    # bc.standardize_features()
    # bc.pca()
    # bc.ds_rate = 0.5
    # bc.tsne()
    # bc.pre_embedding()
    bc.border = np.max(np.abs(bc.embedded))/5
    bc.kernel_density_estimation(500j)
    bc.watershed_segmentation()
    bc.classify()

    return bc


def main():
    sns.set_theme()
    # bc = load_data_to_pipeline()
    # pickle_idun_pipeline(bc)
    #bc = load_pipeline()
    #bc = train_model(bc)
    #pickle_idun_pipeline(bc)
    bc = load_pipeline()

    describe_pipeline(bc)

    fig = plt.figure()
    contours = get_contours(bc.kde, bc.ws_labels)
    plot_watershed_heat(bc.embedded, bc.kde, contours, bc.border)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.savefig("/cluster/work/ulrikbd/specialization_project/code/final_heatmap.pdf", bbox_inches = "tight")

    plt.figure(figsize = (12, 8))
    perplexity_tuning(bc)
    plt.savefig("/cluster/work/ulrikbd/specialization_project/code/perplexity_tuning.pdf", bbox_inches = "tight")
    


if __name__ == "__main__":
    main()
