import analysis
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def centroid_iteration(data, centroid_list, K):

    iteration = 0

    while True:
        iteration += 1
        clusters = analysis.cluster_assignment(data, centroid_list, K)
        centroid_list = analysis.new_centroid(clusters, centroid_list)

        # Stop when the last two sets of centroids are equal
        if np.allclose(centroid_list[-(K * 2) : -K], centroid_list[-K:]):
            break

    return centroid_list[-K:], clusters


def cost_function(final_centroid, clusters, K):
    """
    returns the cost function

    args:
        final_centroid: list of np array's
        cluster: list of np array's

    returns: float
    """
    # final_centroid = centroid_list[-K: ]
    distance_list = []

    for i in range(len(clusters)):

        for point in clusters[i]:
            distance_list.append(analysis.distance(point, final_centroid[i]))

    return 1 / len(distance_list) * sum(distance_list)


def plot_elbow(num_cluster, cost, file_name):

    """
    plots the cost function by number of clusters

    Args:
        num_cluster: integer
        cost: list
        file_name: string
    """

    fig, ax = plt.subplots(1, 1)

    ax.plot(num_cluster, cost, marker="o")

    ax.set_xlabel("Number of clusters")
    ax.set_xticks(num_cluster)
    ax.xaxis.set_tick_params(direction="in")

    ax.set_ylabel("Cost")
    ax.yaxis.set_tick_params(direction="in")
    ax.set_title("Elbow method")

    sns.despine(ax=ax)

    fig.tight_layout()
    path = "plots/elbow_" + file_name + ".png"
    fig.savefig(path, bbox_inches="tight")
    print("Saved to {}".format(path))


if __name__ == "__main__":

    # Import data
    data = pd.read_csv("data/faithful.csv")

    # Remove first column
    data = data.drop(data.columns[0], axis=1)

    # Convert data into array
    data_array = data.to_numpy()

    # Set the range of number of clusters
    num_cluster = list(range(1, 6))

    # Calculate cost function for each number of clusters
    cost = []
    for K in num_cluster:
        centroid_list = analysis.initialize(data, K)
        data_array = data.to_numpy()
        final_centroid, clusters = centroid_iteration(data_array, centroid_list, K)
        cost.append(cost_function(final_centroid, clusters, K))

    # Plot elbow
    plot_elbow(num_cluster, cost, "faithful")
