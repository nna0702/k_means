# import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math

# import argparse

plt.style.use("default")
plt.rcParams["font.sans-serif"] = "Arial"


def normalization(feature):
    """
    applies mean normalization to the feature

    Args:
        feature: pandas series

    Returns: panda seroeies
    """
    feature_mean = feature.mean()
    feature_range = max(feature) - min(feature)

    return feature.apply(lambda x: (x - feature_mean) / feature_range)


def initialize(data, K):
    """
    returns K random points from the data

    Args:
        data: dataframe
        K: integer

    Returns: list of np.array
    """
    random_list = data.sample(K)
    random_list = random_list.to_numpy()
    centroid_list = []
    for k in range(K):
        centroid_list.append(random_list[k])

    return centroid_list


def distance(x, y):
    """
    calculates the distance between x and y
    """
    return math.sqrt(np.sum((x - y) ** 2))


def mean_distance(cluster):
    """
    calculates mean distance of all points in a cluster

    Args:
        cluster: list of vector np.array's

    Returns: vector np.array
    """
    return 1 / len(cluster) * sum(cluster)


def cluster_assignment(data, centroid_list, K):

    """
    assigns each point in data to the closest centroid

    Args:
        data: list of np.array
        centroid_list: vector np.array
        K: number of clusters

    Returns: list of np.array
    """

    clusters = [[] for _ in range(K)]

    for i in range(len(data)):

        # Calculate the distance from the point to centroids
        distance_list = []
        for centroid in centroid_list[-K:]:
            distance_list.append(distance(data[i], centroid))

        # Assign the point to the closest centroid
        clusters[distance_list.index(min(distance_list))].append(data[i])

    return clusters


def new_centroid(clusters, centroid_list):
    """
    takes the cluster and returns the updated list of centroids

    Args:
        clusters: list of vector np.array's
        centroid_list: list of vector np.array's

    Returns: vector np.array
    """
    for cluster in clusters:
        centroid_list.append(mean_distance(cluster))
    return centroid_list


def plot_coordinate(coordinate_list):
    """
    returns a list of K arrays for eruptions and waiting time from centroid list

    args:
        coordinate_list: list of np array

    return: list of np array
    """
    eruptions = [coordinate[0] for coordinate in coordinate_list]
    waiting = [coordinate[1] for coordinate in coordinate_list]

    return eruptions, waiting


def plot_iteration(iteration, clusters, centroid_list, K, colors):
    """
    plot the clustering for each iteration

    args:
        iteration: integer
        clusters: list of np arrays
        centroid_list: list of np arrays
        K: integer
        colors: list (of colors corresponding to cluster's centroid)
    """

    fig, ax = plt.subplots(1, 1)

    for cluster, color, centroid in zip(clusters, colors, centroid_list[-K:]):
        plot_list = plot_coordinate(cluster)

        eruptions = plot_list[0]
        waiting = plot_list[1]
        ax.scatter(eruptions, waiting, color=color)

        plot_centroid = plot_coordinate([centroid])
        ax.scatter(plot_centroid[0], plot_centroid[1], color="black", marker="x", s=100)

    # x-axis
    ax.set_xlabel("Duration of the eruption (in mins)")
    ax.xaxis.set_tick_params(direction="in")

    # y axis
    ax.set_ylabel("Waititing times between eruptions (in mins)")
    ax.yaxis.set_tick_params(direction="in")

    # Title
    ax.set_title("Iteration " + str(iteration))

    sns.despine(ax=ax)

    fig.tight_layout()
    path = "plots/iteration" + str(iteration) + ".png"
    fig.savefig(path, bbox_inches="tight")
    print("Saved to {}".format(path))


def get_rgb(color):
    """
    normalizes RGB values
    """
    r, g, b = color
    color = (r / 255.0, g / 255.0, b / 255.0)
    return color


def plot_exploratory(data):
    fig, ax = plt.subplots(1, 1)

    eruption = data["eruptions"]
    waiting = data["waiting"]

    ax.scatter(eruption, waiting, color=get_rgb([124, 252, 0]))

    # x-axis
    ax.set_xlabel("Duration of the eruption (in mins)")
    ax.xaxis.set_tick_params(direction="in")

    # y axis
    ax.set_ylabel("Waititing times between eruptions (in mins)")
    ax.yaxis.set_tick_params(direction="in")

    # Title
    ax.set_title("Exploratory scatterplot")

    sns.despine(ax=ax)

    fig.tight_layout()
    path = "plots/exploratory.png"
    fig.savefig(path, bbox_inches="tight")
    print("Saved to {}".format(path))


def plot_initialization(data, K, centroid_list):

    fig, ax = plt.subplots(1, 1)

    plot_list = plot_coordinate(data)

    eruptions = plot_list[0]
    waiting = plot_list[1]

    ax.scatter(eruptions, waiting, color=get_rgb([124, 252, 0]))

    for centroid in centroid_list:
        plot_centroid = plot_coordinate([centroid])
        ax.scatter(plot_centroid[0], plot_centroid[1], color="black", marker="x", s=100)

    # x-axis
    ax.set_xlabel("Duration of the eruption (in mins)")
    ax.xaxis.set_tick_params(direction="in")

    # y axis
    ax.set_ylabel("Waititing times between eruptions (in mins)")
    ax.yaxis.set_tick_params(direction="in")

    # Title
    ax.set_title("After initialization")

    sns.despine(ax=ax)

    fig.tight_layout()
    path = "plots/initialization.png"
    fig.savefig(path, bbox_inches="tight")
    print("Saved to {}".format(path))
