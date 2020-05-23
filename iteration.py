import pandas as pd
import analysis
import numpy as np
import argparse
import os


def export_table(clusters):

    # Convert clusters into a data frame
    cluster_list = []
    for cluster in clusters:
        cluster_list.append(pd.DataFrame(cluster))
    output = pd.concat([frame for frame in cluster_list], ignore_index=True)

    # Create a cluster column
    final_cluster = []
    cluster_type = 0

    for cluster in clusters:
        cluster_type += 1
        final_cluster = final_cluster + [cluster_type] * len(cluster)

    # Add the cluster column to data frame
    output["final_cluster"] = pd.Series(final_cluster)

    # Save the table
    output.to_csv("outputs/output.csv")
    print("Saved output.csv to outputs")


if __name__ == "__main__":

    # Remove old files
    for filename in os.listdir("plots/"):
        if "iteration" in filename:
            path = f"plots/{filename}"
            os.remove(path)
            print(f"Deleted {path}")

    # Import data
    data = pd.read_csv("data/faithful.csv")

    # Remove first column
    data = data.drop(data.columns[0], axis=1)

    # Normalize features
    for column in data.columns:
        data[column] = analysis.normalization(data[column])

    # Set args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--K", type=int, default=2, help=" ")  # Number of clusters
    parser.add_argument("--x", type=str, default="waiting", help=" ")  # Variable x in case of 2D
    parser.add_argument("--y", type=str, default="eruptions", help=" ")  # Variable y in case of 2D
    args = parser.parse_args()

    # Randomly initialize K points
    centroid_list = analysis.initialize(data, args.K)

    # Convert data into array
    data_array = data.to_numpy()

    # Produce plots for 2D
    if len(data.columns) == 2:

        # Plot exploratory data
        analysis.plot_exploratory(data_array, args.x, args.y)

        # Plot data with initialization
        analysis.plot_initialization(data_array, args.K, centroid_list, args.x, args.y)

    # Find the clusters through iterations

    iteration = 0

    while True:
        iteration += 1
        clusters = analysis.cluster_assignment(data_array, centroid_list, args.K)
        centroid_list = analysis.new_centroid(clusters, centroid_list)

        if len(data.columns) == 2:

            colors = [(214, 39, 40), (23, 190, 207), (148, 0, 211), (128, 128, 0), (165, 42, 42)]
            colors = [analysis.get_rgb(color) for color in colors]

            analysis.plot_iteration(
                iteration, clusters, centroid_list, args.K, colors, args.x, args.y,
            )

            if np.allclose(centroid_list[-(args.K * 2) : -args.K], centroid_list[-args.K :]):
                break

        else:

            if np.allclose(centroid_list[-(args.K * 2) : -args.K], centroid_list[-args.K :]):
                break

    # Table output
    export_table(clusters)
