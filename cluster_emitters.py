import sys
from pathlib import Path
import numpy as np

import polars as pl
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split

import hdbscan 

from parse_data import load_window


def compute_cluster_purity(df, true_label_col="true_label", cluster_label_col="cluster_label"):
    """
    Compute the purity and size of each cluster, as well as the overall weighted purity.
    
    Purity of a cluster is defined as:
        purity = (number of points in the dominant true label) / (total points in cluster)
    
    Returns:
        cluster_purities: A dictionary mapping each cluster label to a tuple (purity, cluster_size).
        overall_purity: The weighted average purity across all clusters.
    """
    cluster_purities = {}
    total_points = len(df)
    weighted_sum = 0
    
    # Group by cluster label.
    for cluster, group in df.groupby(cluster_label_col):
        counts = group[true_label_col].value_counts()
        dominant_count = counts.iloc[0]
        purity = dominant_count / counts.sum()
        cluster_size = len(group)
        cluster_purities[cluster] = (purity, cluster_size)
        
        # Weight by the number of points in the cluster.
        weighted_sum += purity * cluster_size
    
    overall_purity = weighted_sum / total_points if total_points > 0 else 0
    return cluster_purities, overall_purity


def evaluate_hdbscan(
    df: pl.DataFrame,
    varlist: list[str],
    test_size: float = 0.3,
    random_state: int = 0,
    min_cluster_size: int = 1,
):
    """
    Split the data into training and testing sets, apply HDBSCAN clustering
    on the test set (with the labels masked during clustering), and evaluate
    how well the clusters reflect the true labels using external metrics.
    
    Also, compute a cluster purity measure: we are happy if every specific cluster
    does NOT contain a mix of labels (i.e. each cluster is "pure").

    The HDBSCAN parameters are set such that the algorithm is very unlikely to
    produce any outliers.

    Parameters:
        df (pl.DataFrame): The input Polars DataFrame.
        varlist (list[str]): List of feature names to use (e.g., ["elevation", "azimuth"]).
        test_size (float): Proportion of data to use as the test set.
        random_state (int): Random state for reproducibility.
        min_cluster_size (int): The minimum size of clusters for HDBSCAN.

    Returns:
        test_df (pd.DataFrame): The test DataFrame with additional columns "true_label" and "cluster_label".
    """
    # Convert the Polars DataFrame to a pandas DataFrame.
    df_pd = df.to_pandas()

    # Extract features and labels.
    # Assumes there is a column "emitter" containing the ground truth labels.
    X = df_pd[varlist]
    y = df_pd["emitter"]

    # Perform a train/test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Standardize features using the training set's statistics.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply HDBSCAN clustering on the test set.
    # By setting min_samples=1 we encourage every point to join a cluster,
    # reducing the chance of outliers (points labeled as -1).
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=10,
        cluster_selection_epsilon=0.2,
    )
    cluster_labels = clusterer.fit_predict(X_test_scaled)

    print("HDBSCAN clustering is done.")

    # Build a test DataFrame with the true labels and the cluster assignments.
    test_df = X_test.copy()
    test_df["true_label"] = y_test.values
    test_df["cluster_label"] = cluster_labels

    # Evaluate the clustering performance using external metrics.
    ari = adjusted_rand_score(test_df["true_label"], test_df["cluster_label"])
    nmi = normalized_mutual_info_score(test_df["true_label"], test_df["cluster_label"])
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")

    # Print the number of unique true labels and the number of unique clustering labels.
    num_true_labels = test_df["true_label"].nunique()
    num_cluster_labels = test_df["cluster_label"].nunique()
    print(f"Number of unique true labels: {num_true_labels}")
    print(f"Number of unique clustering labels: {num_cluster_labels}")

    # Compute cluster purity to check if each cluster is mostly one label.
    cluster_purities, overall_purity = compute_cluster_purity(test_df)
    print("Cluster purities (per cluster):")
    for cluster, (purity, cluster_size) in cluster_purities.items():
        print(f"  Cluster {cluster}: Purity = {purity:.4f} (Size = {cluster_size})")
    print(f"Overall weighted purity: {overall_purity:.4f}")

    return test_df


if __name__ == "__main__":
    data_file = Path(sys.argv[1])
    df = load_window(data_file)
    feature_list = [
        #"arrival_time",
        "elevation",
        "azimuth",
        "amplitude",
        "frequency",
        "pulse_width",
    ]

    # Adjust min_cluster_size as needed based on your data characteristics.
    test_df = evaluate_hdbscan(df, feature_list, test_size=0.9, random_state=0, min_cluster_size=5)
    print(test_df.head())
