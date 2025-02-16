import sys
from pathlib import Path
import numpy as np
import polars as pl
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture

from parse_data import load_window


def compute_cluster_purity(df, true_label_col="true_label", cluster_label_col="cluster_label"):
    """
    Compute the purity and size of each cluster, as well as the overall weighted purity.
    """
    cluster_purities = {}
    total_points = len(df)
    weighted_sum = 0

    for cluster, group in df.groupby(cluster_label_col):
        counts = group[true_label_col].value_counts()
        dominant_count = counts.iloc[0]
        purity = dominant_count / counts.sum()
        cluster_size = len(group)
        cluster_purities[cluster] = (purity, cluster_size)
        weighted_sum += purity * cluster_size

    overall_purity = weighted_sum / total_points if total_points > 0 else 0
    return cluster_purities, overall_purity


def evaluate_gmm(
    df: pl.DataFrame,
    varlist: list[str],
    test_size: float = 0.3,
    random_state: int = 0,
    n_components: int = 20,
    covariance_type: str = "full",  # 'full', 'diag', 'tied', or 'spherical'
    reg_covar: float = 1e-6,        # regularization term for covariance matrices
    n_init: int = 1,                # number of initializations
    max_iter: int = 100,            # maximum number of iterations
):
    """
    Split the data into training and testing sets, apply a Gaussian Mixture Model (GMM)
    clustering on the test set, and evaluate how well the clusters reflect the true labels.

    Important parameters exposed for tuning:
        - covariance_type: Controls the shape of the clusters.
        - reg_covar: Regularization added to covariance for numerical stability.
        - n_init: Number of initializations to avoid local minima.
        - max_iter: Maximum number of iterations for convergence.

    Parameters:
        df (pl.DataFrame): The input Polars DataFrame.
        varlist (list[str]): List of feature names.
        test_size (float): Proportion of data to use as the test set.
        random_state (int): Random state for reproducibility.
        n_components (int): Number of mixture components (clusters).
        covariance_type (str): Covariance type.
        reg_covar (float): Regularization parameter.
        n_init (int): Number of initializations.
        max_iter (int): Maximum number of iterations.

    Returns:
        test_df (pd.DataFrame): The test DataFrame with additional columns "true_label" and "cluster_label".
    """
    # Convert the Polars DataFrame to a pandas DataFrame.
    df_pd = df.to_pandas()

    # Extract features and labels (assumes "emitter" column holds the ground truth).
    X = df_pd[varlist]
    y = df_pd["emitter"]

    # Split the data into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Standardize features using the training set's statistics.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Configure and apply the Gaussian Mixture Model.
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    gmm.fit(X_train_scaled)
    cluster_labels = gmm.predict(X_test_scaled)
    print("GMM clustering is done.")
    print(f"Used parameters: covariance_type={covariance_type}, reg_covar={reg_covar}, n_init={n_init}, max_iter={max_iter}")

    # Build a test DataFrame with the true labels and cluster assignments.
    test_df = X_test.copy()
    test_df["true_label"] = y_test.values
    test_df["cluster_label"] = cluster_labels

    # Evaluate clustering performance using ARI and NMI.
    ari = adjusted_rand_score(test_df["true_label"], test_df["cluster_label"])
    nmi = normalized_mutual_info_score(test_df["true_label"], test_df["cluster_label"])
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")

    # Display unique label counts.
    num_true_labels = test_df["true_label"].nunique()
    num_cluster_labels = test_df["cluster_label"].nunique()
    print(f"Number of unique true labels: {num_true_labels}")
    print(f"Number of unique clustering labels: {num_cluster_labels}")

    # Compute and display cluster purities.
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
        #"elevation",
        #"azimuth",
        "amplitude",
        "frequency",
        "pulse_width",
    ]

    # Feel free to adjust the following parameters to experiment with the GMM:
    test_df = evaluate_gmm(
        df,
        feature_list,
        test_size=0.9,
        random_state=0,
        n_components=150,
        covariance_type="full",  # options: "full", "diag", "tied", "spherical"
        reg_covar=1e-6,
        n_init=2,
        max_iter=50,
    )
    print("\nHead of final test DataFrame:")
    print(test_df.head())
