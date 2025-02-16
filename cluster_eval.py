import sys
from pathlib import Path
import numpy as np
import polars as pl
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

from parse_data import load_window


def cluster_unlabeled_data(
    df: pl.DataFrame,
    varlist: list[str],
    n_components: int = 20,
    covariance_type: str = "full",  # Options: 'full', 'diag', 'tied', or 'spherical'
    reg_covar: float = 1e-6,        # Regularization term for covariance matrices
    n_init: int = 1,                # Number of initializations
    max_iter: int = 100,            # Maximum number of iterations
    random_state: int = 0,
):
    """
    Cluster the entire unlabeled dataset using a Gaussian Mixture Model (GMM)
    and add a new column "label" with the cluster assignments.

    Parameters:
        df (pl.DataFrame): Input unlabeled Polars DataFrame.
        varlist (list[str]): List of feature names to use for clustering.
        n_components (int): Number of mixture components (clusters).
        covariance_type (str): Covariance type for the GMM.
        reg_covar (float): Regularization parameter added to covariance matrices.
        n_init (int): Number of initializations.
        max_iter (int): Maximum number of iterations for convergence.
        random_state (int): Random state for reproducibility.

    Returns:
        pd.DataFrame: A Pandas DataFrame with a new column "label" containing cluster labels.
    """
    # Convert the Polars DataFrame to a Pandas DataFrame.
    df_pd = df.to_pandas()

    # Extract features.
    X = df_pd[varlist]

    # Standardize features.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Configure and apply the Gaussian Mixture Model.
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)

    # Add the new "label" column to the DataFrame.
    df_pd["label"] = labels
    print("GMM clustering completed. 'label' column added.")

    return df_pd


if __name__ == "__main__":
    # Expect two command-line arguments:
    #   1. The input data file path.
    #   2. The output file path where the new DataFrame should be saved as a .ipc file.
    data_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    # Load the data using your existing load_window function.
    df = load_window(data_file)

    # Define the feature list to use for clustering.
    feature_list = [
        "amplitude",
        "frequency",
        "pulse_width",
    ]

    # Run the clustering algorithm on the entire dataset.
    clustered_df = cluster_unlabeled_data(
        df,
        feature_list,
        n_components=150,
        covariance_type="full",
        reg_covar=1e-6,
        n_init=2,
        max_iter=50,
        random_state=0,
    )

    table = pa.Table.from_pandas(clustered_df)

    # Save the Arrow Table to an IPC file using RecordBatchFileWriter.
    with pa.OSFile(str(output_file), "wb") as sink:
        with ipc.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)
    print(f"Clustered data saved as IPC file to {output_file}")
