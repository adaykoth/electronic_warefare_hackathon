import sys
import os
from pathlib import Path
from scipy.fft import fft

import numpy as np
import polars as pl

import matplotlib.pyplot as plt

def load_window(file: Path) -> pl.DataFrame:
    df = pl.read_ipc(file)
    return df

def plot_values(df: pl.DataFrame, emitter: str, output_folder: str):
    filtered_df = df.filter(df["emitter"] == emitter)

    time_list = filtered_df["arrival_time"].to_list()
    time_list = time_list / ((10^9)*np.ones(len(time_list))) # ns -> s
    time_list = time_list[2:-2] # Remove extreme DC components at either end of the FFT result
    y_list = filtered_df["amplitude"].to_list()
    y_list = fft(y_list)
    y_list = y_list[2:-2] # Remove extreme DC components at either end of the FFT result

    plt.figure()
    plt.plot(time_list, y_list, marker="o", linestyle="-")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.title(f"Emitter {emitter}: amplitude vs frequency")
    plt.tight_layout()

    os.makedirs(output_folder, exist_ok=True)   
    file_name = os.path.join(output_folder, f"fft_{emitter}_amplitude.png")
    plt.savefig(file_name)
    plt.show()
    plt.close()
    print(time_list)


if __name__ == "__main__":

    file = Path(sys.argv[1])
    df = load_window(file)

    print(df)

    plot_values(df, "Emitter11", "output")