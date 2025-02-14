import sys
import os
from pathlib import Path

import numpy as np
import polars as pl

import matplotlib.pyplot as plt

def load_window(file: Path) -> pl.DataFrame:
    df = pl.read_ipc(file)
    return df

def plot_values(df: pl.DataFrame, emitter: int, output_folder: str):
    filtered_df = df.filter(df["emitter"] == emitter)

    for col in set(filtered_df.columns) - {"emitter", "arrival_time"}:

        time_list = filtered_df["arrival_time"].to_list()
        y_list = filtered_df[col].to_list()

        plt.figure()
        plt.plot(time_list, y_list, marker="o", linestyle="-")
        plt.xlabel("Time")
        plt.ylabel(col)
        plt.title(f"Emitter {emitter}: {col} vs Time")
        plt.tight_layout()
        
        os.makedirs(output_folder, exist_ok=True)   
        file_name = os.path.join(output_folder, f"emitter_{emitter}_{col}.png")
        plt.savefig(file_name)
        plt.close()


if __name__ == "__main__":

    file = Path(sys.argv[1])
    df = load_window(file)

    

    
    
        


