from pathlib import Path
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from parse_data import load_window
from geometry import (
    lla_to_enu,
    sensor_direction_vector_enu,
    compute_intersection,
    ecef_to_lla,
    enu_to_ecef
    
)

def process_consecutive_points(df, ref_lat, ref_lon, ref_alt, delta=1):
    """
    Processes consecutive sensor readings (rows) in the DataFrame.
    
    For each pair (separated by `delta`), it converts the sensor positions 
    from LLA to ENU (using the given reference point) and computes the sensor 
    direction vectors directly in ENU. It then computes the intersection 
    (in ENU) of the rays defined by the sensor position and its direction.
    
    Returns a new DataFrame with the estimated intersection points and the
    corresponding arrival_time (from the current row).
    """
    df_sorted = df.sort_index()
    
    intersections = []
    times = []
    
    for i in range(1, len(df_sorted)):
        prev = df_sorted.iloc[i - delta]
        curr = df_sorted.iloc[i]
        
        # Convert sensor positions from LLA (in degrees) to ENU (in meters) 
        # using the emitter (or chosen) reference coordinates.
        P1_enu = np.array(lla_to_enu(
            prev['sensor_lat'], prev['sensor_lon'], prev['sensor_alt'],
            ref_lat, ref_lon, ref_alt
        ))
        P2_enu = np.array(lla_to_enu(
            curr['sensor_lat'], curr['sensor_lon'], curr['sensor_alt'],
            ref_lat, ref_lon, ref_alt
        ))
        
        # Compute the sensor's direction vector in ENU coordinates.
        d1_enu = sensor_direction_vector_enu(
            prev['azimuth'], prev['elevation'],
            prev['sensor_yaw'], prev['sensor_pitch'], prev['sensor_roll']
        )
        d2_enu = sensor_direction_vector_enu(
            curr['azimuth'], curr['elevation'],
            curr['sensor_yaw'], curr['sensor_pitch'], curr['sensor_roll']
        )
        
        # Compute the intersection point (in ENU) of the two sensor rays.
        Q = compute_intersection(P1_enu, d1_enu, P2_enu, d2_enu)
        
        times.append(curr.name)
        intersections.append(Q)
    
    intersections = np.array(intersections)  # shape: (n_pairs, 3)
    result_df = pd.DataFrame(intersections, index=times, columns=['E', 'N', 'U'])
    result_df.index.name = 'arrival_time'
    return result_df

def compute_emitter_enu(row, ref_lat, ref_lon, ref_alt):
    """
    Computes the emitter location in ENU coordinates.
    
    The emitter's LLA (lat, lon in degrees, alt in meters) is converted into ENU,
    using the provided reference point.
    """
    return np.array(lla_to_enu(
        row['emitter_lat'], row['emitter_lon'], row['emitter_alt'],
        ref_lat, ref_lon, ref_alt
    ))

def perform_checks_and_comparisons(df, delta=1000):
    """
    Process the input data file to compute sensor intersections and emitter coordinates,
    and then compare the estimated emitter position with the original.

    Parameters:
        file_path (str or Path): Path to the input file.
        emitter (str): The emitter identifier to filter the data.
        delta (int): The delta parameter used in processing consecutive points.
    """

    # Use the emitter coordinates from the first row as the ENU reference.
    first_row = df.iloc[0]
    ref_lat = first_row['emitter_lat']
    ref_lon = first_row['emitter_lon']
    ref_alt = first_row['emitter_alt']

    # Compute intersections from consecutive sensor readings (in ENU).
    intersection_df = process_consecutive_points(df, ref_lat, ref_lon, ref_alt, delta=delta)

    # Compute emitter ENU coordinates for each sensor reading.
    emitter_enu = df.apply(lambda row: compute_emitter_enu(row, ref_lat, ref_lon, ref_alt), axis=1)
    emitter_enu = pd.DataFrame(emitter_enu.tolist(), index=df.index, columns=['E_emitter', 'N_emitter', 'U_emitter'])

    # Final Calculation: Convert Estimated Emitter Position to LLA.
    estimated_lla = []
    for idx, row in intersection_df.iterrows():
        e, n, u = row['E'], row['N'], row['U']
        x, y, z = enu_to_ecef(e, n, u, ref_lat, ref_lon, ref_alt)
        lat_est, lon_est, alt_est = ecef_to_lla(x, y, z)
        estimated_lla.append([lat_est, lon_est, alt_est])
    estimated_lla = pd.DataFrame(estimated_lla, index=intersection_df.index, columns=['lat_est', 'lon_est', 'alt_est'])

    # Extract the original emitter LLA (assumed constant across readings).
    original_lla = first_row[['emitter_lat', 'emitter_lon', 'emitter_alt']].to_frame().T
    original_lla.index = ['Original']
    
    # Optionally, return the computed DataFrames for further processing.
    return {
        "intersection_df": intersection_df,
        "emitter_enu": emitter_enu,
        "estimated_lla": estimated_lla,
        "original_lla": original_lla
    }



if __name__ == "__main__":
    file = Path(sys.argv[1])
    df = load_window(file)

    emitter = "Emitter1"

    # Filter the DataFrame for the specified emitter and select the required columns.
    df = df.filter(df["emitter"] == emitter).drop("emitter")
    columns_to_keep = [
        'arrival_time', 'azimuth', 'elevation', 'sensor_lat', 'sensor_lon', 
        'sensor_alt', 'sensor_yaw', 'sensor_pitch', 'sensor_roll', 
        'emitter_lat', 'emitter_lon', 'emitter_alt'
    ]
    df = df.select(columns_to_keep)

    df = df.to_pandas()
    df.set_index("arrival_time", inplace=True)

    # Perform the computations and comparisons.
    results = perform_checks_and_comparisons(df, delta=1000)
    print(results)
    

