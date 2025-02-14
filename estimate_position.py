from pathlib import Path
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from parse_data import load_window

from geometry import latlonalt_to_ecef, compute_global_direction, compute_intersection


def process_consecutive_points(df, delta=1):
    """
    Processes all consecutive pairs in the DataFrame. For each pair of consecutive
    points, computes the intersection (in ECEF coordinates) of the rays defined by
    the sensor's position and direction. Returns a new DataFrame with the estimated
    intersection points and the corresponding arrival_time (from the current point).
    """
    df_sorted = df.sort_index()
    
    intersections = []
    times = []
    
    for i in range(1, len(df_sorted)):
        prev = df_sorted.iloc[i-delta]
        curr = df_sorted.iloc[i]
        
        # Convert sensor positions to ECEF coordinates:
        P1 = latlonalt_to_ecef(prev['sensor_lat'], prev['sensor_lon'], prev['sensor_alt'])
        P2 = latlonalt_to_ecef(curr['sensor_lat'], curr['sensor_lon'], curr['sensor_alt'])
        
        # Convert angles from degrees to radians (assume angles in the df are in degrees)
        prev_angles = prev[['azimuth', 'elevation', 'sensor_yaw', 'sensor_pitch', 'sensor_roll']].astype(float) * np.pi / 180.0
        curr_angles = curr[['azimuth', 'elevation', 'sensor_yaw', 'sensor_pitch', 'sensor_roll']].astype(float) * np.pi / 180.0
        
        # Build dictionaries with the angles in radians for the previous and current points
        prev_row = {
            'azimuth': prev_angles['azimuth'],
            'elevation': prev_angles['elevation'],
            'sensor_yaw': prev_angles['sensor_yaw'],
            'sensor_pitch': prev_angles['sensor_pitch'],
            'sensor_roll': prev_angles['sensor_roll']
        }
        curr_row = {
            'azimuth': curr_angles['azimuth'],
            'elevation': curr_angles['elevation'],
            'sensor_yaw': curr_angles['sensor_yaw'],
            'sensor_pitch': curr_angles['sensor_pitch'],
            'sensor_roll': curr_angles['sensor_roll']
        }
        
        # Compute global direction vectors
        d1 = compute_global_direction(prev_row)
        d2 = compute_global_direction(curr_row)
        
        # Compute the intersection point of the two rays
        Q = compute_intersection(P1, d1, P2, d2)
        
        # Store the intersection and the corresponding arrival_time (from the current row)
        times.append(curr.name)  # using index as time (arrival_time)
        intersections.append(Q)
    
    # Create a DataFrame from the results
    intersections = np.array(intersections)  # shape: (n_pairs, 3)
    result_df = pd.DataFrame(intersections, index=times, columns=['X', 'Y', 'Z'])
    result_df.index.name = 'arrival_time'
    return result_df


def compute_emitter_ecef(row):
    return latlonalt_to_ecef(row['emitter_lat'], row['emitter_lon'], row['emitter_alt'])


if __name__ == "__main__":
    file = Path(sys.argv[1])

    df = load_window(file)

    emitter = "Emitter1"

    df = df.filter(df["emitter"] == emitter).drop("emitter")
    columns_to_keep = ['arrival_time','azimuth', 'elevation', 'sensor_lat', 'sensor_lon', 
                       'sensor_alt', 'sensor_yaw', 'sensor_pitch', 'sensor_roll', 'emitter_lat', 'emitter_lon', 'emitter_alt']
    df = df.select(columns_to_keep)

    df = df.to_pandas()
    df.set_index("arrival_time", inplace=True)

    print(df)

    intersection_series = process_consecutive_points(df, delta=1)
    print(intersection_series)

    emitter_ecef = df.apply(lambda row: compute_emitter_ecef(row), axis=1)
    emitter_ecef = pd.DataFrame(emitter_ecef.tolist(), index=df.index, columns=['X_emitter', 'Y_emitter', 'Z_emitter'])

    print(emitter_ecef)


