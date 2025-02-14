from pathlib import Path
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from parse_data import load_window
from geometry import (
    lla_to_ecef,
    global_direction_vector,
    compute_intersection
)

### TODO: CHECK IF THIS IS WORKING AT ALL

def enu_to_ecef_vector(enu_vector, sensor_lat, sensor_lon):
    """
    Convert a direction vector from local North-East-Up (NEU) coordinates to ECEF coordinates.
    
    Parameters:
      enu_vector : array_like
          The 3D unit vector in NEU coordinates.
      sensor_lat : float
          Sensor latitude in degrees.
      sensor_lon : float
          Sensor longitude in degrees.
    
    Returns:
      A 3D vector (numpy array) in ECEF coordinates.
    """
    # Convert sensor latitude and longitude to radians.
    lat = np.radians(sensor_lat)
    lon = np.radians(sensor_lon)
    
    # Rotation matrix from local ENU to ECEF.
    R = np.array([
        [-np.sin(lon), -np.sin(lat)*np.cos(lon),  np.cos(lat)*np.cos(lon)],
        [ np.cos(lon), -np.sin(lat)*np.sin(lon),  np.cos(lat)*np.sin(lon)],
        [          0,              np.cos(lat),              np.sin(lat)]
    ])
    return R @ enu_vector

def process_consecutive_points(df, delta=1):
    """
    Processes consecutive pairs in the DataFrame. For each pair, it computes the intersection
    (in ECEF coordinates) of the rays defined by the sensor's position and its direction.
    
    The sensor's position is converted to ECEF and the sensor's direction vector (which is
    originally in a local [North, East, Up] frame) is converted into ECEF using the sensor's
    latitude and longitude.
    
    Returns a new DataFrame with the estimated intersection points and the corresponding
    arrival_time (from the current row).
    """
    df_sorted = df.sort_index()
    
    intersections = []
    times = []
    
    for i in range(1, len(df_sorted)):
        prev = df_sorted.iloc[i - delta]
        curr = df_sorted.iloc[i]
        
        # Convert sensor positions (lat, lon in degrees) to ECEF coordinates.
        # lla_to_ecef expects lat and lon in radians.
        P1 = np.array(lla_to_ecef(
            np.radians(prev['sensor_lat']),
            np.radians(prev['sensor_lon']),
            prev['sensor_alt']
        ))
        P2 = np.array(lla_to_ecef(
            np.radians(curr['sensor_lat']),
            np.radians(curr['sensor_lon']),
            curr['sensor_alt']
        ))
        
        # Compute the local direction vectors (in [North, East, Up]) for each sensor reading.
        d1_local = global_direction_vector(
            prev['azimuth'],
            prev['elevation'],
            prev['sensor_yaw'],
            prev['sensor_pitch'],
            prev['sensor_roll']
        )
        d2_local = global_direction_vector(
            curr['azimuth'],
            curr['elevation'],
            curr['sensor_yaw'],
            curr['sensor_pitch'],
            curr['sensor_roll']
        )
        
        # Convert the local NEU direction vectors to ECEF.
        d1_ecef = enu_to_ecef_vector(d1_local, prev['sensor_lat'], prev['sensor_lon'])
        d2_ecef = enu_to_ecef_vector(d2_local, curr['sensor_lat'], curr['sensor_lon'])
        
        # Compute the intersection point of the two rays (both in ECEF).
        Q = compute_intersection(P1, d1_ecef, P2, d2_ecef)
        
        # Use the current row's index as arrival_time.
        times.append(curr.name)
        intersections.append(Q)
    
    intersections = np.array(intersections)  # shape: (n_pairs, 3)
    result_df = pd.DataFrame(intersections, index=times, columns=['X', 'Y', 'Z'])
    result_df.index.name = 'arrival_time'
    return result_df


def compute_emitter_ecef(row):
    """
    Computes the emitter location in ECEF coordinates.
    Converts emitter latitude and longitude from degrees to radians.
    """
    return np.array(lla_to_ecef(
        np.radians(row['emitter_lat']),
        np.radians(row['emitter_lon']),
        row['emitter_alt']
    ))


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

    print("Input DataFrame:")
    print(df)

    # Compute intersections from consecutive sensor readings (in ECEF).
    intersection_series = process_consecutive_points(df, delta=1)
    print("\nIntersections (ECEF):")
    print(intersection_series)

    # Compute emitter ECEF coordinates for each sensor reading.
    emitter_ecef = df.apply(lambda row: compute_emitter_ecef(row), axis=1)
    emitter_ecef = pd.DataFrame(emitter_ecef.tolist(), index=df.index, 
                                columns=['X_emitter', 'Y_emitter', 'Z_emitter'])
    print("\nEmitter ECEF coordinates:")
    print(emitter_ecef)

