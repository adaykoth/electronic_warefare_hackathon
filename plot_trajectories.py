#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go

from geometry import lla_to_enu, global_direction_vector

from parse_data import load_window


def plot_trajectories(df, ray_length=1000, ray_interval=1000):
    """
    Plot the sensor and emitter trajectories (converted into local ENU coordinates)
    as well as sensor rays (from sensor position in the direction of the global vector).

    The global vectors (computed earlier) are in a coordinate system where:
       [global_vector_x, global_vector_y, global_vector_z] = [North, East, Up]
    For plotting in an ENU frame (x=East, y=North, z=Up) we swap the first two components.
    
    Parameters:
      - ray_length: length of each ray
      - ray_interval: only draw a ray every `ray_interval` points.
    """
    # Use the first sensor coordinate as the ENU reference point.
    lat_ref = df["emitter_lat"].iloc[0]
    lon_ref = df["emitter_lon"].iloc[0]
    alt_ref = df["emitter_alt"].iloc[0]
    
    sensor_east, sensor_north, sensor_up = [], [], []
    emitter_east, emitter_north, emitter_up = [], [], []
    
    # Compute ENU coordinates for sensor and emitter positions.
    for idx, row in df.iterrows():
        se, sn, su = lla_to_enu(row["sensor_lat"], row["sensor_lon"], row["sensor_alt"],
                                  lat_ref, lon_ref, alt_ref)
        sensor_east.append(se)
        sensor_north.append(sn)
        sensor_up.append(su)
        
        ee, en, eu = lla_to_enu(row["emitter_lat"], row["emitter_lon"], row["emitter_alt"],
                                  lat_ref, lon_ref, alt_ref)
        emitter_east.append(ee)
        emitter_north.append(en)
        emitter_up.append(eu)
    
    # Prepare the sensor trajectory trace.
    trace_sensor = go.Scatter3d(
        x=sensor_east,
        y=sensor_north,
        z=sensor_up,
        mode='lines+markers',
        name='Sensor Trajectory',
        line=dict(color='blue'),
        marker=dict(size=3, color='blue')
    )
    
    # Prepare the emitter trajectory trace.
    trace_emitter = go.Scatter3d(
        x=emitter_east,
        y=emitter_north,
        z=emitter_up,
        mode='lines+markers',
        name='Emitter Trajectory',
        line=dict(color='red'),
        marker=dict(size=3, color='red')
    )
    
    # Build the sensor rays (only every `ray_interval` points).
    gvx = df["global_vector_x"].values  # North component
    gvy = df["global_vector_y"].values  # East component
    gvz = df["global_vector_z"].values  # Up component
    
    ray_east, ray_north, ray_up = [], [], []
    for i in range(len(sensor_east)):
        if i % ray_interval == 0:
            ray_east.extend([sensor_east[i], sensor_east[i] + ray_length * gvy[i], None])
            ray_north.extend([sensor_north[i], sensor_north[i] + ray_length * gvx[i], None])
            ray_up.extend([sensor_up[i], sensor_up[i] + ray_length * gvz[i], None])
    
    trace_rays = go.Scatter3d(
        x=ray_east,
        y=ray_north,
        z=ray_up,
        mode='lines',
        name='Sensor Rays',
        line=dict(color='green', width=2)
    )
    
    # Assemble the figure.
    fig = go.Figure(data=[trace_sensor, trace_emitter, trace_rays])
    fig.update_layout(
        scene=dict(
            xaxis_title="East (m)",
            yaxis_title="North (m)",
            zaxis_title="Up (m)"
        ),
        title="Sensor & Emitter Trajectories with Direction Rays"
    )
    
    fig.show()


    
if __name__ == "__main__":
    file_path = Path(sys.argv[1])
    df_pl = load_window(file_path)
    
    emitter = "Emitter7"
    # Use Polars filtering (not pandas boolean indexing)
    df_pl = df_pl.filter(pl.col("emitter") == emitter).drop("emitter")
    
    columns_to_keep = [
        'arrival_time', 'azimuth', 'elevation',
        'sensor_lat', 'sensor_lon', 'sensor_alt',
        'sensor_yaw', 'sensor_pitch', 'sensor_roll',
        'emitter_lat', 'emitter_lon', 'emitter_alt'
    ]
    df_pl = df_pl.select(columns_to_keep)
    
    # Convert to pandas DataFrame and set the index.
    df = df_pl.to_pandas()
    df.set_index("arrival_time", inplace=True)

    print(df)
    
    # Compute global direction vectors for each row.
    global_vec_x = []
    global_vec_y = []
    global_vec_z = []
    
    for idx, row in df.iterrows():
        # Angles are assumed to be in radians.
        azimuth      = row["azimuth"]
        elevation    = row["elevation"]
        sensor_yaw   = row["sensor_yaw"]
        sensor_pitch = row["sensor_pitch"]
        sensor_roll  = row["sensor_roll"]
        
        d_global = global_direction_vector(azimuth, elevation, sensor_yaw, sensor_pitch, sensor_roll)
        # d_global is in [North, East, Up] convention.
        global_vec_x.append(d_global[0])
        global_vec_y.append(d_global[1])
        global_vec_z.append(d_global[2])
    
    df["global_vector_x"] = global_vec_x
    df["global_vector_y"] = global_vec_y
    df["global_vector_z"] = global_vec_z

    # Optionally, print the DataFrame to verify.
    print(df.head())
    
    # Plot the trajectories and the sensor rays.
    # You can change ray_length and ray_interval as needed.
    plot_trajectories(df, ray_length=50000, ray_interval=1000)
