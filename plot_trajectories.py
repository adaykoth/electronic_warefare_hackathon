#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go

from geometry import lla_to_enu, global_direction_vector
from parse_data import load_window

def neu_to_enu(neu_vector):
    """
    Convert a vector from NEU ([North, East, Up]) to ENU ([East, North, Up]) coordinates.
    """
    # Swap the first two components: [N, E, U] -> [E, N, U]
    return np.array([neu_vector[1], neu_vector[0], neu_vector[2]])

def compute_global_vectors(df):
    """
    Compute and add the global direction vector components (in [North, East, Up])
    for each row in the DataFrame.
    """
    gvx, gvy, gvz = [], [], []
    for _, row in df.iterrows():
        d_global = global_direction_vector(
            row["azimuth"],
            row["elevation"],
            row["sensor_yaw"],
            row["sensor_pitch"],
            row["sensor_roll"],
        )
        gvx.append(d_global[0])
        gvy.append(d_global[1])
        gvz.append(d_global[2])
    df["global_vector_x"] = gvx  # North component
    df["global_vector_y"] = gvy  # East component
    df["global_vector_z"] = gvz  # Up component
    return df

def plot_trajectories(df, ray_length=1000, ray_interval=1000):
    """
    Plot sensor and emitter trajectories (converted to local ENU coordinates)
    along with sensor rays.

    The global direction vector is computed in the local [North, East, Up] frame.
    To plot in an ENU frame ([East, North, Up]), we convert it using neu_to_enu.
    
    Parameters:
      ray_length: length of each ray.
      ray_interval: only draw a ray every `ray_interval` points.
    """
    # Use the emitter's first position as the ENU reference.
    lat_ref = df["emitter_lat"].iloc[0]
    lon_ref = df["emitter_lon"].iloc[0]
    alt_ref = df["emitter_alt"].iloc[0]

    # Compute sensor and emitter positions in ENU using list comprehensions.
    sensor_enu = np.array([
        lla_to_enu(row["sensor_lat"], row["sensor_lon"], row["sensor_alt"],
                   lat_ref, lon_ref, alt_ref)
        for _, row in df.iterrows()
    ])
    emitter_enu = np.array([
        lla_to_enu(row["emitter_lat"], row["emitter_lon"], row["emitter_alt"],
                   lat_ref, lon_ref, alt_ref)
        for _, row in df.iterrows()
    ])

    # Build traces for sensor and emitter trajectories.
    trace_sensor = go.Scatter3d(
        x=sensor_enu[:, 0],
        y=sensor_enu[:, 1],
        z=sensor_enu[:, 2],
        mode="lines+markers",
        name="Sensor Trajectory",
        line=dict(color="blue"),
        marker=dict(size=3, color="blue"),
    )
    trace_emitter = go.Scatter3d(
        x=emitter_enu[:, 0],
        y=emitter_enu[:, 1],
        z=emitter_enu[:, 2],
        mode="lines+markers",
        name="Emitter Trajectory",
        line=dict(color="red"),
        marker=dict(size=3, color="red"),
    )

    # Build sensor rays.
    # The global direction vector is in [North, East, Up]. We convert it to ENU.
    gvx = df["global_vector_x"].values  # North component
    gvy = df["global_vector_y"].values  # East component
    gvz = df["global_vector_z"].values  # Up component

    ray_x, ray_y, ray_z = [], [], []
    for i in range(0, len(sensor_enu), ray_interval):
        start = sensor_enu[i]
        # Convert the NEU vector to ENU.
        neu_vector = np.array([gvx[i], gvy[i], gvz[i]])
        enu_vector = neu_to_enu(neu_vector)
        # Compute ray endpoint in ENU.
        ray_endpoint = start + ray_length * enu_vector
        ray_x.extend([start[0], ray_endpoint[0], None])
        ray_y.extend([start[1], ray_endpoint[1], None])
        ray_z.extend([start[2], ray_endpoint[2], None])

    trace_rays = go.Scatter3d(
        x=ray_x,
        y=ray_y,
        z=ray_z,
        mode="lines",
        name="Sensor Rays",
        line=dict(color="green", width=2),
    )

    # Assemble and show the figure.
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
    emitter = "Emitter2"
    df_pl = df_pl.filter(pl.col("emitter") == emitter).drop("emitter")
    columns = [
        'arrival_time', 'azimuth', 'elevation',
        'sensor_lat', 'sensor_lon', 'sensor_alt',
        'sensor_yaw', 'sensor_pitch', 'sensor_roll',
        'emitter_lat', 'emitter_lon', 'emitter_alt'
    ]
    df_pl = df_pl.select(columns)
    df = df_pl.to_pandas()
    df.set_index("arrival_time", inplace=True)

    # Compute the global direction vectors (in NEU).
    df = compute_global_vectors(df)
    print(df.head())
    
    # Adjust ray_length and ray_interval as needed.
    plot_trajectories(df, ray_length=50000, ray_interval=1000)
