import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
import polars as pl

from parse_data import load_window
from geometry import sensor_direction_vector_enu, lla_to_enu


def compute_global_vectors_enu(df):
    """
    For each row in the DataFrame, compute and add the sensor direction vector 
    components in ENU coordinates.
    
    The resulting vector is stored as [East, North, Up] in new DataFrame columns.
    """
    evx, evy, evz = [], [], []
    for _, row in df.iterrows():
        d_enu = sensor_direction_vector_enu(
            row["azimuth"],
            row["elevation"],
            row["sensor_yaw"],
            row["sensor_pitch"],
            row["sensor_roll"],
        )
        evx.append(d_enu[0])
        evy.append(d_enu[1])
        evz.append(d_enu[2])
    df["global_vector_east"] = evx
    df["global_vector_north"] = evy
    df["global_vector_up"] = evz
    return df

def plot_trajectories(df, ray_length=1000, ray_interval=1000):
    """
    Plot sensor and emitter trajectories (converted to local ENU coordinates)
    along with sensor rays.
    
    Since all our computations are now in ENU (with axes: [East, North, Up]),
    we directly use these coordinates.
    """
    # Use the emitter's first position as the ENU reference.
    lat_ref = df["emitter_lat"].iloc[0]
    lon_ref = df["emitter_lon"].iloc[0]
    alt_ref = df["emitter_alt"].iloc[0]

    # Compute sensor and emitter positions in ENU.
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

    # Build sensor rays using the already computed ENU sensor direction vectors.
    ray_x, ray_y, ray_z = [], [], []
    for i in range(0, len(sensor_enu), ray_interval):
        start = sensor_enu[i]
        # Use the ENU direction vector from the DataFrame (already in ENU).
        d_enu = np.array([
            df["global_vector_east"].iloc[i],
            df["global_vector_north"].iloc[i],
            df["global_vector_up"].iloc[i],
        ])
        ray_endpoint = start + ray_length * d_enu
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

    df = compute_global_vectors_enu(df)
    plot_trajectories(df, ray_length=50000, ray_interval=1000)
