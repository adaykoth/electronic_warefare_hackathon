import sys
from pathlib import Path

import numpy as np
import pandas as pd

import plotly.graph_objects as go

from parse_data import load_window


def plot_euclidean_trajectory(pdf):

    # Earth's radius in meters
    R = 6371000

    # Convert latitudes and longitudes from degrees to radians.
    pdf['sensor_lat_rad'] = np.radians(pdf['sensor_lat'])
    pdf['sensor_lon_rad'] = np.radians(pdf['sensor_lon'])
    pdf['emitter_lat_rad'] = np.radians(pdf['emitter_lat'])
    pdf['emitter_lon_rad'] = np.radians(pdf['emitter_lon'])

    # Compute Cartesian coordinates for sensor positions.
    pdf['sensor_x'] = (R + pdf['sensor_alt']) * np.cos(pdf['sensor_lat_rad']) * np.cos(pdf['sensor_lon_rad'])
    pdf['sensor_y'] = (R + pdf['sensor_alt']) * np.cos(pdf['sensor_lat_rad']) * np.sin(pdf['sensor_lon_rad'])
    pdf['sensor_z'] = (R + pdf['sensor_alt']) * np.sin(pdf['sensor_lat_rad'])

    # Compute Cartesian coordinates for emitter positions.
    pdf['emitter_x'] = (R + pdf['emitter_alt']) * np.cos(pdf['emitter_lat_rad']) * np.cos(pdf['emitter_lon_rad'])
    pdf['emitter_y'] = (R + pdf['emitter_alt']) * np.cos(pdf['emitter_lat_rad']) * np.sin(pdf['emitter_lon_rad'])
    pdf['emitter_z'] = (R + pdf['emitter_alt']) * np.sin(pdf['emitter_lat_rad'])

    # Create a 3D scatter trace for sensor positions.
    sensor_trace = go.Scatter3d(
        x=pdf['sensor_x'],
        y=pdf['sensor_y'],
        z=pdf['sensor_z'],
        mode='markers',
        marker=dict(
            size=5,
            color=pdf['arrival_time'],  # Color by arrival_time.
            colorscale='Viridis',
            colorbar=dict(title='Arrival Time'),
            opacity=0.8
        ),
        name='Sensor'
    )

    # Create a 3D scatter trace for emitter positions.
    emitter_trace = go.Scatter3d(
        x=pdf['emitter_x'],
        y=pdf['emitter_y'],
        z=pdf['emitter_z'],
        mode='markers',
        marker=dict(
            size=5,
            color=pdf['arrival_time'],  # Color by arrival_time.
            colorscale='Plasma',
            opacity=0.8
        ),
        name='Emitter'
    )

    # Combine the traces into a figure.
    fig = go.Figure(data=[sensor_trace, emitter_trace])

    # Update layout with axis labels and a title.
    fig.update_layout(
        title="3D Cartesian Coordinates for Sensor and Emitter Positions Over Arrival Time",
        scene=dict(
            xaxis_title="X (meters)",
            yaxis_title="Y (meters)",
            zaxis_title="Z (meters)"
        )
    )

    fig.show()
    

if __name__ == "__main__":
    file = Path(sys.argv[1])

    df = load_window(file)

    emitter = "Emitter1"

    df = df.filter(df["emitter"] == emitter).drop("emitter")
    columns_to_keep = ['arrival_time', 'sensor_lat', 'sensor_lon', 'sensor_alt', 'emitter_lat', 'emitter_lon', 'emitter_alt']
    df = df.select(columns_to_keep)
    pdf = df.to_pandas()

    plot_euclidean_trajectory(pdf)