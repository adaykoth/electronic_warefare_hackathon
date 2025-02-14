import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
import polars as pl

from parse_data import load_window

# ---------------- Geometry Functions ----------------

def sensor_direction_vector(azimuth, elevation):
    """
    Compute the unit direction vector in the sensor (plane) coordinate frame.
    
    Assumptions (angles in radians):
      - Sensor coordinate system:
            x = cos(elevation) * cos(azimuth)   (forward)
            y = cos(elevation) * sin(azimuth)   (to the right)
            z = sin(elevation)                  (up)
    """
    x = np.cos(elevation) * np.cos(azimuth)
    y = np.cos(elevation) * np.sin(azimuth)
    z = np.sin(elevation)
    return np.array([x, y, z])


def rotation_matrix(yaw, pitch, roll):
    """
    Build a rotation matrix to transform a vector from the sensor's
    coordinate frame to the global frame.
    
    The sensor's orientation is given by Euler angles (in radians):
      - yaw   : rotation about the global z-axis (0 means north),
      - pitch : rotation about the sensor's y-axis (positive for nose-up),
      - roll  : rotation about the sensor's x-axis.
    
    We use the intrinsic rotation convention:
         R = R_z(yaw) @ R_y(pitch) @ R_x(roll)
    """
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    Ry = np.array([
        [np.cos(pitch), 0, -np.sin(pitch)],
        [0, 1, 0],
        [np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    
    return Rz @ Ry @ Rx


def global_direction_vector(azimuth, elevation, sensor_yaw, sensor_pitch, sensor_roll):
    """
    Compute the sensor direction vector in global NEU coordinates.
    
    The result is in NEU format, i.e. [North, East, Up].
    """
    # Convert degrees to radians
    az = np.radians(azimuth)
    el = np.radians(elevation)
    yaw = np.radians(sensor_yaw)
    pitch = np.radians(sensor_pitch)
    roll = np.radians(sensor_roll)
    
    d_sensor = sensor_direction_vector(az, el)
    R = rotation_matrix(yaw, pitch, roll)
    return R @ d_sensor


def sensor_direction_vector_enu(azimuth, elevation, sensor_yaw, sensor_pitch, sensor_roll):
    """
    Compute the sensor direction vector directly in ENU coordinates.
    
    (ENU = [East, North, Up]).  
    We first compute the global vector in NEU (i.e. [North, East, Up])
    and then swap the first two components.
    """
    d_neu = global_direction_vector(azimuth, elevation, sensor_yaw, sensor_pitch, sensor_roll)
    # Convert NEU -> ENU by swapping the first two components.
    return np.array([d_neu[1], d_neu[0], d_neu[2]])


# --- ECEF and ENU Conversion Functions ---

def lla_to_ecef(lat_rad, lon_rad, alt):
    """
    Convert latitude, longitude (in radians) and altitude (in meters)
    into ECEF (Earth-Centered, Earth-Fixed) coordinates.
    """
    # WGS84 parameters
    a = 6378137.0            # semi-major axis in meters
    f = 1 / 298.257223563    # flattening
    e2 = f * (2 - f)         # eccentricity squared
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = ((1 - e2) * N + alt) * np.sin(lat_rad)
    return x, y, z

def ecef_to_enu(x, y, z, lat_ref_rad, lon_ref_rad, alt_ref):
    """
    Convert ECEF coordinates to local ENU coordinates with respect to the reference point.
    lat_ref_rad and lon_ref_rad are in radians.
    """
    x0, y0, z0 = lla_to_ecef(lat_ref_rad, lon_ref_rad, alt_ref)
    dx = x - x0
    dy = y - y0
    dz = z - z0

    sin_lat = np.sin(lat_ref_rad)
    cos_lat = np.cos(lat_ref_rad)
    sin_lon = np.sin(lon_ref_rad)
    cos_lon = np.cos(lon_ref_rad)

    # East component
    east = -sin_lon * dx + cos_lon * dy
    # North component
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    # Up component
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return east, north, up

def lla_to_enu(lat, lon, alt, lat_ref, lon_ref, alt_ref):
    """
    Convert latitude, longitude (in degrees) and altitude (in meters)
    into local ENU coordinates.
    The reference point is given by lat_ref, lon_ref (in degrees) and alt_ref.
    """
    # Convert degrees to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat_ref_rad = np.radians(lat_ref)
    lon_ref_rad = np.radians(lon_ref)
    
    x, y, z = lla_to_ecef(lat_rad, lon_rad, alt)
    east, north, up = ecef_to_enu(x, y, z, lat_ref_rad, lon_ref_rad, alt_ref)
    return east, north, up


def compute_intersection(P1, d1, P2, d2):
    """
    Compute the closest points on the two rays:
       L1 = P1 + t1*d1, and L2 = P2 + t2*d2.
    Returns the midpoint between these two points.
    """
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    
    w0 = P1 - P2
    
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, w0)
    e = np.dot(d2, w0)
    
    denominator = a * c - b * b
    if np.abs(denominator) < 1e-6:
        t1 = 0.0
        t2 = d / b if np.abs(b) > 1e-6 else 0.0
    else:
        t1 = (b * e - c * d) / denominator
        t2 = (a * e - b * d) / denominator
    
    Q1 = P1 + t1 * d1
    Q2 = P2 + t2 * d2
    Q = (Q1 + Q2) / 2
    return Q


# ------------------ Data & Plotting Functions ------------------

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


# ------------------ Main Script ------------------

if __name__ == "__main__":
    # Load the data using your parser (assumes 'load_window' returns a Polars DataFrame)
    file_path = Path(sys.argv[1])
    df_pl = load_window(file_path)
    emitter = "Emitter1"
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
    print(df.head())

    # Compute sensor direction vectors in ENU.
    df = compute_global_vectors_enu(df)
    
    # Adjust ray_length and ray_interval as needed.
    plot_trajectories(df, ray_length=50000, ray_interval=1000)
