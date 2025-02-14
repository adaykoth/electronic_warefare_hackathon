import numpy as np

def sensor_direction_vector(azimuth, elevation):
    """
    Compute the unit direction vector in the sensor (plane) coordinate frame.
    
    Assumptions:
      - azimuth and elevation are given in radians.
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
    coordinate frame to the global frame. The sensor's orientation is
    given by Euler angles:
      - yaw   (rotation about the global vertical, z-axis; 0 means north),
      - pitch (rotation about the sensor's y-axis; with respect to the horizon),
      - roll  (rotation about the sensor's x-axis).
    
    We use the convention:
         R = R_z(yaw) * R_y(pitch) * R_x(roll)
    """
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    
    return Rz @ Rx @ Ry

def global_direction_vector(azimuth, elevation, sensor_yaw, sensor_pitch, sensor_roll):
    """
    Compute the emitter direction vector in the global frame.
    The sensor angles are assumed to be in degrees.
    They are converted to radians before computing the vector.
    
    Returns a unit vector in the global coordinate system 
    with the convention [North, East, Up].
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


def lla_to_ecef(lat_rad, lon_rad, alt):
    """
    Convert latitude, longitude (in radians) and altitude (in meters)
    into ECEF (Earth-Centered, Earth-Fixed) coordinates.
    """
    # WGS84 parameters
    a = 6378137.0            # semi-major axis in meters
    f = 1/298.257223563      # flattening
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
    Convert latitude, longitude (in degrees) and altitude (in meters) into local ENU coordinates.
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