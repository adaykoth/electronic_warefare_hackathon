import numpy as np

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

def ecef_to_lla(x, y, z):
    """
    Convert ECEF coordinates (in meters) to LLA (latitude and longitude in degrees, altitude in meters)
    using the WGS84 ellipsoid.
    
    Based on the Bowring formula for initial guess.
    """
    # WGS84 parameters
    a = 6378137.0            # semi-major axis in meters
    f = 1 / 298.257223563    # flattening
    b = a * (1 - f)          # semi-minor axis
    e_sq = f * (2 - f)       # eccentricity squared
    e = np.sqrt(e_sq)
    
    p = np.sqrt(x**2 + y**2)
    # Initial guess for latitude
    theta = np.arctan2(z * a, p * b)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z + e**2 * b * np.sin(theta)**3, p - e_sq * a * np.cos(theta)**3)
    N = a / np.sqrt(1 - e_sq * np.sin(lat)**2)
    alt = p / np.cos(lat) - N
    
    # Convert lat, lon to degrees
    lat_deg = np.degrees(lat)
    lon_deg = np.degrees(lon)
    
    return lat_deg, lon_deg, alt

def enu_to_ecef(e, n, u, ref_lat, ref_lon, ref_alt):
    """
    Convert local ENU coordinates to ECEF coordinates.
    
    The ENU coordinate system is defined with respect to a reference point
    specified by (ref_lat, ref_lon, ref_alt), where ref_lat and ref_lon are in degrees
    and ref_alt in meters. This function uses the inverse of the ENU conversion.
    """
    # Convert reference latitude/longitude to radians.
    lat0 = np.radians(ref_lat)
    lon0 = np.radians(ref_lon)
    # Get the ECEF coordinates of the reference point.
    x0, y0, z0 = lla_to_ecef(lat0, lon0, ref_alt)
    
    # Inverse rotation from ENU to ECEF (see, e.g., standard geodesy texts)
    x = x0 - np.sin(lon0)*e - np.sin(lat0)*np.cos(lon0)*n + np.cos(lat0)*np.cos(lon0)*u
    y = y0 + np.cos(lon0)*e - np.sin(lat0)*np.sin(lon0)*n + np.cos(lat0)*np.sin(lon0)*u
    z = z0 + np.cos(lat0)*n + np.sin(lat0)*u
    return x, y, z

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