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

def compute_emitter_position(s1, u1, A1, s2, u2, A2):
    """
    Compute emitter position using both geometric and amplitude-based methods.
    All inputs and outputs are in ENU coordinates.
    
    Parameters:
        s1, s2: sensor positions in ENU
        u1, u2: direction vectors in ENU
        A1, A2: received amplitudes in dBm
    
    Returns:
        dict containing:
            - geometric: np.array(3,) position from geometric method
            - amplitude: np.array(3,) position from amplitude method or None
    """
    # Geometric method: compute intersection
    geometric_enu = compute_intersection(s1, u1, s2, u2)
    
    # Amplitude method
    try:
        amplitude_enu = compute_amplitude_position(
            s1, u1, A1,
            s2, u2, A2
        )
    except ValueError:
        amplitude_enu = None
    
    return {
        'geometric': geometric_enu,
        'amplitude': amplitude_enu
    }

def dbm_to_power_ratio(A1_dbm, A2_dbm):
    """
    Convert two dBm measurements to a distance ratio.
    
    Parameters:
        A1_dbm, A2_dbm: Signal strengths in dBm (typically negative values)
                        e.g., -60 dBm represents 10^(-6) mW
        
    Returns:
        float: ratio of distances (d2/d1)
        
    Notes:
        In free space:
        - dBm = 10 * log10(P/1mW)
        - Power ~ 1/r^2
        - Therefore, r ~ 1/sqrt(P)
        - ratio = d2/d1 = sqrt(P1/P2)
        
    Example:
        A1 = -60 dBm, A2 = -70 dBm
        P1 = 10^(-6) mW, P2 = 10^(-7) mW
        ratio = sqrt(P2/P1) ≈ 3.16
        This means the second sensor is about 3.16 times farther from the emitter
    """
    # Convert dBm to linear power (mW)
    # P(mW) = 10^(dBm/10)
    # For negative dBm, this gives a fraction of a mW
    P1 = 10.0 ** (A1_dbm/10.0)
    P2 = 10.0 ** (A2_dbm/10.0)
    
    # Validate power values
    if P1 <= 0 or P2 <= 0:
        raise ValueError(f"Invalid power values calculated: P1={P1}, P2={P2}")
    
    # Distance ratio = sqrt(P1/P2)
    # Note: higher power means shorter distance
    return np.sqrt(P2/P1)

def compute_amplitude_position(s1, u1, A1, s2, u2, A2):
    """
    Compute emitter position using amplitude-based method.
    All inputs are in ENU coordinates.
    
    Parameters:
        s1, s2: sensor positions in ENU
        u1, u2: direction vectors in ENU
        A1, A2: received amplitudes in dBm
    
    Returns:
        np.array: estimated position in ENU coordinates
    """
    # Get distance ratio from dBm measurements
    ratio = dbm_to_power_ratio(A1, A2)
    
    # Equation: s1 + d1*u1 = s2 + d2*u2, with d2 = ratio*d1
    delta = s2 - s1
    direction_diff = u1 - ratio * u2
    norm_diff = np.linalg.norm(direction_diff)
    
    if norm_diff < 1e-6:
        raise ValueError("Degenerate geometry: direction difference too small")
    
    # Solve for d1 by projecting delta onto direction_diff
    d1 = np.dot(delta, direction_diff) / (norm_diff**2)
    d2 = ratio * d1
    
    # Compute emitter positions from both sensor measurements
    E1 = s1 + d1 * u1
    E2 = s2 + d2 * u2
    
    # Return average position
    return (E1 + E2) / 2.0

if __name__ == "__main__":
    file = Path(sys.argv[1])
    df = load_window(file)

    emitter = "Emitter2"

    # Filter the DataFrame for the specified emitter and select the required columns.
    df = df.filter(df["emitter"] == emitter).drop("emitter")
    columns_to_keep = [
        'arrival_time', 'azimuth', 'elevation', 'amplitude',
        'sensor_lat', 'sensor_lon', 'sensor_alt', 
        'sensor_yaw', 'sensor_pitch', 'sensor_roll', 
        'emitter_lat', 'emitter_lon', 'emitter_alt'
    ]
    df = df.select(columns_to_keep)

    df = df.to_pandas()
    df.set_index("arrival_time", inplace=True)

    print("Input DataFrame:")
    print(df)

    # Use the emitter coordinates from the first row as the ENU reference.
    first_row = df.iloc[0]
    ref_lat = first_row['emitter_lat']
    ref_lon = first_row['emitter_lon']
    ref_alt = first_row['emitter_alt']

    # Process pairs of measurements
    results = []
    for i in range(1, len(df)):
        prev_data = df.iloc[i-1].to_dict()
        curr_data = df.iloc[i].to_dict()

        # Prepare data dictionaries
        data1 = {
            'lat': prev_data['sensor_lat'],
            'lon': prev_data['sensor_lon'],
            'alt': prev_data['sensor_alt'],
            'azimuth': prev_data['azimuth'],
            'elevation': prev_data['elevation'],
            'yaw': prev_data['sensor_yaw'],
            'pitch': prev_data['sensor_pitch'],
            'roll': prev_data['sensor_roll'],
            'amplitude': prev_data['amplitude']
        }
        
        data2 = {
            'lat': curr_data['sensor_lat'],
            'lon': curr_data['sensor_lon'],
            'alt': curr_data['sensor_alt'],
            'azimuth': curr_data['azimuth'],
            'elevation': curr_data['elevation'],
            'yaw': curr_data['sensor_yaw'],
            'pitch': curr_data['sensor_pitch'],
            'roll': curr_data['sensor_roll'],
            'amplitude': curr_data['amplitude']
        }

        # Compute position estimates
        try:
            position_estimates = compute_emitter_position(
                data1['lat'], data1['lon'], data1['alt'],
                data2['lat'], data2['lon'], data2['alt'],
                data1['amplitude'], data2['amplitude']
            )
            
            result = {
                'timestamp': df.index[i],
                'geometric_lat': position_estimates['geometric'][0],
                'geometric_lon': position_estimates['geometric'][1],
                'geometric_alt': position_estimates['geometric'][2],
            }
            
            if position_estimates['amplitude'] is not None:
                result.update({
                    'amplitude_lat': position_estimates['amplitude'][0],
                    'amplitude_lon': position_estimates['amplitude'][1],
                    'amplitude_alt': position_estimates['amplitude'][2],
                })
            else:
                result.update({
                    'amplitude_lat': None,
                    'amplitude_lon': None,
                    'amplitude_alt': None,
                })
            
            results.append(result)
            
        except ValueError as e:
            print(f"Error processing measurement pair at index {i}: {e}")
            continue

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df.set_index('timestamp', inplace=True)

    # Calculate errors
    true_position = {
        'lat': first_row['emitter_lat'],
        'lon': first_row['emitter_lon'],
        'alt': first_row['emitter_alt']
    }

    # Function to calculate distance error in meters
    def calculate_distance_error(row, method):
        if pd.isna(row[f'{method}_lat']):
            return None
            
        est_enu = lla_to_enu(
            row[f'{method}_lat'], 
            row[f'{method}_lon'], 
            row[f'{method}_alt'],
            ref_lat, ref_lon, ref_alt
        )
        true_enu = lla_to_enu(
            true_position['lat'],
            true_position['lon'],
            true_position['alt'],
            ref_lat, ref_lon, ref_alt
        )
        return np.linalg.norm(np.array(est_enu) - np.array(true_enu))

    # Calculate errors for both methods
    results_df['geometric_error'] = results_df.apply(
        lambda row: calculate_distance_error(row, 'geometric'), axis=1
    )
    results_df['amplitude_error'] = results_df.apply(
        lambda row: calculate_distance_error(row, 'amplitude'), axis=1
    )

    # Print summary statistics
    print("\nError Statistics (meters):")
    print("\nGeometric Method:")
    print(results_df['geometric_error'].describe())
    print("\nAmplitude Method:")
    print(results_df['amplitude_error'].describe())

    # Optional: Plot error comparison
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(results_df.index, results_df['geometric_error'], label='Geometric Method')
        plt.plot(results_df.index, results_df['amplitude_error'], label='Amplitude Method')
        plt.xlabel('Time')
        plt.ylabel('Error (meters)')
        plt.title('Position Estimation Error Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"Error creating plot: {e}")
