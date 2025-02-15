import numpy as np
from estimate_position import compute_emitter_position
import sys
from geometry import (
    lla_to_ecef, ecef_to_lla, lla_to_enu,
    ecef_to_enu, enu_to_ecef, sensor_direction_vector_enu
)
import pandas as pd
from filterpy.kalman import KalmanFilter as FilterPyKF

class KalmanFilter:
    def __init__(self, min_measurement_dt, meas_noise_std=5.0):
        """
        Initialize Kalman filter for stationary emitter tracking.
        
        Parameters:
            min_measurement_dt: float, minimum time difference between measurements (seconds)
            meas_noise_std: float, measurement noise standard deviation (m)
        """
        # Create FilterPy Kalman filter with 3 state variables (position only) and 3 measurements
        self.kf = FilterPyKF(dim_x=3, dim_z=3)
        self.min_measurement_dt = min_measurement_dt
        
        # State transition matrix is identity (stationary target)
        self.kf.F = np.eye(3)
        
        # Measurement matrix is identity (we directly measure position)
        self.kf.H = np.eye(3)
        
        # Initialize state covariance
        self.kf.P = np.eye(3) * 100.0  # High initial position uncertainty
        
        # Measurement noise
        self.kf.R = np.eye(3) * (meas_noise_std ** 2)
        
        # Very small process noise for stationary target
        self.kf.Q = np.eye(3) * 0.01
        
        # Initialize state to zero
        self.kf.x = np.zeros((3, 1))
        
        # Store transformation parameters
        self.ref_ecef = None
        self.ref_lla = None

    def setup_coordinates(self, ref_lat, ref_lon, ref_alt):
        """Setup coordinate transformation parameters."""
        self.ref_lla = (ref_lat, ref_lon, ref_alt)
        # Store reference point for coordinate transformations
        self.ref_ecef = lla_to_ecef(np.radians(ref_lat), np.radians(ref_lon), ref_alt)

    def process_dataframe(self, df):
        """
        Process the entire dataframe of measurements.
        """
        if self.ref_lla is None:
            raise ValueError("Coordinate system not initialized. Call setup_coordinates first.")
            
        results = []
        prev_row = None
        prev_time = None
        
        # Initialize state with first valid measurement
        first_valid_found = False
        
        for idx, row in df.iterrows():
            if prev_row is None:
                prev_row = row
                prev_time = idx
                continue
                
            # Convert nanoseconds to seconds for dt
            dt = (idx - prev_time) / 1e9
            if dt < self.min_measurement_dt:
                continue
                
            # Convert sensor positions to ENU
            s1 = np.array(lla_to_enu(
                prev_row['sensor_lat'], 
                prev_row['sensor_lon'], 
                prev_row['sensor_alt'],
                *self.ref_lla
            ))
            s2 = np.array(lla_to_enu(
                row['sensor_lat'], 
                row['sensor_lon'], 
                row['sensor_alt'],
                *self.ref_lla
            ))
            
            # Get direction vectors
            u1 = sensor_direction_vector_enu(
                prev_row['azimuth'], prev_row['elevation'],
                prev_row['sensor_yaw'], prev_row['sensor_pitch'], prev_row['sensor_roll']
            )
            u2 = sensor_direction_vector_enu(
                row['azimuth'], row['elevation'],
                row['sensor_yaw'], row['sensor_pitch'], row['sensor_roll']
            )
            
            try:
                # Get position estimate in ENU
                position_estimates = compute_emitter_position(
                    s1, u1, prev_row['amplitude'],
                    s2, u2, row['amplitude']
                )
                
                # Only use geometric estimate
                if position_estimates['geometric'] is not None:
                    E_meas = position_estimates['geometric']
                else:
                    raise ValueError("No valid geometric position estimate available")
                
                # Initialize state with first valid measurement
                if not first_valid_found:
                    self.kf.x = E_meas.reshape(3, 1)
                    first_valid_found = True
                    prev_row = row
                    prev_time = idx
                    continue
                
                # Update Kalman filter
                self.kf.predict()
                self.kf.update(E_meas.reshape(3, 1))
                
                # Store results with the current timestamp
                state = self.kf.x.flatten()
                enu_pos = state[0:3]
                ecef_pos = enu_to_ecef(enu_pos[0], enu_pos[1], enu_pos[2], 
                    self.ref_lla[0], self.ref_lla[1], self.ref_lla[2])
                lat, lon, alt = ecef_to_lla(*ecef_pos)
                
                results.append({
                    'timestamp': idx,
                    'lat': lat,
                    'lon': lon,
                    'alt': alt
                })
                
            except ValueError as e:
                print(f"Error processing measurement at time {idx}: {e}", file=sys.stderr)
            
            prev_row = row
            prev_time = idx
            
        return pd.DataFrame(results).set_index('timestamp')

    @property
    def x(self):
        return self.kf.x

    @x.setter
    def x(self, value):
        self.kf.x = value

def main():
    from pathlib import Path
    from parse_data import load_window
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Load data
    if len(sys.argv) < 2:
        print("Usage: python kalman.py <data_file>")
        return
        
    data_file = Path(sys.argv[1])
    df = load_window(data_file)
    
    # Filter for Emitter2
    emitter = "Emitter1"
    df = df.filter(df["emitter"] == emitter).drop("emitter")
    
    # Select required columns
    columns_to_keep = [
        'arrival_time', 'azimuth', 'elevation', 'amplitude',
        'sensor_lat', 'sensor_lon', 'sensor_alt', 
        'sensor_yaw', 'sensor_pitch', 'sensor_roll', 
        'emitter_lat', 'emitter_lon', 'emitter_alt'
    ]
    df = df.select(columns_to_keep)
    
    # Convert to pandas DataFrame if not already
    if not isinstance(df, pd.DataFrame):
        df = df.to_pandas()
    df.set_index('arrival_time', inplace=True)
    
    print("Input DataFrame:")
    print(df)
    
    # Initialize Kalman filter with better parameters for slow-moving target
    kf = KalmanFilter(
        min_measurement_dt=0.5,     # minimum time between measurements in seconds
        meas_noise_std=1.0         # slightly increased measurement noise
    )
    
    # Setup coordinate system using first sensor position as reference
    first_row = df.iloc[0]
    kf.setup_coordinates(
        first_row['sensor_lat'],
        first_row['sensor_lon'],
        first_row['sensor_alt']
    )
    
    # Process all measurements
    results_df = kf.process_dataframe(df)
    
    # Print results
    print("\nResults:")
    print(results_df)
    
    # Calculate true emitter position in ENU
    true_enu = np.array(lla_to_enu(
        first_row['emitter_lat'],
        first_row['emitter_lon'],
        first_row['emitter_alt'],
        first_row['sensor_lat'],
        first_row['sensor_lon'],
        first_row['sensor_alt']
    ))
    print("\nTrue emitter position (ENU):", true_enu)
    
    # Calculate RMS error
    errors = []
    for _, row in results_df.iterrows():
        est_enu = np.array(lla_to_enu(
            row['lat'], row['lon'], row['alt'],
            first_row['sensor_lat'],
            first_row['sensor_lon'],
            first_row['sensor_alt']
        ))
        error = np.linalg.norm(est_enu - true_enu)
        errors.append(error)
    
    rms_error = np.sqrt(np.mean(np.array(errors)**2))
    print(f"\nRMS Error: {rms_error:.2f} meters")
    
    # Plot results
    try:
        # Position plots
        plt.figure(figsize=(12, 12))
        
        plt.subplot(411)
        plt.plot(results_df.index, results_df['lat'], 'b-', label='Estimated')
        plt.axhline(y=first_row['emitter_lat'], color='r', linestyle='--', label='True')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(412)
        plt.plot(results_df.index, results_df['lon'], 'g-', label='Estimated')
        plt.axhline(y=first_row['emitter_lon'], color='r', linestyle='--', label='True')
        plt.ylabel('Longitude')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(413)
        plt.plot(results_df.index, results_df['alt'], 'k-', label='Estimated')
        plt.axhline(y=first_row['emitter_alt'], color='r', linestyle='--', label='True')
        plt.ylabel('Altitude (m)')
        plt.legend()
        plt.grid(True)
        
        # Error plot
        plt.subplot(414)
        plt.plot(results_df.index, errors, 'm-')
        plt.axhline(y=rms_error, color='r', linestyle='--', label=f'RMS Error: {rms_error:.2f}m')
        plt.ylabel('Error (m)')
        plt.xlabel('Time')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error creating plot: {e}")

if __name__ == "__main__":
    main()
