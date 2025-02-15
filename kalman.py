import numpy as np
from estimate_position import compute_emitter_position
import sys
from geometry import (
    lla_to_ecef, ecef_to_lla, lla_to_enu,
    ecef_to_enu, enu_to_ecef, sensor_direction_vector_enu
)
import pandas as pd
from filterpy.kalman import KalmanFilter as FilterPyKF
import matplotlib.pyplot as plt

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
        first_valid_found = False
        
        i = 0
        while i < len(df) - 1:
            current_row = df.iloc[i]
            current_time = df.index[i]
            
            # Find next valid measurement time (at least min_measurement_dt seconds later)
            next_time_ns = current_time + int(self.min_measurement_dt * 1e9)  # Convert to nanoseconds
            future_indices = df.index[df.index > next_time_ns]
            
            if len(future_indices) == 0:
                break
                
            # Get the next valid measurement
            next_idx = df.index.get_loc(future_indices[0])
            next_row = df.iloc[next_idx]
            next_time = future_indices[0]
            
            # Convert sensor positions to ENU
            s1 = np.array(lla_to_enu(
                current_row['sensor_lat'], 
                current_row['sensor_lon'], 
                current_row['sensor_alt'],
                *self.ref_lla
            ))
            s2 = np.array(lla_to_enu(
                next_row['sensor_lat'], 
                next_row['sensor_lon'], 
                next_row['sensor_alt'],
                *self.ref_lla
            ))
            
            # Get direction vectors
            u1 = sensor_direction_vector_enu(
                current_row['azimuth'], current_row['elevation'],
                current_row['sensor_yaw'], current_row['sensor_pitch'], current_row['sensor_roll']
            )
            u2 = sensor_direction_vector_enu(
                next_row['azimuth'], next_row['elevation'],
                next_row['sensor_yaw'], next_row['sensor_pitch'], next_row['sensor_roll']
            )
            
            try:
                # Get position estimate in ENU
                position_estimates = compute_emitter_position(
                    s1, u1, current_row['amplitude'],
                    s2, u2, next_row['amplitude']
                )
                
                # Only use geometric estimate
                if position_estimates['geometric'] is not None:
                    E_meas = position_estimates['geometric']
                else:
                    raise ValueError("No valid geometric position estimate available")
                
                # Initialize state with first valid measurement
                if not first_valid_found:
                    self.kf.x = np.array(E_meas).reshape(3, 1)
                    first_valid_found = True
                    i = next_idx  # Move to the next measurement
                    continue
                
                # Update Kalman filter
                self.kf.predict()
                self.kf.update(np.array(E_meas).reshape(3, 1))
                
                # Store results with the current timestamp
                state = self.kf.x.flatten()
                enu_pos = state[0:3]
                ecef_pos = enu_to_ecef(enu_pos[0], enu_pos[1], enu_pos[2], 
                    self.ref_lla[0], self.ref_lla[1], self.ref_lla[2])
                lat, lon, alt = ecef_to_lla(*ecef_pos)
                
                results.append({
                    'index': next_time,
                    'lat': lat,
                    'lon': lon,
                    'alt': alt
                })
                
            except ValueError as e:
                print(f"Error processing measurement pair at time {next_time}: {e}", file=sys.stderr)
            
            # Move to the next measurement
            i = next_idx
            
        # Create DataFrame and set index
        results_df = pd.DataFrame(results)
        if len(results_df) > 0:
            results_df.set_index('index', inplace=True)
        return results_df

    @property
    def x(self):
        return self.kf.x

    @x.setter
    def x(self, value):
        self.kf.x = value

    def process_multiple_emitters(self, df):
        """
        Process measurements for all emitters in the dataframe.
        
        Parameters:
            df: pandas DataFrame with 'emitter' column and all required measurement columns
            
        Returns:
            dict: Dictionary mapping emitter IDs to their respective result DataFrames
        """
        if 'emitter' not in df.columns:
            raise ValueError("DataFrame must contain 'emitter' column")
            
        if self.ref_lla is None:
            raise ValueError("Coordinate system not initialized. Call setup_coordinates first.")
            
        # Get unique emitters
        emitters = df['emitter'].unique()
        results = {}
        
        # Process each emitter separately
        for emitter in emitters:
            # Filter data for this emitter
            emitter_df = df[df['emitter'] == emitter].copy()
            emitter_df = emitter_df.drop('emitter', axis=1)
            
            # Reset Kalman filter state for new emitter
            self.kf.x = np.zeros((3, 1))
            self.kf.P = np.eye(3) * 100.0  # Reset covariance
            
            # Process this emitter's data
            results[emitter] = self.process_dataframe(emitter_df)
            
        return results

def main():
    from pathlib import Path
    from parse_data import load_window
    
    # Load data
    if len(sys.argv) < 2:
        print("Usage: python kalman.py <data_file>")
        return
        
    data_file = Path(sys.argv[1])
    df = load_window(data_file)
    
    # Modified main function to demonstrate multiple emitter processing
    df = load_window(data_file)
    
    # Select required columns including emitter
    columns_to_keep = [
        'arrival_time', 'azimuth', 'elevation', 'amplitude',
        'sensor_lat', 'sensor_lon', 'sensor_alt', 
        'sensor_yaw', 'sensor_pitch', 'sensor_roll', 
        'emitter_lat', 'emitter_lon', 'emitter_alt',
        'emitter'  # Keep the emitter column
    ]
    df = df.select(columns_to_keep)
    
    # Convert to pandas DataFrame if not already
    if not isinstance(df, pd.DataFrame):
        df = df.to_pandas()
    df.set_index('arrival_time', inplace=True)
    
    print("Input DataFrame:")
    print(df)
    
    # Initialize Kalman filter
    kf = KalmanFilter(
        min_measurement_dt=1,
        meas_noise_std=5.0
    )
    
    # Setup coordinate system using first sensor position as reference
    first_row = df.iloc[0]
    kf.setup_coordinates(
        first_row['sensor_lat'],
        first_row['sensor_lon'],
        first_row['sensor_alt']
    )
    
    # Process all emitters
    results = kf.process_multiple_emitters(df)
    
    # Print and plot results for each emitter
    all_errors = []
    emitter_stats = {}
    
    for emitter, results_df in results.items():
        print(f"\nResults for {emitter}:")
        print(results_df)
        
        # Get true position for this emitter
        emitter_data = df[df['emitter'] == emitter].iloc[0]
        true_enu = np.array(lla_to_enu(
            emitter_data['emitter_lat'],
            emitter_data['emitter_lon'],
            emitter_data['emitter_alt'],
            first_row['sensor_lat'],
            first_row['sensor_lon'],
            first_row['sensor_alt']
        ))
        print(f"\nTrue {emitter} position (ENU):", true_enu)
        
        # Calculate errors
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
            all_errors.append(error)
        
        # Calculate statistics for this emitter
        rms_error = np.sqrt(np.mean(np.array(errors)**2))
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        std_error = np.std(errors)
        max_error = np.max(errors)
        min_error = np.min(errors)
        
        emitter_stats[emitter] = {
            'rms': rms_error,
            'mean': mean_error,
            'median': median_error,
            'std': std_error,
            'max': max_error,
            'min': min_error,
            'num_measurements': len(errors)
        }
        
        print(f"\nStatistics for {emitter}:")
        print(f"RMS Error: {rms_error:.2f} meters")
        print(f"Mean Error: {mean_error:.2f} meters")
        print(f"Median Error: {median_error:.2f} meters")
        print(f"Std Dev: {std_error:.2f} meters")
        print(f"Max Error: {max_error:.2f} meters")
        print(f"Min Error: {min_error:.2f} meters")
        print(f"Number of measurements: {len(errors)}")
        
        # Plot results for this emitter
        try:
            # Calculate indices for middle 70%
            n_measurements = len(results_df)
            start_idx = int(n_measurements * 0.25)  # Skip first 15%
            end_idx = int(n_measurements * 0.8)    # Skip last 15%
            
            # Get the time slice for middle 70%
            middle_times = results_df.index[start_idx:end_idx]
            middle_slice = results_df.loc[middle_times]
            middle_errors = errors[start_idx:end_idx]
            
            fig = plt.figure(figsize=(6, 6))
            plt.suptitle(f'Results for {emitter}')
            
            plt.subplot(411)
            plt.plot(middle_slice.index, middle_slice['lat'], 'b-', label='Estimated')
            plt.axhline(y=emitter_data['emitter_lat'], color='r', linestyle='--', label='True')
            plt.ylabel('Latitude')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(412)
            plt.plot(middle_slice.index, middle_slice['lon'], 'g-', label='Estimated')
            plt.axhline(y=emitter_data['emitter_lon'], color='r', linestyle='--', label='True')
            plt.ylabel('Longitude')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(413)
            plt.plot(middle_slice.index, middle_slice['alt'], 'k-', label='Estimated')
            plt.axhline(y=emitter_data['emitter_alt'], color='r', linestyle='--', label='True')
            plt.ylabel('Altitude (m)')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(414)
            plt.plot(middle_slice.index, middle_errors, 'm-')
            middle_rms = np.sqrt(np.mean(np.array(middle_errors)**2))
            plt.axhline(y=middle_rms, color='r', linestyle='--', 
                       label=f'{middle_rms:.2f}m')
            plt.ylabel('Error (m)')
            plt.xlabel('Time')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            
            # Save figure with DPI=200
            fig.savefig(f'kalman_results_{emitter}.png', dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()
            
        except Exception as e:
            print(f"Error creating plot for {emitter}: {e}")
    
    # Calculate and print overall statistics
    all_errors = np.array(all_errors)
    total_measurements = sum(stats['num_measurements'] for stats in emitter_stats.values())
    
    print("\n=== OVERALL STATISTICS ===")
    print(f"Total number of measurements: {total_measurements}")
    print(f"Overall RMS Error: {np.sqrt(np.mean(all_errors**2)):.2f} meters")
    print(f"Overall Mean Error: {np.mean(all_errors):.2f} meters")
    print(f"Overall Median Error: {np.median(all_errors):.2f} meters")
    print(f"Overall Std Dev: {np.std(all_errors):.2f} meters")
    print(f"Overall Max Error: {np.max(all_errors):.2f} meters")
    print(f"Overall Min Error: {np.min(all_errors):.2f} meters")
    
    # Print summary table with added median column
    print("\n=== EMITTER SUMMARY ===")
    print("Emitter      | RMS Error | Mean Error | Median Error | Std Dev | Max Error | Min Error | # Measurements")
    print("-" * 98)
    for emitter, stats in emitter_stats.items():
        print(f"{emitter:11} | {stats['rms']:9.2f} | {stats['mean']:10.2f} | {stats['median']:12.2f} | "
              f"{stats['std']:7.2f} | {stats['max']:9.2f} | {stats['min']:9.2f} | {stats['num_measurements']:14d}")

if __name__ == "__main__":
    main()
