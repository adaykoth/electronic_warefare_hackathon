import numpy as np
from estimate_position import compute_emitter_position
import sys
from geometry import (
    lla_to_ecef, ecef_to_lla, lla_to_enu,
    ecef_to_enu, enu_to_ecef, sensor_direction_vector_enu
)
import pandas as pd
from filterpy.kalman import KalmanFilter as FilterPyKF

import pyarrow as pa
import pyarrow.ipc as ipc

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
        
        # Initialize state covariance (high initial uncertainty)
        self.kf.P = np.eye(3) * 100.0
        
        # Measurement noise covariance
        self.kf.R = np.eye(3) * (meas_noise_std ** 2)
        
        # Very small process noise for a stationary target
        self.kf.Q = np.eye(3) * 0.01
        
        # Initialize state to zero
        self.kf.x = np.zeros((3, 1))
        
        # Transformation parameters (to be set later)
        self.ref_ecef = None
        self.ref_lla = None

    def setup_coordinates(self, ref_lat, ref_lon, ref_alt):
        """Setup coordinate transformation parameters."""
        self.ref_lla = (ref_lat, ref_lon, ref_alt)
        # Store reference point for coordinate transformations
        self.ref_ecef = lla_to_ecef(np.radians(ref_lat), np.radians(ref_lon), ref_alt)

    def process_dataframe(self, df):
        """
        Process the entire dataframe of measurements for one label.
        
        Parameters:
            df: pandas DataFrame with sensor measurement data
            
        Returns:
            pandas DataFrame: Contains timestamped estimates with columns lat, lon, alt.
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
                
            # Get the next valid measurement index
            next_idx = df.index.get_loc(future_indices[0])
            if isinstance(next_idx, slice):
                next_idx = next_idx.start
            next_row = df.iloc[next_idx]
            next_time = future_indices[0]
            
            # Convert sensor positions to ENU coordinates
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
            
            # Get sensor direction vectors in ENU
            u1 = sensor_direction_vector_enu(
                current_row['azimuth'], current_row['elevation'],
                current_row['sensor_yaw'], current_row['sensor_pitch'], current_row['sensor_roll']
            )
            u2 = sensor_direction_vector_enu(
                next_row['azimuth'], next_row['elevation'],
                next_row['sensor_yaw'], next_row['sensor_pitch'], next_row['sensor_roll']
            )
            
            try:
                # Compute emitter position estimate in ENU coordinates
                position_estimates = compute_emitter_position(
                    s1, u1, current_row['amplitude'],
                    s2, u2, next_row['amplitude']
                )
                
                if position_estimates['geometric'] is not None:
                    E_meas = position_estimates['geometric']
                else:
                    raise ValueError("No valid geometric position estimate available")
                
                # For the first valid measurement, initialize the Kalman filter state
                if not first_valid_found:
                    self.kf.x = np.array(E_meas).reshape(3, 1)
                    first_valid_found = True
                    i = next_idx  # Move to next measurement
                    continue
                
                # Update Kalman filter with new measurement
                self.kf.predict()
                self.kf.update(np.array(E_meas).reshape(3, 1))
                
                # Get the current estimated state in ENU
                state = self.kf.x.flatten()
                enu_pos = state[0:3]
                # Convert ENU state back to ECEF
                ecef_pos = enu_to_ecef(enu_pos[0], enu_pos[1], enu_pos[2], 
                    self.ref_lla[0], self.ref_lla[1], self.ref_lla[2])
                # Convert ECEF to LLA (latitude, longitude, altitude)
                lat, lon, alt = ecef_to_lla(*ecef_pos)
                
                results.append({
                    'arrival_time': next_time,
                    'lat': lat,
                    'lon': lon,
                    'alt': alt
                })
                
            except ValueError as e:
                print(f"Error processing measurement pair at time {next_time}: {e}", file=sys.stderr)
            
            # Move to the next measurement
            i = next_idx
        
        # Create DataFrame from the results and set 'arrival_time' as the index
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df.set_index('arrival_time', inplace=True)
        return results_df

    @property
    def x(self):
        return self.kf.x

    @x.setter
    def x(self, value):
        self.kf.x = value

    def process_multiple_emitters(self, df):
        """
        Process measurements for all labels in the dataframe.
        
        Parameters:
            df: pandas DataFrame with a 'label' column and all required measurement columns.
            
        Returns:
            dict: Dictionary mapping label values to their respective result DataFrames.
        """
        if 'label' not in df.columns:
            raise ValueError("DataFrame must contain 'label' column")
            
        if self.ref_lla is None:
            raise ValueError("Coordinate system not initialized. Call setup_coordinates first.")
            
        # Get unique labels
        labels = df['label'].unique()
        results = {}
        
        # Process each label separately
        for lbl in labels:
            # Filter data for this label
            label_df = df[df['label'] == lbl].copy()
            label_df = label_df.drop('label', axis=1)
            
            # Reset Kalman filter state for new label
            self.kf.x = np.zeros((3, 1))
            self.kf.P = np.eye(3) * 100.0  # Reset covariance
            
            # Process data for this label
            res_df = self.process_dataframe(label_df)
            # Add a column to identify the label
            res_df['label'] = lbl
            results[lbl] = res_df
            
        return results

def main():
    from pathlib import Path
    from parse_data import load_window
    
        
    data_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    df = load_window(data_file)
    
    # Select required columns including 'label'
    columns_to_keep = [
        'arrival_time', 'azimuth', 'elevation', 'amplitude',
        'sensor_lat', 'sensor_lon', 'sensor_alt', 
        'sensor_yaw', 'sensor_pitch', 'sensor_roll', 
        'label'
    ]
    df = df.select(columns_to_keep)
    
    # Convert to a pandas DataFrame if not already
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
    
    # Setup coordinate system using the first sensor position as reference
    first_row = df.iloc[0]
    kf.setup_coordinates(
        first_row['sensor_lat'],
        first_row['sensor_lon'],
        first_row['sensor_alt']
    )
    
    # Process all labels
    results_dict = kf.process_multiple_emitters(df)
    
    # Concatenate results from all labels and sort by arrival_time (the index)
    if results_dict:
        final_results = pd.concat(results_dict.values())
        final_results.sort_index(inplace=True)
        print("\nConcatenated Results Sorted by arrival_time:")
        print(final_results)

        table = pa.Table.from_pandas(final_results)
        with pa.OSFile(str(output_file), "wb") as sink:
            with ipc.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
        print(f"Clustered data saved as IPC file to {output_file}")
    
if __name__ == "__main__":
    main()
