import numpy as np
import plotly.graph_objects as go
from geometry_new import sensor_to_enu_vector, aircraft_to_enu_rotation, create_enu_from_lla, ecef_to_enu, lla_to_ecef

def create_test_visualization(test_cases):
    """
    Create a 3D visualization of test vectors.
    Each test case will show:
    - The original vector
    - The coordinate axes (ENU)
    - The aircraft orientation (if applicable)
    """
    # Create figure
    fig = go.Figure()
    
    # Origin point
    origin = np.array([0, 0, 0])
    
    # Add coordinate system axes
    axis_length = 1.0
    # East axis (red)
    fig.add_trace(go.Scatter3d(
        x=[0, axis_length], y=[0, 0], z=[0, 0],
        line=dict(color='red', width=2),
        name='East'
    ))
    # North axis (green)
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, axis_length], z=[0, 0],
        line=dict(color='green', width=2),
        name='North'
    ))
    # Up axis (blue)
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, axis_length],
        line=dict(color='blue', width=2),
        name='Up'
    ))
    
    # Add test vectors
    for i, test in enumerate(test_cases):
        vector = test['vector']
        # Normalize vector for visualization
        vector_norm = vector / np.linalg.norm(vector)
        
        fig.add_trace(go.Scatter3d(
            x=[0, vector_norm[0]], 
            y=[0, vector_norm[1]], 
            z=[0, vector_norm[2]],
            line=dict(color='purple', width=4),
            name=f'Test {i+1}: {test["description"]}'
        ))
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title="East",
            yaxis_title="North",
            zaxis_title="Up",
            aspectmode='cube'
        ),
        title="Geometry Test Visualization"
    )
    
    fig.show()

def run_basic_tests():
    """Run basic tests for sensor vector calculations."""
    test_cases = []
    
    # Test 1: Forward-pointing vector (zero angles)
    vector = sensor_to_enu_vector(
        azimuth=0,
        elevation=0,
        yaw=0,
        pitch=0,
        roll=0
    )
    test_cases.append({
        'description': 'Forward (North)',
        'vector': vector,
        'expected': np.array([0, 1, 0])  # Should point North in ENU
    })
    
    # Test 2: 45-degree elevation
    vector = sensor_to_enu_vector(
        azimuth=0,
        elevation=45,
        yaw=0,
        pitch=0,
        roll=0
    )
    test_cases.append({
        'description': 'Forward+Up (45° elevation)',
        'vector': vector,
        'expected': np.array([0, 1/np.sqrt(2), 1/np.sqrt(2)])
    })
    
    # Test 3: 90-degree azimuth (right)
    vector = sensor_to_enu_vector(
        azimuth=90,
        elevation=0,
        yaw=0,
        pitch=0,
        roll=0
    )
    test_cases.append({
        'description': 'Right (East)',
        'vector': vector,
        'expected': np.array([1, 0, 0])
    })
    
    # Test 4: Aircraft yawed 90° right
    vector = sensor_to_enu_vector(
        azimuth=0,
        elevation=0,
        yaw=90,
        pitch=0,
        roll=0
    )
    test_cases.append({
        'description': 'Forward with 90° Yaw',
        'vector': vector,
        'expected': np.array([1, 0, 0])
    })
    
    # Print numerical results
    print("=== Numerical Test Results ===")
    for i, test in enumerate(test_cases):
        vector = test['vector']
        expected = test['expected']
        print(f"\nTest {i+1}: {test['description']}")
        print(f"Vector:   {vector}")
        print(f"Expected: {expected}")
        print(f"Magnitude: {np.linalg.norm(vector):.6f}")
        if 'expected' in test:
            angle = np.arccos(np.dot(vector, expected) / 
                            (np.linalg.norm(vector) * np.linalg.norm(expected)))
            print(f"Angle error: {np.degrees(angle):.2f}°")
    
    # Visualize results
    create_test_visualization(test_cases)

def run_aircraft_orientation_tests():
    """Test various aircraft orientations."""
    test_cases = []
    
    # Test different aircraft orientations with expected vectors
    orientations = [
        {
            'yaw': 0, 'pitch': 30, 'roll': 0, 
            'desc': '30° Pitch Up',
            'expected': np.array([0, np.cos(np.radians(30)), np.sin(np.radians(30))])  # Forward vector pitched up 30°
        },
        {
            'yaw': 0, 'pitch': 0, 'roll': 45, 
            'desc': '45° Roll Right',
            'expected': np.array([0, 1, 0])  # Forward vector should still point North
        },
        {
            'yaw': 45, 'pitch': 30, 'roll': 0, 
            'desc': 'Yaw 45° + Pitch 30°',
            'expected': np.array(
                [np.cos(np.radians(45)) * np.cos(np.radians(30)),  # East component
                 np.sin(np.radians(45)) * np.cos(np.radians(30)),  # North component
                 np.sin(np.radians(30))]                           # Up component
            )
        },
    ]
    
    for orient in orientations:
        vector = sensor_to_enu_vector(
            azimuth=0,
            elevation=0,
            yaw=orient['yaw'],
            pitch=orient['pitch'],
            roll=orient['roll']
        )
        test_cases.append({
            'description': orient['desc'],
            'vector': vector,
            'expected': orient['expected']
        })
    
    # Print numerical results
    print("\n=== Aircraft Orientation Tests ===")
    for i, test in enumerate(test_cases):
        vector = test['vector']
        expected = test['expected']
        print(f"\nTest {i+1}: {test['description']}")
        print(f"Vector:   {vector}")
        print(f"Expected: {expected}")
        print(f"Magnitude: {np.linalg.norm(vector):.6f}")
        
        # Calculate angle error
        angle = np.arccos(np.dot(vector, expected) / 
                         (np.linalg.norm(vector) * np.linalg.norm(expected)))
        print(f"Angle error: {np.degrees(angle):.2f}°")
        
        # Calculate component errors
        error = vector - expected
        print(f"Component errors (E,N,U): ({error[0]:.3f}, {error[1]:.3f}, {error[2]:.3f})")
    
    # Visualize results
    create_test_visualization(test_cases)

def compare_position_and_sensor_vectors(df):
    """
    Compare direction vectors calculated from:
    1. Relative positions (emitter - sensor)
    2. Sensor measurements (azimuth/elevation)
    
    Args:
        df: DataFrame containing sensor and emitter positions plus sensor measurements
    """
    print("\n=== Position vs Sensor Vector Comparison ===")
    
    # Get reference point (first sensor position)
    lat_ref = df["sensor_lat"].iloc[0]
    lon_ref = df["sensor_lon"].iloc[0]
    alt_ref = df["sensor_alt"].iloc[0]
    
    # Create ENU transformation matrices
    R_ecef_to_enu, R_enu_to_ecef, ref_ecef = create_enu_from_lla(lat_ref, lon_ref, alt_ref)
    
    results = []
    for idx, row in df.iterrows():
        # Calculate vector from positions
        sensor_ecef = lla_to_ecef(row["sensor_lat"], row["sensor_lon"], row["sensor_alt"])
        emitter_ecef = lla_to_ecef(row["emitter_lat"], row["emitter_lon"], row["emitter_alt"])
        
        # Convert both to ENU
        sensor_enu = ecef_to_enu(sensor_ecef, R_ecef_to_enu, ref_ecef)
        emitter_enu = ecef_to_enu(emitter_ecef, R_ecef_to_enu, ref_ecef)
        
        # Calculate direction vector from positions (in ENU)
        pos_vector = emitter_enu - sensor_enu
        pos_vector = pos_vector / np.linalg.norm(pos_vector)
        
        # Calculate direction vector from sensor measurements
        sensor_vector = sensor_to_enu_vector(
            row["azimuth"],
            row["elevation"],
            row["sensor_yaw"],
            row["sensor_pitch"],
            row["sensor_roll"]
        )
        
        # Calculate angle between vectors
        angle = np.arccos(np.dot(pos_vector, sensor_vector))
        angle_deg = np.degrees(angle)
        
        # Store results
        results.append({
            'time': idx,
            'pos_vector': pos_vector,
            'sensor_vector': sensor_vector,
            'angle_error': angle_deg,
            'pos_enu': (sensor_enu, emitter_enu)
        })
        
        # Print detailed results for first few points
        if len(results) <= 5:
            print(f"\nTime: {idx}")
            print(f"Position-based vector (ENU): {pos_vector}")
            print(f"Sensor-based vector (ENU):   {sensor_vector}")
            print(f"Angle difference: {angle_deg:.2f}°")
            print(f"Component differences (E,N,U): {(pos_vector - sensor_vector)}")
    
    # Print summary statistics
    angles = [r['angle_error'] for r in results]
    print("\nSummary Statistics:")
    print(f"Mean angle error: {np.mean(angles):.2f}°")
    print(f"Max angle error:  {np.max(angles):.2f}°")
    print(f"Min angle error:  {np.min(angles):.2f}°")
    print(f"Std angle error:  {np.std(angles):.2f}°")
    
    # Visualize a few test cases
    visualize_vector_comparison(results[:5])

def visualize_vector_comparison(results):
    """
    Create a 3D visualization comparing position-based and sensor-based vectors.
    """
    fig = go.Figure()
    
    # Add coordinate system axes
    axis_length = 1.0
    fig.add_trace(go.Scatter3d(
        x=[0, axis_length], y=[0, 0], z=[0, 0],
        line=dict(color='red', width=2),
        name='East'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, axis_length], z=[0, 0],
        line=dict(color='green', width=2),
        name='North'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, axis_length],
        line=dict(color='blue', width=2),
        name='Up'
    ))
    
    # Add vectors for each test case
    colors = ['purple', 'orange', 'cyan', 'yellow', 'pink']
    for i, result in enumerate(results):
        # Position-based vector
        fig.add_trace(go.Scatter3d(
            x=[0, result['pos_vector'][0]],
            y=[0, result['pos_vector'][1]],
            z=[0, result['pos_vector'][2]],
            line=dict(color=colors[i], width=4),
            name=f'Time {result["time"]}: Position Vector'
        ))
        
        # Sensor-based vector
        fig.add_trace(go.Scatter3d(
            x=[0, result['sensor_vector'][0]],
            y=[0, result['sensor_vector'][1]],
            z=[0, result['sensor_vector'][2]],
            line=dict(color=colors[i], width=4, dash='dash'),
            name=f'Time {result["time"]}: Sensor Vector'
        ))
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title="East",
            yaxis_title="North",
            zaxis_title="Up",
            aspectmode='cube'
        ),
        title="Position vs Sensor Vector Comparison"
    )
    
    fig.show()

def debug_transformation_chain(azimuth, elevation, yaw, pitch, roll):
    """
    Debug the complete transformation chain from sensor frame to ENU.
    Shows the vector at each step of the transformation.
    """
    print(f"\n=== Debugging Transformation Chain ===")
    print(f"Input angles (degrees):")
    print(f"  Azimuth: {azimuth}")
    print(f"  Elevation: {elevation}")
    print(f"  Aircraft yaw: {yaw}")
    print(f"  Aircraft pitch: {pitch}")
    print(f"  Aircraft roll: {roll}")
    
    # 1. Initial vector in sensor frame
    az_rad = np.radians(azimuth)
    el_rad = np.radians(elevation)
    sensor_vec = np.array([
        np.cos(el_rad) * np.cos(az_rad),
        np.cos(el_rad) * np.sin(az_rad),
        np.sin(el_rad)
    ])
    print(f"\n1. Sensor frame vector:")
    print(f"   {sensor_vec}")
    
    # 2. Convert to aircraft body frame
    R_sensor_to_body = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])
    body_vec = R_sensor_to_body @ sensor_vec
    print(f"\n2. Aircraft body frame vector (after Z-flip):")
    print(f"   {body_vec}")
    
    # 3. Individual aircraft rotations
    # Convert angles to radians
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)
    
    # Roll
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll_rad), np.sin(roll_rad)],
        [0, -np.sin(roll_rad), np.cos(roll_rad)]
    ])
    vec_after_roll = R_roll @ body_vec
    print(f"\n3a. After roll rotation:")
    print(f"    {vec_after_roll}")
    
    # Pitch
    R_pitch = np.array([
        [np.cos(pitch_rad), 0, -np.sin(pitch_rad)],
        [0, 1, 0],
        [np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])
    vec_after_pitch = R_pitch @ vec_after_roll
    print(f"\n3b. After pitch rotation:")
    print(f"    {vec_after_pitch}")
    
    # Yaw
    R_yaw = np.array([
        [np.cos(yaw_rad), np.sin(yaw_rad), 0],
        [-np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])
    vec_after_yaw = R_yaw @ vec_after_pitch
    print(f"\n3c. After yaw rotation (in NED frame):")
    print(f"    {vec_after_yaw}")
    
    # 4. Convert from NED to ENU
    R_ned_to_enu = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1]
    ])
    final_vec = R_ned_to_enu @ vec_after_yaw
    print(f"\n4. Final ENU vector:")
    print(f"   {final_vec}")
    
    return final_vec

def run_transformation_debug():
    """Run specific test cases with detailed transformation debugging."""
    print("\n=== Transformation Debug Tests ===")
    
    # Test cases that should have known results
    test_cases = [
        {
            'name': 'Forward pointing',
            'azimuth': 0,
            'elevation': 0,
            'yaw': 0,
            'pitch': 0,
            'roll': 0,
            'expected': np.array([0, 1, 0])  # Should point North in ENU
        },
        {
            'name': 'Up pointing',
            'azimuth': 0,
            'elevation': 90,
            'yaw': 0,
            'pitch': 0,
            'roll': 0,
            'expected': np.array([0, 0, 1])  # Should point Up in ENU
        },
        {
            'name': 'Right pointing',
            'azimuth': 90,
            'elevation': 0,
            'yaw': 0,
            'pitch': 0,
            'roll': 0,
            'expected': np.array([1, 0, 0])  # Should point East in ENU
        }
    ]
    
    for test in test_cases:
        print(f"\n=== Test: {test['name']} ===")
        result = debug_transformation_chain(
            test['azimuth'],
            test['elevation'],
            test['yaw'],
            test['pitch'],
            test['roll']
        )
        
        print(f"\nResult vs Expected:")
        print(f"Result:   {result}")
        print(f"Expected: {test['expected']}")
        angle = np.arccos(np.dot(result, test['expected']) / 
                         (np.linalg.norm(result) * np.linalg.norm(test['expected'])))
        print(f"Angle error: {np.degrees(angle):.2f}°")

def run_edge_case_tests():
    """Test edge cases for angles including negative angles and angles > 360°"""
    print("\n=== Edge Case Tests ===")
    
    test_cases = [
        # Equivalent angles for azimuth
        {
            'name': 'Azimuth 370° (equivalent to 10°)',
            'azimuth': 370,
            'elevation': 0,
            'yaw': 0,
            'pitch': 0,
            'roll': 0,
            'expected': sensor_to_enu_vector(10, 0, 0, 0, 0)
        },
        {
            'name': 'Azimuth -30° (equivalent to 330°)',
            'azimuth': -30,
            'elevation': 0,
            'yaw': 0,
            'pitch': 0,
            'roll': 0,
            'expected': sensor_to_enu_vector(330, 0, 0, 0, 0)
        },
        # Equivalent angles for elevation
        {
            'name': 'Elevation 100° (high look)',
            'azimuth': 0,
            'elevation': 100,
            'yaw': 0,
            'pitch': 0,
            'roll': 0,
            'expected': sensor_to_enu_vector(0, 80, 0, 0, 0)  # Should be equivalent to elevation 80°
        },
        {
            'name': 'Elevation -45° (looking down)',
            'azimuth': 0,
            'elevation': -45,
            'yaw': 0,
            'pitch': 0,
            'roll': 0,
            'expected': None  # Will compare with theoretical calculation
        },
        # Combined edge cases
        {
            'name': 'All negative angles',
            'azimuth': -45,
            'elevation': -30,
            'yaw': -90,
            'pitch': -10,
            'roll': -20,
            'expected': None  # Will compare with positive equivalent
        }
    ]
    
    for test in test_cases:
        print(f"\n=== Test: {test['name']} ===")
        result = debug_transformation_chain(
            test['azimuth'],
            test['elevation'],
            test['yaw'],
            test['pitch'],
            test['roll']
        )
        
        if test['expected'] is not None:
            print(f"\nResult vs Expected:")
            print(f"Result:   {result}")
            print(f"Expected: {test['expected']}")
            angle = np.arccos(np.clip(np.dot(result, test['expected']), -1.0, 1.0))
            print(f"Angle error: {np.degrees(angle):.2f}°")

def run_aircraft_extreme_orientation_tests():
    """Test extreme aircraft orientations and combinations"""
    print("\n=== Aircraft Extreme Orientation Tests ===")
    
    test_cases = [
        # Extreme pitch cases
        {
            'name': 'Nose up 90°',
            'azimuth': 0,
            'elevation': 0,
            'yaw': 0,
            'pitch': 90,
            'roll': 0,
            'expected': np.array([0, 0, 1])  # Should point straight up
        },
        {
            'name': 'Nose down 90°',
            'azimuth': 0,
            'elevation': 0,
            'yaw': 0,
            'pitch': -90,
            'roll': 0,
            'expected': np.array([0, 0, -1])  # Should point straight down
        },
        # Extreme roll cases
        {
            'name': 'Inverted flight',
            'azimuth': 0,
            'elevation': 0,
            'yaw': 0,
            'pitch': 0,
            'roll': 180,
            'expected': np.array([0, 1, 0])  # Should still point North but inverted
        },
        # Combined extreme orientations
        {
            'name': 'Complex maneuver 1',
            'azimuth': 45,
            'elevation': 30,
            'yaw': 180,
            'pitch': 45,
            'roll': 90,
            'expected': None  # Will verify magnitude and general direction
        },
        {
            'name': 'Complex maneuver 2',
            'azimuth': -30,
            'elevation': 45,
            'yaw': -90,
            'pitch': 30,
            'roll': -45,
            'expected': None  # Will verify magnitude and general direction
        },
        # Test for gimbal lock conditions
        {
            'name': 'Near gimbal lock (pitch 89.9°)',
            'azimuth': 45,
            'elevation': 0,
            'yaw': 45,
            'pitch': 89.9,
            'roll': 0,
            'expected': None  # Will verify stability near gimbal lock
        }
    ]
    
    for test in test_cases:
        print(f"\n=== Test: {test['name']} ===")
        result = debug_transformation_chain(
            test['azimuth'],
            test['elevation'],
            test['yaw'],
            test['pitch'],
            test['roll']
        )
        
        print(f"\nResult vector: {result}")
        print(f"Vector magnitude: {np.linalg.norm(result):.6f}")
        
        if test['expected'] is not None:
            print(f"Expected vector: {test['expected']}")
            angle = np.arccos(np.clip(np.dot(result, test['expected']), -1.0, 1.0))
            print(f"Angle error: {np.degrees(angle):.2f}°")
        
        # Additional checks for all cases
        print(f"Vector components (E,N,U): ({result[0]:.3f}, {result[1]:.3f}, {result[2]:.3f})")
        
        # Check for unit vector
        magnitude_error = abs(1.0 - np.linalg.norm(result))
        print(f"Magnitude error: {magnitude_error:.6f}")
        assert magnitude_error < 1e-10, "Vector is not unit length!"

if __name__ == "__main__":
    print("Running edge case tests...")
    #run_edge_case_tests()
    
    print("\nRunning aircraft extreme orientation tests...")
    run_aircraft_extreme_orientation_tests()
    
    # Run the original tests
    print("\nRunning basic vector tests...")
    #run_basic_tests()
    
    print("\nRunning aircraft orientation tests...")
    #run_aircraft_orientation_tests()
    
    