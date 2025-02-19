import click
from pathlib import Path

@click.group()
def cli():
    """Electronic Warfare Analysis Tools.
    
    This CLI provides tools for analyzing electronic warfare sensor data,
    including trajectory visualization and target tracking.
    """
    pass

@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--emitter', default=1, help='Emitter ID to plot')
@click.option('--ray-length', default=50000, help='Length of sensor direction rays in meters')
@click.option('--ray-interval', default=1000, help='Interval between plotted rays')
def plot(data_file, emitter, ray_length, ray_interval):
    """Plot sensor and emitter trajectories with direction rays.
    
    DATA_FILE: Path to the input data file (.ipc format)
    """
    import polars as pl
    from plot_trajectories import compute_global_vectors_enu, plot_trajectories
    from parse_data import load_window

    click.echo(f"Loading data from {data_file}")
    df_pl = load_window(Path(data_file))

    # Filter for specific emitter and select required columns
    # TODO: clean emitter column and convert id to int
    _emitter = f'Emitter{emitter}'
    df_pl = df_pl.filter(pl.col("emitter") == _emitter).drop("emitter")
    columns = [
        'arrival_time', 'azimuth', 'elevation',
        'sensor_lat', 'sensor_lon', 'sensor_alt',
        'sensor_yaw', 'sensor_pitch', 'sensor_roll',
        'emitter_lat', 'emitter_lon', 'emitter_alt'
    ]
    df_pl = df_pl.select(columns)
    df = df_pl.to_pandas()
    df.set_index("arrival_time", inplace=True)

    click.echo("Computing global vectors...")
    df = compute_global_vectors_enu(df)
    
    click.echo("Generating plot...")
    plot_trajectories(df, ray_length=ray_length, ray_interval=ray_interval)

@cli.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--min-dt', default=1.0, help='Minimum time difference between measurements (seconds)')
@click.option('--noise-std', default=5.0, help='Measurement noise standard deviation (meters)')
@click.option('--skip-clustering', is_flag=True, help='Skip clustering step (use existing labels)')
@click.option('--n-clusters', default=150, help='Number of clusters for GMM (if clustering)')
@click.option('--cluster-features', default=['amplitude', 'frequency', 'pulse_width'],
              multiple=True, help='Features to use for clustering (if clustering)')
@click.option('--plot/--no-plot', default=False, help='Generate tracking result plots (only with --skip-clustering)')
@click.option('--plot-dir', type=click.Path(), default='.',
              help='Directory to save result plots (if plotting)')
def track(data_file, output_file, min_dt, noise_std, skip_clustering, 
          n_clusters, cluster_features, plot, plot_dir):
    """Process and track emitters using Kalman filtering.
    
    DATA_FILE: Path to the input data file (.ipc format)
    OUTPUT_FILE: Path where to save the processed data (.ipc format)
    """
    from parse_data import load_window
    from cluster_eval import cluster_unlabeled_data
    from kalman import KalmanFilter
    import pandas as pd
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt
    from geometry import lla_to_enu

    click.echo(f"Loading data from {data_file}")
    df = load_window(Path(data_file))
    
    if skip_clustering:
        click.echo("Using existing labels...")
        if 'emitter' not in df.columns:
            raise click.ClickException("No 'emitter' column found in data. Cannot skip clustering.")
        clustered_df = df.to_pandas()
        clustered_df = clustered_df.rename(columns={"emitter": "label"})
    else:
        # Perform clustering
        click.echo("Performing GMM clustering...")
        clustered_df = cluster_unlabeled_data(
            df,
            list(cluster_features),
            n_components=n_clusters,
            covariance_type="full",
            reg_covar=1e-6,
            n_init=2,
            max_iter=50,
            random_state=0,
        )
    
    # Select required columns for Kalman filtering
    columns_to_keep = [
        'arrival_time', 'azimuth', 'elevation', 'amplitude',
        'sensor_lat', 'sensor_lon', 'sensor_alt', 
        'sensor_yaw', 'sensor_pitch', 'sensor_roll',
        'label'
    ]
    
    # Add true positions if available (for plotting)
    if skip_clustering:
        columns_to_keep.extend(['emitter_lat', 'emitter_lon', 'emitter_alt'])
        
    df_kalman = clustered_df[columns_to_keep]
    df_kalman.set_index('arrival_time', inplace=True)
    
    click.echo("Initializing Kalman filter...")
    kf = KalmanFilter(
        min_measurement_dt=min_dt,
        meas_noise_std=noise_std
    )
    
    # Setup coordinate system using first sensor position as reference
    first_row = df_kalman.iloc[0]
    kf.setup_coordinates(
        first_row['sensor_lat'],
        first_row['sensor_lon'],
        first_row['sensor_alt']
    )
    
    click.echo("Processing measurements with Kalman filter...")
    results_dict = kf.process_multiple_emitters(df_kalman)
    
    if not results_dict:
        click.echo("No results to save!")
        return

    final_results = pd.concat(results_dict.values())
    final_results.sort_index(inplace=True)
    
    click.echo("Saving results...")
    table = pa.Table.from_pandas(final_results)
    with pa.OSFile(str(output_file), "wb") as sink:
        with ipc.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)
    click.echo(f"Results saved to {output_file}")

    if plot:
        plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        click.echo("\nGenerating plots...")
        all_errors = []
        emitter_stats = {}

        for emitter, results_df in results_dict.items():
            try:
                # Get true position for this emitter
                emitter_data = df_kalman[df_kalman['label'] == emitter].iloc[0]
                true_enu = np.array(lla_to_enu(
                    emitter_data['emitter_lat'],
                    emitter_data['emitter_lon'],
                    emitter_data['emitter_alt'],
                    first_row['sensor_lat'],
                    first_row['sensor_lon'],
                    first_row['sensor_alt']
                ))

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

                # Calculate indices for middle 70%
                n_measurements = len(results_df)
                start_idx = int(n_measurements * 0.25)  # Skip first 25%
                end_idx = int(n_measurements * 0.8)     # Skip last 20%
                
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
                           label=f'RMS: {middle_rms:.2f}m')
                plt.ylabel('Error (m)')
                plt.xlabel('Time')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                
                # Save figure
                output_file = plot_dir / f'kalman_results_{emitter}.png'
                fig.savefig(output_file, dpi=300, bbox_inches='tight')
                click.echo(f"Saved plot to {output_file}")
                plt.close()

                # Calculate and store statistics
                rms_error = np.sqrt(np.mean(np.array(errors)**2))
                mean_error = np.mean(errors)
                median_error = np.median(errors)
                std_error = np.std(errors)
                
                click.echo(f"\nStatistics for {emitter}:")
                click.echo(f"RMS Error: {rms_error:.2f} meters")
                click.echo(f"Mean Error: {mean_error:.2f} meters")
                click.echo(f"Median Error: {median_error:.2f} meters")
                click.echo(f"Std Dev: {std_error:.2f} meters")
                
            except Exception as e:
                click.echo(f"Error creating plot for {emitter}: {e}", err=True)

        # Print overall statistics
        if all_errors:
            all_errors = np.array(all_errors)
            click.echo("\n=== OVERALL STATISTICS ===")
            click.echo(f"Overall RMS Error: {np.sqrt(np.mean(all_errors**2)):.2f} meters")
            click.echo(f"Overall Mean Error: {np.mean(all_errors):.2f} meters")
            click.echo(f"Overall Median Error: {np.median(all_errors):.2f} meters")
            click.echo(f"Overall Std Dev: {np.std(all_errors):.2f} meters")

        click.echo("\nPlotting complete!")

if __name__ == '__main__':
    cli()
