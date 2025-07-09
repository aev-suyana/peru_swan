"""
DATA PREPARATION STEP 0 - INITIAL PROCESSING
============================================

This script loads and processes raw wave data (SWAN and WAVERYS) and creates
initial processed files for the reference port approach.

Converted to use centralized configuration from config.py

Author: Wave Analysis Team
Date: 2024
"""

# ============================================================================
# IMPORTS AND CONFIGURATION
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import os

# Import centralized configuration
try:
    from config import config, get_input_files, get_output_files
    print("✅ Using centralized configuration")
except ImportError:
    print("❌ Error: Cannot import config. Make sure config.py is in the same directory.")
    print("Please run this script from the project root directory.")
    exit(1)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def max_consecutive_above_threshold(series, threshold):
    """Calculate maximum consecutive periods above threshold"""
    above_threshold = (series > threshold).astype(int)
    
    if above_threshold.sum() == 0:
        return 0
    
    # Find consecutive sequences
    consecutive_counts = []
    current_count = 0
    
    for value in above_threshold:
        if value == 1:
            current_count += 1
        else:
            if current_count > 0:
                consecutive_counts.append(current_count)
            current_count = 0
    
    # Don't forget the last sequence if it ends with 1s
    if current_count > 0:
        consecutive_counts.append(current_count)
    
    return max(consecutive_counts) if consecutive_counts else 0

def load_swan_csv_data(csv_path):
    """
    Load SWAN wave data from CSV file
    
    Parameters:
    csv_path: Path to the CSV file with columns ['date', 'swh']
    
    Returns:
    df_swan: DataFrame with processed wave data
    """
    print("Loading SWAN wave data from CSV...")
    
    try:
        # Load the CSV
        df_swan = pd.read_csv(csv_path)
        df_swan.rename(columns={'fecha': 'date', 'sea_wave_height': 'swh'}, inplace=True)
        
        # Check required columns
        required_cols = ['date', 'swh']
        missing_cols = [col for col in required_cols if col not in df_swan.columns]
        
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            print(f"Available columns: {list(df_swan.columns)}")
            return pd.DataFrame()
        
        # Process datetime
        df_swan['datetime'] = pd.to_datetime(df_swan['date'])
        df_swan['date'] = df_swan['datetime'].dt.date
        
        # Remove any NaN values
        initial_length = len(df_swan)
        df_swan = df_swan.dropna(subset=['swh'])
        final_length = len(df_swan)
        
        if initial_length != final_length:
            print(f"Removed {initial_length - final_length} rows with NaN SWH values")
        
        # Sort by datetime
        df_swan = df_swan.sort_values('datetime').reset_index(drop=True)
        
        print(f"Loaded {len(df_swan):,} wave data records")
        print(f"Date range: {df_swan['date'].min()} to {df_swan['date'].max()}")
        print(f"SWH range: {df_swan['swh'].min():.3f} to {df_swan['swh'].max():.3f} m")
        
        return df_swan
        
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return pd.DataFrame()

def filter_waverys_data_by_reference_port(waverys_dir, reference_port, closed_ports, year_range=(2023, 2025)):
    """
    Load waverys data filtered by reference port name only
    
    Parameters:
    waverys_dir: Path to waverys data directory
    reference_port: Name of the reference port to filter by
    closed_ports: DataFrame with port closure data
    year_range: Tuple of (start_year, end_year) for data loading
    
    Returns:
    swh_processed: Filtered DataFrame with only the reference port data
    """
    
    print(f"Loading WAVERYS data for reference port: {reference_port}")
    
    # Load and filter the data
    df_list = []
    
    for year in range(year_range[0], year_range[1]):
        print(f'Processing WAVERYS {year}...')
        
        try:
            waverys_file = os.path.join(waverys_dir, f'waverys_all_{year}.csv')
            if not os.path.exists(waverys_file):
                print(f"  Warning: File not found: {waverys_file}")
                continue
                
            swh_processed = pd.read_csv(waverys_file)
            
            # Standardize port names
            swh_processed['port_name'] = np.where(swh_processed['port_name'].str.contains('PIMENTEL'), 
                                                 'PUERTO_DE_PIMENTEL', swh_processed['port_name'])
            
            # Filter to only include the reference port
            swh_processed = swh_processed[swh_processed['port_name'] == reference_port].copy()
            
            if len(swh_processed) == 0:
                print(f"  No data for {reference_port} in {year}")
                continue
            
            # Process datetime and create additional columns
            swh_processed['datetime'] = pd.to_datetime(swh_processed['datetime'])
            swh_processed['daymon'] = swh_processed['datetime'].dt.strftime('%d-%m')
            swh_processed['year'] = year
            
            print(f"  Loaded {len(swh_processed):,} records for {reference_port}")
            df_list.append(swh_processed)
            
        except Exception as e:
            print(f"  Error processing {year}: {e}")
            continue
    
    if not df_list:
        print(f"No WAVERYS data loaded for {reference_port}!")
        return pd.DataFrame()
    
    # Concatenate all years
    swh_processed = pd.concat(df_list, ignore_index=True)
    
    # Calculate climatologies and anomalies
    print("Calculating climatologies and anomalies...")
    swh_processed['clima_swh'] = swh_processed.groupby('daymon')['swh'].transform('mean')
    swh_processed['clima_swe'] = swh_processed.groupby('daymon')['swell_height'].transform('mean')
    swh_processed['anom_swh'] = swh_processed['swh'] - swh_processed['clima_swh']
    swh_processed['anom_swe'] = swh_processed['swell_height'] - swh_processed['clima_swe']
    swh_processed['date'] = swh_processed['datetime'].dt.date
    
    # Merge with closed ports data
    print("Merging with port closure data...")
    swh_processed = swh_processed.merge(closed_ports, how='left',
                                       on=['port_name', 'date', 'year'])
    
    # Sort data
    swh_processed.sort_values(by='date', inplace=True)
    
    # Summary
    print(f"\nWAVERYS dataset summary:")
    print(f"Total records: {len(swh_processed):,}")
    print(f"Date range: {swh_processed['date'].min()} to {swh_processed['date'].max()}")
    print(f"Port: {reference_port}")
    
    return swh_processed

def process_swan_data(df_swan_hourly, reference_port):
    """
    Process SWAN hourly data and create daily aggregates with enhanced features
    
    Parameters:
    df_swan_hourly: Hourly SWAN data
    reference_port: Name of reference port
    
    Returns:
    df_swan_daily_enhanced: Daily aggregated data with enhanced features
    """
    
    print(f"Processing SWAN data for {reference_port}...")
    
    # Filter out 2025 data
    df_swan_hourly = df_swan_hourly[df_swan_hourly['datetime'].dt.year < 2025].reset_index(drop=True)
    
    # Sort by datetime
    df_swan_hourly.sort_values(by='datetime', inplace=True)
    
    # Calculate daily climatology
    df_swan_hourly['daymon'] = df_swan_hourly['datetime'].dt.strftime('%m-%d')
    df_swan_hourly['clima_swh'] = df_swan_hourly.groupby('daymon')['swh'].transform('mean')
    df_swan_hourly['anom_swh'] = df_swan_hourly['swh'] - df_swan_hourly['clima_swh']
    
    # Calculate threshold from pre-2024 data
    threshold_2023 = df_swan_hourly[df_swan_hourly['datetime'].dt.year < 2024]['swh'].quantile(0.60)
    print(f"60th percentile threshold from 2023: {threshold_2023:.2f}m")
    
    # Create daily aggregates with enhanced features
    df_swan_daily_enhanced = df_swan_hourly.groupby(df_swan_hourly['datetime'].dt.date).agg(
        # Original features
        swh_mean=('swh', 'mean'),
        swh_max=('swh', 'max'),
        swh_min=('swh', 'min'),
        swh_median=('swh', 'median'),
        swh_p80=('swh', lambda x: np.percentile(x, 80)),
        swh_p25=('swh', lambda x: np.percentile(x, 25)),
        swh_p75=('swh', lambda x: np.percentile(x, 75)),
        swh_p60=('swh', lambda x: np.percentile(x, 60)),
        swh_sd=('swh', 'std'),
        clima_swh_mean=('clima_swh', 'mean'),
        anom_swh_mean=('anom_swh', 'mean'),
        anom_swh_max=('anom_swh', 'max'),
        anom_swh_min=('anom_swh', 'min'),
        anom_swh_median=('anom_swh', 'median'),
        anom_swh_p25=('anom_swh', lambda x: np.percentile(x, 25)),
        anom_swh_p75=('anom_swh', lambda x: np.percentile(x, 75)),
        anom_swh_p80=('anom_swh', lambda x: np.percentile(x, 80)),
        anom_swh_sd=('anom_swh', 'std'),
        # Duration above threshold features
        hours_above_p60_2023=('swh', lambda x: (x > threshold_2023).sum()),
        pct_day_above_p60=('swh', lambda x: (x > threshold_2023).mean() * 100),
        max_consecutive_above_p60=('swh', lambda x: max_consecutive_above_threshold(x, threshold_2023)),
        total_obs=('swh', 'count'),
    ).reset_index()
    
    # Rename date column
    df_swan_daily_enhanced.rename(columns={'datetime': 'date'}, inplace=True)
    
    # Calculate derived features
    df_swan_daily_enhanced['swh_cv'] = df_swan_daily_enhanced['swh_sd'] / df_swan_daily_enhanced['swh_mean']
    df_swan_daily_enhanced['swh_range'] = df_swan_daily_enhanced['swh_max'] - df_swan_daily_enhanced['swh_min']
    df_swan_daily_enhanced['swh_iqr'] = df_swan_daily_enhanced['swh_p75'] - df_swan_daily_enhanced['swh_p25']
    df_swan_daily_enhanced['anom_cv'] = df_swan_daily_enhanced['anom_swh_sd'] / df_swan_daily_enhanced['anom_swh_mean']
    df_swan_daily_enhanced['anom_range'] = df_swan_daily_enhanced['anom_swh_max'] - df_swan_daily_enhanced['anom_swh_min']
    df_swan_daily_enhanced['anom_iqr'] = df_swan_daily_enhanced['anom_swh_p75'] - df_swan_daily_enhanced['anom_swh_p25']
    df_swan_daily_enhanced['duration_intensity_p60'] = (
        df_swan_daily_enhanced['hours_above_p60_2023'] * df_swan_daily_enhanced['swh_mean']
    )
    
    # Filter to start from 2018
    df_swan_daily_enhanced = df_swan_daily_enhanced[df_swan_daily_enhanced['date'] >= pd.to_datetime('2018-01-01').date()].reset_index(drop=True)
    
    print(f"✅ SWAN daily processing complete: {df_swan_daily_enhanced.shape}")
    
    return df_swan_daily_enhanced

def process_waverys_data(df_waverys_hourly, closed_ports):
    """
    Process WAVERYS hourly data and create daily aggregates
    
    Parameters:
    df_waverys_hourly: Hourly WAVERYS data
    closed_ports: Port closure data
    
    Returns:
    df_waverys_daily: Daily aggregated WAVERYS data
    """
    
    print("Processing WAVERYS data...")
    
    # Calculate threshold from pre-2024 data
    df_waverys_hourly['date'] = pd.to_datetime(df_waverys_hourly['date'])
    threshold_2023 = df_waverys_hourly[df_waverys_hourly['date'].dt.year < 2024]['swh'].quantile(0.75)
    print(f"75th percentile threshold from 2023: {threshold_2023:.2f}m")
    
    # Create daily aggregation
    df_waverys_daily = df_waverys_hourly.groupby([df_waverys_hourly['date'].dt.date, 'port_name']).agg(
        swh_max=('swh', 'max'),
        swh_min=('swh', 'min'),
        swh_mean=('swh', 'mean'),
        swh_median=('swh', 'median'),
        swh_p80=('swh', lambda x: np.percentile(x, 80)),
        swh_p75=('swh', lambda x: np.percentile(x, 75)),
        swh_p25=('swh', lambda x: np.percentile(x, 25)),
        swh_sd=('swh', 'std'),
        swh_cv=('swh', lambda x: np.std(x) / np.mean(x) if np.mean(x) != 0 else 0),
        swh_range=('swh', lambda x: np.max(x) - np.min(x)),
        clima_swh_mean=('clima_swh', 'mean'),
        anom_swh_mean=('anom_swh', 'mean'),
        anom_swh_max=('anom_swh', 'max'),
        anom_swh_min=('anom_swh', 'min'),
        anom_swh_sd=('anom_swh', 'std'),
        anom_swh_p80=('anom_swh', lambda x: np.percentile(x, 80)),
        anom_swh_cv=('anom_swh', lambda x: np.std(x) / np.mean(x) if np.mean(x) != 0 else 0),
        anom_swh_range=('anom_swh', lambda x: np.max(x) - np.min(x)),
        duracion=('duracion', 'max'),
        clima_swe_mean=('clima_swe', 'mean'),
        anom_swe_mean=('anom_swe', 'mean'),
        anom_swe_max=('anom_swe', 'max'),
        anom_swe_min=('anom_swe', 'min'),
        anom_swe_sd=('anom_swe', 'std'),
        anom_swe_p80=('anom_swe', lambda x: np.percentile(x, 80)),
        anom_swe_cv=('anom_swe', lambda x: np.std(x) / np.mean(x) if np.mean(x) != 0 else 0),
        anom_swe_range=('anom_swe', lambda x: np.max(x) - np.min(x)),
        # Duration above threshold features
        hours_above_p60_2023=('swh', lambda x: (x > threshold_2023).sum()),
        pct_day_above_p60=('swh', lambda x: (x > threshold_2023).mean() * 100),
        max_consecutive_above_p60=('swh', lambda x: max_consecutive_above_threshold(x, threshold_2023)),
        total_obs=('swh', 'count'),
        year=('year', 'max'),
        latitude=('latitude', 'max'),
        longitude=('longitude', 'max'),
        nearest_latitude=('nearest_latitude', 'max'),
        nearest_longitude=('nearest_longitude', 'max'),
    ).reset_index()
    
    # Filter to 2018 and later
    df_waverys_daily = df_waverys_daily[df_waverys_daily['year'] >= 2018].reset_index(drop=True)
    
    # Ensure date column is datetime format
    df_waverys_daily['date'] = pd.to_datetime(df_waverys_daily['date'])
    df_waverys_daily['date'] = df_waverys_daily['date'].dt.date
    
    # Ensure duration is numeric
    df_waverys_daily['duracion'] = pd.to_numeric(df_waverys_daily['duracion'], errors='coerce')
    
    # Calculate additional derived features
    df_waverys_daily['duration_intensity_p60'] = (
        df_waverys_daily['hours_above_p60_2023'] * df_waverys_daily['swh_mean']
    )
    df_waverys_daily['swh_iqr'] = df_waverys_daily['swh_p75'] - df_waverys_daily['swh_p25']
    
    # Create event dummy from duration column
    print("Creating event indicators from duration data...")
    events_df = df_waverys_daily[df_waverys_daily['duracion'] > 0].copy()
    
    # Create all event date ranges at once
    event_dates = []
    for _, row in events_df.iterrows():
        start_date = row['date']
        end_date = start_date + pd.Timedelta(days=row['duracion'])
        date_range = pd.date_range(start=start_date, end=end_date - pd.Timedelta(days=1), freq='D')
        
        for date in date_range:
            event_dates.append({'port_name': row['port_name'], 'date': date})
    
    # Convert to DataFrame and merge
    if event_dates:
        event_dates_df = pd.DataFrame(event_dates)
        event_dates_df['event_dummy_1'] = 1
        df_waverys_daily['date'] = pd.to_datetime(df_waverys_daily['date'])
        event_dates_df['date'] = pd.to_datetime(event_dates_df['date'])
        df_waverys_daily['date'] = df_waverys_daily['date'].dt.date
        event_dates_df['date'] = event_dates_df['date'].dt.date
        
        # Merge back to main data
        df_waverys_daily = df_waverys_daily.merge(
            event_dates_df, on=['port_name', 'date'], how='left'
        )
    
    df_waverys_daily['event_dummy_1'] = df_waverys_daily['event_dummy_1'].fillna(0).astype(int)
    
    # Print event statistics
    event_count = sum(df_waverys_daily['event_dummy_1'] == 1)
    print(f"Total rows marked as events: {event_count}")
    print(f"Percentage of rows marked as events: {event_count/len(df_waverys_daily)*100:.2f}%")
    
    print(f"✅ WAVERYS daily processing complete: {df_waverys_daily.shape}")
    
    return df_waverys_daily

def main():
    """
    Main execution function for data_prep_0.py
    """
    
    print("🚀 DATA PREPARATION STEP 0 - INITIAL PROCESSING")
    print("="*80)
    print(f"Run: {config.RUN_PATH}")
    print(f"Reference port: {config.reference_port}")
    print(f"Coordinates: ({config.reference_latitude}, {config.reference_longitude})")
    print("="*80)
    
    # Get file paths
    input_files = get_input_files()
    
    # ========================================================================
    # STEP 1: Load port closure data
    # ========================================================================
    
    print("\n📊 STEP 1: Loading port closure data...")
    
    closed_ports_file = input_files['closed_ports']
    if not os.path.exists(closed_ports_file):
        print(f"❌ Error: Closed ports file not found: {closed_ports_file}")
        print("Please place the file in data/raw/ directory")
        return
    
    try:
        closed_ports = pd.read_csv(closed_ports_file)
        
        # Standardize port names
        closed_ports['port_name'] = np.where(closed_ports['port_name'].str.contains('PIMENTEL'), 
                                           'PUERTO_DE_PIMENTEL', closed_ports['port_name'])
        closed_ports['capitania'] = np.where(closed_ports['port_name'].str.contains('PIMENTEL'), 
                                            'PUERTO_DE_PIMENTEL', closed_ports['capitania'])
        
        closed_ports['date'] = pd.to_datetime(closed_ports['date']).dt.date
        closed_ports.sort_values(by=['port_name', 'date'], inplace=True)
        closed_ports = closed_ports.reset_index(drop=True)
        
        print(f"✅ Loaded {len(closed_ports)} port closure records")
        print(f"Years: {sorted(closed_ports['year'].unique())}")
        print(f"Ports: {len(closed_ports['port_name'].unique())} unique ports")
        
    except Exception as e:
        print(f"❌ Error loading closed ports data: {e}")
        return
    
    # ========================================================================
    # STEP 2: Load SWAN wave data
    # ========================================================================
    
    print("\n🌊 STEP 2: Loading SWAN wave data...")
    
    wave_csv_file = input_files['wave_height_csv']
    if not os.path.exists(wave_csv_file):
        print(f"❌ Error: Wave height CSV not found: {wave_csv_file}")
        print("Please place the file in data/raw/ directory")
        return
    
    df_swan_hourly = load_swan_csv_data(wave_csv_file)
    if df_swan_hourly.empty:
        print("❌ Failed to load SWAN data")
        return
    
    # Add event data to SWAN hourly data
    df_swan_hourly['date'] = pd.to_datetime(df_swan_hourly['datetime']).dt.date
    
    # Create daily events summary
    daily_events = closed_ports.groupby('date')['event_dummy_1'].max().reset_index() if 'event_dummy_1' in closed_ports.columns else pd.DataFrame()
    
    if not daily_events.empty:
        daily_events['date'] = pd.to_datetime(daily_events['date']).dt.date
        daily_events['event_dummy_1'] = daily_events['event_dummy_1'].astype(int)
        
        # Merge with SWAN data
        df_swan_hourly = df_swan_hourly.merge(daily_events, how='left', on=['date'])
        df_swan_hourly['event_dummy_1'] = df_swan_hourly['event_dummy_1'].fillna(0).astype(int)
        
        event_pct = df_swan_hourly['event_dummy_1'].mean() * 100
        print(f"✅ Added event data: {event_pct:.1f}% of days have port closures")
    
    # ========================================================================
    # STEP 3: Process SWAN data to daily aggregates
    # ========================================================================
    
    print("\n📈 STEP 3: Processing SWAN data to daily aggregates...")
    
    df_swan_daily_enhanced = process_swan_data(df_swan_hourly, config.reference_port)
    
    # ========================================================================
    # STEP 4: Load and process WAVERYS data
    # ========================================================================
    
    print("\n📡 STEP 4: Loading and processing WAVERYS data...")
    
    waverys_dir = os.path.join(config.RAW_DATA_DIR, 'waverys')
    if os.path.exists(waverys_dir):
        df_waverys_hourly = filter_waverys_data_by_reference_port(
            waverys_dir, config.reference_port, closed_ports, year_range=(1993, 2025)
        )
        
        if not df_waverys_hourly.empty:
            df_waverys_daily = process_waverys_data(df_waverys_hourly, closed_ports)
        else:
            print("⚠️ No WAVERYS data available, creating empty dataset")
            df_waverys_daily = pd.DataFrame()
    else:
        print(f"⚠️ WAVERYS directory not found: {waverys_dir}")
        print("Creating empty WAVERYS datasets")
        df_waverys_hourly = pd.DataFrame()
        df_waverys_daily = pd.DataFrame()
    
    # ========================================================================
    # STEP 5: Save processed files
    # ========================================================================
    
    print("\n💾 STEP 5: Saving processed files...")
    
    # Create output directory
    output_dir = config.run_output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Save hourly files
    df_swan_hourly_filtered = df_swan_hourly[df_swan_hourly['datetime'].dt.year < 2025].reset_index(drop=True)
    df_swan_hourly_filtered.to_csv(os.path.join(output_dir, 'df_swan_hourly.csv'), index=False)
    print(f"✅ Saved: df_swan_hourly.csv ({df_swan_hourly_filtered.shape})")
    
    df_swan_daily_filtered = df_swan_daily_enhanced[df_swan_daily_enhanced['date'] >= pd.to_datetime('2018-01-01').date()].reset_index(drop=True)
    df_swan_daily_filtered.to_csv(os.path.join(output_dir, 'df_swan_daily_enhanced.csv'), index=False)
    print(f"✅ Saved: df_swan_daily_enhanced.csv ({df_swan_daily_filtered.shape})")
    
    if not df_waverys_hourly.empty:
        df_waverys_hourly.to_csv(os.path.join(output_dir, 'df_waverys_hourly.csv'), index=False)
        print(f"✅ Saved: df_waverys_hourly.csv ({df_waverys_hourly.shape})")
    
    if not df_waverys_daily.empty:
        df_waverys_daily.to_csv(os.path.join(output_dir, 'df_waverys_daily.csv'), index=False)
        print(f"✅ Saved: df_waverys_daily.csv ({df_waverys_daily.shape})")
    
    # ========================================================================
    # STEP 6: Create renamed versions for pipeline
    # ========================================================================
    
    print("\n🏷️ STEP 6: Creating renamed versions for pipeline...")
    
    # SWAN data with _sw suffix
    df_swan_daily_new_colnames = df_swan_daily_filtered.copy()
    cols_to_rename = [col for col in df_swan_daily_new_colnames.columns if col not in ['date', 'event_dummy_1']]
    rename_dict = {col: f"{col}_sw" for col in cols_to_rename}
    df_swan_daily_new_colnames.rename(columns=rename_dict, inplace=True)
    df_swan_daily_new_colnames.to_csv(os.path.join(output_dir, 'df_swan_daily_new_colnames.csv'), index=False)
    print(f"✅ Saved: df_swan_daily_new_colnames.csv ({df_swan_daily_new_colnames.shape})")
    
    # WAVERYS data with _wa suffix
    if not df_waverys_daily.empty:
        df_waverys_daily_new_colnames = df_waverys_daily.copy()
        cols_to_rename = [col for col in df_waverys_daily_new_colnames.columns 
                         if col not in ['date', 'event_dummy_1', 'port_name']]
        rename_dict = {col: f"{col}_wa" for col in cols_to_rename}
        df_waverys_daily_new_colnames.rename(columns=rename_dict, inplace=True)
        
        # Drop unnecessary columns
        drop_cols = [col for col in ['latitude_wa', 'longitude_wa', 'nearest_latitude_wa', 
                                   'nearest_longitude_wa', 'year_wa', 'duracion_wa'] 
                    if col in df_waverys_daily_new_colnames.columns]
        df_waverys_daily_new_colnames.drop(columns=drop_cols, inplace=True)
        
        df_waverys_daily_new_colnames.to_csv(os.path.join(output_dir, 'df_waverys_daily_new_colnames.csv'), index=False)
        print(f"✅ Saved: df_waverys_daily_new_colnames.csv ({df_waverys_daily_new_colnames.shape})")