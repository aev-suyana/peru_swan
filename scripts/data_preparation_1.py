"""
DATA PREPARATION STEP 1 - ENHANCED PROCESSING
============================================

This script processes daily wave data for a single reference port approach:
1. Detrend and deseasonalize (single reference point)
2. Enhanced feature engineering (on reference point data)
3. Combine with WAVERYS data

All configuration and paths are sourced from config.py for full repo integration.

Author: Wave Analysis Team (refactored)
Date: 2025
"""

# ============================================================================
# IMPORTS AND CONFIGURATION
# ============================================================================
import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import signal
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold

# Add project root to Python path
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from config import config
    print("✅ Using centralized configuration")
except ImportError as e:
    print(f"❌ Error: Cannot import config: {e}")
    exit(1)

# ============================================================================
# MAIN PROCESSING FUNCTIONS (to be filled in next steps)
# ============================================================================

def load_and_validate_daily_data(swan_daily_path, waverys_daily_path, reference_port):
    print(f"\nSTEP 1: Loading and validating daily data")
    # Load SWAN data
    try:
        if os.path.exists(swan_daily_path):
            df_swan_daily = pd.read_csv(swan_daily_path)
            df_swan_daily['date'] = pd.to_datetime(df_swan_daily['date'])
            print(f"✓ Loaded SWAN daily data: {df_swan_daily.shape}")
        else:
            print(f"✗ SWAN daily file not found: {swan_daily_path}")
            df_swan_daily = pd.DataFrame()
    except Exception as e:
        print(f"✗ Error loading SWAN data: {e}")
        df_swan_daily = pd.DataFrame()
    # Load WAVERYS data
    try:
        if os.path.exists(waverys_daily_path):
            df_waverys_daily = pd.read_csv(waverys_daily_path)
            df_waverys_daily['date'] = pd.to_datetime(df_waverys_daily['date'])
            df_waverys_daily = df_waverys_daily[df_waverys_daily['port_name'] == reference_port].copy()
            print(f"✓ Loaded WAVERYS data for {reference_port}: {df_waverys_daily.shape}")
        else:
            print(f"✗ WAVERYS daily file not found: {waverys_daily_path}")
            df_waverys_daily = pd.DataFrame()
    except Exception as e:
        print(f"✗ Error loading WAVERYS data: {e}")
        df_waverys_daily = pd.DataFrame()
    if not df_swan_daily.empty:
        print(f"SWAN data date range: {df_swan_daily['date'].min().date()} to {df_swan_daily['date'].max().date()}")
    if not df_waverys_daily.empty:
        print(f"WAVERYS data date range: {df_waverys_daily['date'].min().date()} to {df_waverys_daily['date'].max().date()}")
        print(f"WAVERYS ports: {df_waverys_daily['port_name'].unique()}")
    return df_swan_daily, df_waverys_daily

def detrend_and_deseasonalize_reference_point(df_daily, apply_detrending=True, apply_deseasonalizing=True):
    print(f"\nSTEP 2: Detrending and deseasonalizing reference point data")
    print(f"Input data shape: {df_daily.shape}")
    print(f"Apply detrending: {apply_detrending}")
    print(f"Apply deseasonalizing: {apply_deseasonalizing}")
    if not (apply_detrending or apply_deseasonalizing):
        print("No processing requested - returning original data")
        return df_daily
    df_processed = df_daily.copy()
    df_processed['date'] = pd.to_datetime(df_processed['date'])
    df_processed = df_processed.sort_values('date').reset_index(drop=True)
    wave_features = [col for col in df_daily.columns if any(wave_type in col for wave_type in ['swh', 'swe']) and col not in ['date', 'event_dummy_1', 'total_obs_sw', 'port_name', 'year']]
    print(f"Processing {len(wave_features)} wave features...")
    print(f"Example features: {wave_features[:5]}")
    if len(df_processed) < 30:
        print(f"Warning: Insufficient data ({len(df_processed)} days) for processing")
        return df_processed
    if apply_deseasonalizing:
        df_processed['day_of_year'] = df_processed['date'].dt.dayofyear
    valid_features = [f for f in wave_features if f in df_processed.columns]
    if len(valid_features) == 0:
        print("No valid wave features found for processing")
        return df_processed
    print(f"Vectorized processing of {len(valid_features)} features...")
    if apply_deseasonalizing:
        seasonal_avgs = df_processed.groupby('day_of_year')[valid_features].transform('mean')
        feature_means = df_processed[valid_features].mean()
        deseasonalized = df_processed[valid_features] - seasonal_avgs + feature_means
        for feature in valid_features:
            df_processed[f"{feature}_deseasonalized"] = deseasonalized[feature]
    else:
        for feature in valid_features:
            df_processed[f"{feature}_deseasonalized"] = df_processed[feature]
    if apply_detrending:
        deseason_features = [f"{f}_deseasonalized" for f in valid_features]
        deseason_data = df_processed[deseason_features].fillna(method='ffill').fillna(method='bfill')
        for idx, feature in enumerate(deseason_features):
            y = deseason_data[feature].values
            x = np.arange(len(y))
            if np.isnan(y).all():
                continue
            mask = ~np.isnan(y)
            z = np.polyfit(x[mask], y[mask], 1)
            trend = np.polyval(z, x)
            df_processed[feature + '_detrended'] = deseason_data[feature] - trend
    else:
        for feature in valid_features:
            df_processed[f"{feature}_detrended"] = df_processed[f"{feature}_deseasonalized"]
    return df_processed

def create_enhanced_features_reference_point(df_processed, use_processed_features=True):
    print(f"\nSTEP 3: Enhanced feature engineering on reference point data")
    df = df_processed.copy()
    if use_processed_features:
        base_features = [col for col in df.columns if col.endswith('_detrended')]
    else:
        base_features = [col for col in df.columns if col.startswith('swh') or col.startswith('swe')]
    for window in [2, 3, 5, 7, 14]:
        for feature in base_features:
            df[f'{feature}_mean_{window}d'] = df[feature].rolling(window).mean()
            df[f'{feature}_std_{window}d'] = df[feature].rolling(window).std()
            df[f'{feature}_min_{window}d'] = df[feature].rolling(window).min()
            df[f'{feature}_max_{window}d'] = df[feature].rolling(window).max()
    for window in [1, 3, 5, 7, 14]:
        for feature in base_features:
            df[f'{feature}_lag_{window}d'] = df[feature].shift(window)
    print(f"Enhanced features created: {len(df.columns)} total columns")
    return df

def main():
    print("\n🔧 DATA_PREPARATION_1.PY - Enhanced Processing")
    # Get configuration
    reference_port = config.reference_port
    puerto_coords = (config.REFERENCE_PORTS[reference_port]['latitude'], config.REFERENCE_PORTS[reference_port]['longitude'])
    run_path = config.RUN_PATH
    # Input/output paths
    swan_daily_path = os.path.join(config.PROCESSED_DATA_DIR, 'df_swan_daily_enhanced.csv')
    waverys_daily_path = os.path.join(config.PROCESSED_DATA_DIR, 'df_waverys_daily.csv')
    output_dir = config.PROCESSED_DATA_DIR
    # Step 1: Load and validate
    df_swan_daily, df_waverys_daily = load_and_validate_daily_data(swan_daily_path, waverys_daily_path, reference_port)
    if df_swan_daily.empty or df_waverys_daily.empty:
        print("❌ One or more required input files are missing or empty. Exiting.")
        return
    # Step 2: Detrend/deseasonalize
    df_swan_processed = detrend_and_deseasonalize_reference_point(df_swan_daily)
    # Step 3: Enhanced feature engineering
    df_swan_features = create_enhanced_features_reference_point(df_swan_processed)
    # Step 4: Merge with WAVERYS
    print("\nSTEP 4: Merging with WAVERYS data")
    merged = pd.merge(df_swan_features, df_waverys_daily, on=['date', 'port_name'], suffixes=('_swan', '_waverys'), how='inner')
    print(f"Merged dataset shape: {merged.shape}")
    # Step 5: Save outputs
    swan_features_path = os.path.join(output_dir, 'df_swan_daily_features.csv')
    merged_path = os.path.join(output_dir, 'df_swan_waverys_merged.csv')
    df_swan_features.to_csv(swan_features_path, index=False)
    merged.to_csv(merged_path, index=False)
    print(f"✅ Saved: {swan_features_path} ({df_swan_features.shape})")
    print(f"✅ Saved: {merged_path} ({merged.shape})")


if __name__ == "__main__":
    main()
