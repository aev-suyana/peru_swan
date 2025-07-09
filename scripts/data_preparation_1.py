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
    """
    Create enhanced features on the reference point time series (full memory-efficient version)
    """
    import gc
    import numpy as np
    print(f"\nSTEP 3: Enhanced feature engineering on reference point data")
    print(f"Input shape: {df_processed.shape}")
    print(f"Use processed features: {use_processed_features}")
    
    df_enhanced = df_processed.copy()
    df_enhanced['date'] = pd.to_datetime(df_enhanced['date'])
    df_enhanced = df_enhanced.sort_values('date').reset_index(drop=True)
    
    # Choose features to enhance
    if use_processed_features:
        base_features = [col for col in df_enhanced.columns if col.endswith('_processed')]
        feature_type = "processed"
    else:
        base_features = [col for col in df_enhanced.columns if col.endswith('_raw')]
        feature_type = "raw"
    # Also include pct_ columns and other base features
    additional_features = [col for col in df_enhanced.columns 
                          if (col.startswith('pct_') or 
                              (any(wave_type in col for wave_type in ['swh', 'swe']) 
                               and not col.endswith(('_processed', '_raw'))))
                          and col not in ['date', 'event_dummy_1', 'total_obs_sw', 'port_name', 'year']]
    base_features.extend(additional_features)
    base_features = list(set(base_features))  # Remove duplicates
    print(f"Creating enhanced features from {len(base_features)} {feature_type} features...")
    # Get valid features that exist in the dataframe
    valid_base_features = [f for f in base_features if f in df_enhanced.columns]
    if len(valid_base_features) == 0:
        print("No valid base features found!")
        return df_enhanced
    # 3a: Memory-efficient persistence features
    print("  Creating persistence features (memory-efficient)...")
    PERSISTENCE_WINDOWS = [2, 3, 5, 7, 14]
    TREND_WINDOWS = [3, 5, 7, 14]
    CHANGE_WINDOWS = [3, 5, 7, 14]
    LAG_WINDOWS = [1, 3, 5, 7, 14]
    for window in PERSISTENCE_WINDOWS:
        for feature in valid_base_features:
            col_name = f'{feature}_persistence_{window}'
            df_enhanced[col_name] = df_enhanced[feature].rolling(window, min_periods=1).mean()
    gc.collect()
    # 3b: Memory-efficient trend features
    print("  Creating trend features (memory-efficient)...")
    for window in TREND_WINDOWS:
        for feature in valid_base_features:
            mean_col = f'{feature}_rolling_mean_{window}'
            df_enhanced[mean_col] = df_enhanced[feature].rolling(window, min_periods=2).mean()
        print(f"    Computing slopes for window {window}...")
        chunk_size = 10
        for i in range(0, len(valid_base_features), chunk_size):
            chunk_features = valid_base_features[i:i+chunk_size]
            chunk_data = df_enhanced[chunk_features].values
            n_rows, n_features = chunk_data.shape
            slopes = np.full((n_rows, n_features), np.nan)
            x = np.arange(window)
            x_mean = x.mean()
            x_centered = x - x_mean
            x_var = np.sum(x_centered ** 2)
            for row_idx in range(window-1, n_rows):
                start_idx = row_idx - window + 1
                y_window = chunk_data[start_idx:row_idx+1, :]
                valid_mask = ~np.isnan(y_window).all(axis=0)
                if valid_mask.any():
                    y_valid = y_window[:, valid_mask]
                    y_mean = np.nanmean(y_valid, axis=0)
                    y_centered = y_valid - y_mean
                    numerator = np.nansum(x_centered[:, np.newaxis] * y_centered, axis=0)
                    slopes_valid = numerator / x_var
                    slopes[row_idx, valid_mask] = slopes_valid
            for j, feature in enumerate(chunk_features):
                col_name = f'{feature}_trend_{window}'
                df_enhanced[col_name] = slopes[:, j]
        gc.collect()
    # 3c: Memory-efficient change features
    print("  Creating change features (memory-efficient)...")
    for window in CHANGE_WINDOWS:
        for feature in valid_base_features:
            abs_change_col = f'{feature}_abs_change_{window}'
            df_enhanced[abs_change_col] = df_enhanced[feature] - df_enhanced[feature].shift(window)
            rel_change_col = f'{feature}_rel_change_{window}'
            past_values = df_enhanced[feature].shift(window)
            df_enhanced[rel_change_col] = np.where(past_values != 0,
                                                  ((df_enhanced[feature] - past_values) / past_values) * 100,
                                                  0)
        gc.collect()
    # 3d: Memory-efficient lag features
    print("  Creating lag features (memory-efficient)...")
    all_features_to_lag = valid_base_features.copy()
    for feature in valid_base_features:
        for window in PERSISTENCE_WINDOWS:
            col_name = f'{feature}_persistence_{window}'
            if col_name in df_enhanced.columns:
                all_features_to_lag.append(col_name)
        for window in TREND_WINDOWS:
            for suffix in ['_trend_', '_rolling_mean_']:
                col_name = f'{feature}{suffix}{window}'
                if col_name in df_enhanced.columns:
                    all_features_to_lag.append(col_name)
        for window in CHANGE_WINDOWS:
            for suffix in ['_abs_change_', '_rel_change_']:
                col_name = f'{feature}{suffix}{window}'
                if col_name in df_enhanced.columns:
                    all_features_to_lag.append(col_name)
    print(f"    Creating lags for {len(all_features_to_lag)} features...")
    for lag in LAG_WINDOWS:
        print(f"      Processing lag {lag}...")
        chunk_size = 50
        for i in range(0, len(all_features_to_lag), chunk_size):
            chunk_features = all_features_to_lag[i:i+chunk_size]
            valid_chunk_features = [f for f in chunk_features if f in df_enhanced.columns]
            for feature in valid_chunk_features:
                lag_col = f'{feature}_lag_{lag}'
                df_enhanced[lag_col] = df_enhanced[feature].shift(lag)
        gc.collect()
    print("  Lag features complete!")
    original_features = len(df_processed.columns)
    enhanced_features = len(df_enhanced.columns)
    new_features = enhanced_features - original_features
    print(f"Enhanced feature engineering complete:")
    print(f"  Original features: {original_features}")
    print(f"  Enhanced features: {enhanced_features}")
    print(f"  New features created: {new_features}")
    print(f"  Output shape: {df_enhanced.shape}")
    return df_enhanced


def main():
    print("\n🔧 DATA_PREPARATION_1.PY - Enhanced Processing")
    # Get configuration
    reference_port = config.reference_port
    puerto_coords = (config.REFERENCE_PORTS[reference_port]['latitude'], config.REFERENCE_PORTS[reference_port]['longitude'])
    run_path = config.RUN_PATH
    # Input/output paths
    run_dir = os.path.join(config.PROCESSED_DATA_DIR, run_path)
    swan_daily_path = os.path.join(run_dir, 'df_swan_daily_enhanced.csv')
    waverys_daily_path = os.path.join(run_dir, 'df_waverys_daily.csv')
    output_dir = run_dir
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
    df_swan_features['port_name'] = reference_port
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
