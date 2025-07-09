# ============================================================================
# ENHANCED FEATURE SELECTION FOR CV PIPELINE
# Combines systematic 5-method approach with domain-specific optimization
# ============================================================================

from config import config, get_input_files, get_output_files
import pandas as pd
import numpy as np
import re
import os
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# --- Configuration for enhanced feature selection ---
class EnhancedFeatureSelectionConfig:
    TOP_K_PER_METHOD = 200
    FINAL_TOP_K = 150
    MIN_VOTES = 2
    METHOD_WEIGHTS = {
        'random_forest': 2.5,
        'extra_trees': 2.0,
        'mutual_info': 2.2,
        'f_test': 1.8,
        'domain_selection': 3.0
    }
    DOMAIN_PATTERNS = {
        'max_waves': (r'swh_max_sw|swe_max_sw|swh_max_wa|swe_max_wa', 4.0),
        'swan_features': (r'_sw$', 3.5),
        'extreme_percentiles': (r'_p(85|90|95|99)', 3.0),
        'high_percentiles': (r'_p(75|80)', 2.5),
        'medium_trends': (r'_trend_(7|14|21)', 2.8),
        'short_trends': (r'_trend_(3|5)', 2.2),
        'persistence': (r'_persistence_(5|7|14)', 2.6),
        'recent_changes': (r'_change_(1|3|5)', 2.4),
        'medium_changes': (r'_change_(7|14)', 2.0),
        'anomalies': (r'anom_', 2.2),
        'variability': (r'_cv|_range|_iqr|_std', 2.0),
        'immediate_lags': (r'_lag_(1|2)', 2.0),
        'short_lags': (r'_lag_(3|5)', 1.8),
        'waverys_data': (r'_wa$', 1.8),
    }

# ============================================================================
# ENHANCED FEATURE SELECTION, RULE EVALUATION, AND ML PIPELINE FUNCTIONS
# (Restored from SWAN_MOD_rule_evaluation_v3.py)
# ============================================================================

import os
import re
import glob
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from concurrent.futures import ThreadPoolExecutor
import time

warnings.filterwarnings('ignore')

class EnhancedFeatureSelectionConfig:
    """Configuration for enhanced feature selection"""
    TOP_K_PER_METHOD = 200
    FINAL_TOP_K = 150
    MIN_VOTES = 2
    METHOD_WEIGHTS = {
        'random_forest': 2.5,
        'extra_trees': 2.0,
        'mutual_info': 2.2,
        'f_test': 1.8,
        'domain_selection': 3.0
    }
    DOMAIN_PATTERNS = {
        'max_waves': (r'swh_max_sw|swe_max_sw|swh_max_wa|swe_max_wa', 4.0),
        'swan_features': (r'_sw$', 3.5),
        'extreme_percentiles': (r'_p(85|90|95|99)', 3.0),
        'high_percentiles': (r'_p(75|80)', 2.5),
        'medium_trends': (r'_trend_(7|14|21)', 2.8),
        'short_trends': (r'_trend_(3|5)', 2.2),
        'persistence': (r'_persistence_(5|7|14)', 2.6),
        'recent_changes': (r'_change_(1|3|5)', 2.4),
        'medium_changes': (r'_change_(7|14)', 2.0),
        'anomalies': (r'anom_', 2.2),
        'variability': (r'_cv|_range|_iqr|_std', 2.0),
        'immediate_lags': (r'_lag_(1|2)', 2.0),
        'short_lags': (r'_lag_(3|5)', 1.8),
        'waverys_data': (r'_wa$', 1.8),
    }


# =====================
# BEGIN RESTORED FUNCTIONS FROM SWAN_MOD_rule_evaluation_v3.py
# =====================
# ============================================================================
# STEP 1: LOAD AND VALIDATE DATA
# ============================================================================

def load_and_validate_daily_data(swan_daily_path, waverys_daily_path, reference_port):
    """
    Load and validate daily SWAN and WAVERYS data
    
    Parameters:
    -----------
    swan_daily_path : str
        Path to SWAN daily CSV file
    waverys_daily_path : str  
        Path to WAVERYS daily CSV file
    reference_port : str
        Name of reference port
        
    Returns:
    --------
    tuple : (df_swan_daily, df_waverys_daily)
    """
    
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
            
            # Filter for reference port
            df_waverys_daily = df_waverys_daily[
                df_waverys_daily['port_name'] == reference_port
            ].copy()
            
            print(f"✓ Loaded WAVERYS data for {reference_port}: {df_waverys_daily.shape}")
        else:
            print(f"✗ WAVERYS daily file not found: {waverys_daily_path}")
            df_waverys_daily = pd.DataFrame()
    except Exception as e:
        print(f"✗ Error loading WAVERYS data: {e}")
        df_waverys_daily = pd.DataFrame()
    
    # Validation
    if not df_swan_daily.empty:
        print(f"SWAN data date range: {df_swan_daily['date'].min().date()} to {df_swan_daily['date'].max().date()}")
        
    if not df_waverys_daily.empty:
        print(f"WAVERYS data date range: {df_waverys_daily['date'].min().date()} to {df_waverys_daily['date'].max().date()}")
        print(f"WAVERYS ports: {df_waverys_daily['port_name'].unique()}")
    
    return df_swan_daily, df_waverys_daily

# ============================================================================
# STEP 2: DETREND AND DESEASONALIZE (REFERENCE POINT)
# ============================================================================

def detrend_and_deseasonalize_reference_point(df_daily, apply_detrending=True, apply_deseasonalizing=True):
    """
    Apply detrending and deseasonalizing to reference point data
    
    Parameters:
    -----------
    df_daily : DataFrame
        Daily data for reference point
    apply_detrending : bool
        Whether to remove long-term trends
    apply_deseasonalizing : bool
        Whether to remove seasonal cycles
    
    Returns:
    --------
    DataFrame with processed features added
    """
    
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
    
    # Get wave feature columns (exclude metadata)
    wave_features = [col for col in df_daily.columns 
                    if any(wave_type in col for wave_type in ['swh', 'swe']) 
                    and col not in ['date', 'event_dummy_1', 'total_obs_sw', 'port_name', 'year']]
    
    print(f"Processing {len(wave_features)} wave features...")
    print(f"Example features: {wave_features[:5]}")
    
    if len(df_processed) < ProcessingConfig.MIN_DATA_POINTS:
        print(f"Warning: Insufficient data ({len(df_processed)} days) for processing")
        return df_processed
    
    # Create day of year once if deseasonalizing
    if apply_deseasonalizing:
        df_processed['day_of_year'] = df_processed['date'].dt.dayofyear
    
    # Vectorized processing of ALL wave features at once
    valid_features = [f for f in wave_features if f in df_processed.columns]
    
    if len(valid_features) == 0:
        print("No valid wave features found for processing")
        return df_processed
    
    print(f"Vectorized processing of {len(valid_features)} features...")
    
    # Step 2a: Vectorized deseasonalization for all features
    if apply_deseasonalizing:
        # Calculate seasonal climatology for all features at once
        seasonal_avgs = df_processed.groupby('day_of_year')[valid_features].transform('mean')
        feature_means = df_processed[valid_features].mean()
        
        # Vectorized deseasonalization: remove seasonal cycle, preserve mean
        deseasonalized = df_processed[valid_features] - seasonal_avgs + feature_means
        
        # Store deseasonalized versions
        for feature in valid_features:
            df_processed[f"{feature}_deseasonalized"] = deseasonalized[feature]
    else:
        # If not deseasonalizing, copy original features
        for feature in valid_features:
            df_processed[f"{feature}_deseasonalized"] = df_processed[feature]
    
    # Step 2b: Vectorized detrending for all features
    if apply_detrending:
        # Get the deseasonalized features for detrending
        deseason_features = [f"{f}_deseasonalized" for f in valid_features]
        deseason_data = df_processed[deseason_features]
        
        # Fill missing values for detrending (vectorized)
        filled_data = deseason_data.fillna(deseason_data.mean())
        
        # Vectorized detrending for all features at once
        detrended_array = signal.detrend(filled_data.values, axis=0, type='linear')
        
        # Add back means and convert to DataFrame
        means = filled_data.mean().values
        detrended_df = pd.DataFrame(
            detrended_array + means, 
            index=filled_data.index, 
            columns=deseason_features
        )
        
        # Store processed and raw versions
        for i, feature in enumerate(valid_features):
            df_processed[f"{feature}_processed"] = detrended_df[f"{feature}_deseasonalized"]
            df_processed[f"{feature}_raw"] = df_processed[feature]
    else:
        # If not detrending, use deseasonalized as processed
        for feature in valid_features:
            df_processed[f"{feature}_processed"] = df_processed[f"{feature}_deseasonalized"]
            df_processed[f"{feature}_raw"] = df_processed[feature]
        
        # Clean up temporary deseasonalized columns
        temp_cols = [f"{f}_deseasonalized" for f in valid_features]
        df_processed.drop(columns=temp_cols, inplace=True)
    
    # Count processed features
    processed_features = [col for col in df_processed.columns if col.endswith('_processed')]
    raw_features = [col for col in df_processed.columns if col.endswith('_raw')]
    
    print(f"Processing complete:")
    print(f"  Processed features created: {len(processed_features)}")
    print(f"  Raw features preserved: {len(raw_features)}")
    
    return df_processed

# ============================================================================
# STEP 3: ENHANCED FEATURE ENGINEERING
# ============================================================================

def create_enhanced_features_reference_point(df_processed, use_processed_features=True):
    """
    Create enhanced features on the reference point time series
    
    Parameters:
    -----------
    df_processed : DataFrame
        Processed daily time series for reference point
    use_processed_features : bool
        Use processed features (True) or raw features (False)
    
    Returns:
    --------
    DataFrame with enhanced features added
    """
    
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
    for window in ProcessingConfig.PERSISTENCE_WINDOWS:
        # Direct column assignment instead of concat (memory efficient)
        for feature in valid_base_features:
            col_name = f'{feature}_persistence_{window}'
            df_enhanced[col_name] = df_enhanced[feature].rolling(window, min_periods=1).mean()
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # 3b: Memory-efficient trend features 
    print("  Creating trend features (memory-efficient)...")
    for window in ProcessingConfig.TREND_WINDOWS:
        # Direct column assignment for rolling mean
        for feature in valid_base_features:
            mean_col = f'{feature}_rolling_mean_{window}'
            df_enhanced[mean_col] = df_enhanced[feature].rolling(window, min_periods=2).mean()
        
        # Memory-efficient slope calculation
        print(f"    Computing slopes for window {window}...")
        
        # Process in smaller chunks to reduce memory usage
        chunk_size = 10  # Process 10 features at a time
        
        for i in range(0, len(valid_base_features), chunk_size):
            chunk_features = valid_base_features[i:i+chunk_size]
            
            # Get data for this chunk only
            chunk_data = df_enhanced[chunk_features].values
            n_rows, n_features = chunk_data.shape
            
            # Pre-allocate for this chunk only
            slopes = np.full((n_rows, n_features), np.nan)
            
            # Slope calculation for chunk
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
            
            # Assign directly to DataFrame (memory efficient)
            for j, feature in enumerate(chunk_features):
                col_name = f'{feature}_trend_{window}'
                df_enhanced[col_name] = slopes[:, j]
        
        # Force garbage collection after each window
        gc.collect()
    
    # 3c: Memory-efficient change features
    print("  Creating change features (memory-efficient)...")
    
    for window in ProcessingConfig.CHANGE_WINDOWS:
        # Direct column assignment instead of large array operations
        for feature in valid_base_features:
            # Absolute change
            abs_change_col = f'{feature}_abs_change_{window}'
            df_enhanced[abs_change_col] = df_enhanced[feature] - df_enhanced[feature].shift(window)
            
            # Relative change
            rel_change_col = f'{feature}_rel_change_{window}'
            past_values = df_enhanced[feature].shift(window)
            df_enhanced[rel_change_col] = np.where(past_values != 0,
                                                  ((df_enhanced[feature] - past_values) / past_values) * 100,
                                                  0)
        
        # Garbage collection after each window
        gc.collect()
    
    # 3d: Memory-efficient lag features
    print("  Creating lag features (memory-efficient)...")
    
    # Get all newly created feature names (but only ones that exist)
    all_features_to_lag = valid_base_features.copy()
    
    # Add engineered features that actually exist in the dataframe
    for feature in valid_base_features:
        for window in ProcessingConfig.PERSISTENCE_WINDOWS:
            col_name = f'{feature}_persistence_{window}'
            if col_name in df_enhanced.columns:
                all_features_to_lag.append(col_name)
                
        for window in ProcessingConfig.TREND_WINDOWS:
            for suffix in ['_trend_', '_rolling_mean_']:
                col_name = f'{feature}{suffix}{window}'
                if col_name in df_enhanced.columns:
                    all_features_to_lag.append(col_name)
                    
        for window in ProcessingConfig.CHANGE_WINDOWS:
            for suffix in ['_abs_change_', '_rel_change_']:
                col_name = f'{feature}{suffix}{window}'
                if col_name in df_enhanced.columns:
                    all_features_to_lag.append(col_name)
    
    print(f"    Creating lags for {len(all_features_to_lag)} features...")
    
    # Process lags one at a time to avoid memory explosion
    for lag in ProcessingConfig.LAG_WINDOWS:
        print(f"      Processing lag {lag}...")
        
        # Process features in chunks to manage memory
        chunk_size = 50  # Process 50 features at a time
        
        for i in range(0, len(all_features_to_lag), chunk_size):
            chunk_features = all_features_to_lag[i:i+chunk_size]
            valid_chunk_features = [f for f in chunk_features if f in df_enhanced.columns]
            
            # Direct assignment instead of concat
            for feature in valid_chunk_features:
                lag_col = f'{feature}_lag_{lag}'
                df_enhanced[lag_col] = df_enhanced[feature].shift(lag)
        
        # Force garbage collection after each lag
        gc.collect()
    
    print("  Lag features complete!")
    
    # Count new features
    original_features = len(df_processed.columns)
    enhanced_features = len(df_enhanced.columns)
    new_features = enhanced_features - original_features
    
    print(f"Enhanced feature engineering complete:")
    print(f"  Original features: {original_features}")
    print(f"  Enhanced features: {enhanced_features}")
    print(f"  New features created: {new_features}")
    print(f"  Output shape: {df_enhanced.shape}")
    
    return df_enhanced

# ============================================================================
# COMPLETE PIPELINE FUNCTION
# ============================================================================

def run_reference_port_processing_pipeline(swan_daily_path, waverys_daily_path, 
                                          reference_port, puerto_coords, run_path, 
                                          base_drive, save_outputs=True):
    """
    Run the complete processing pipeline for reference port approach
    
    Parameters:
    -----------
    swan_daily_path : str
        Path to SWAN daily CSV file
    waverys_daily_path : str
        Path to WAVERYS daily CSV file
    reference_port : str
        Name of reference port
    puerto_coords : tuple
        (latitude, longitude) of reference port
    run_path : str
        Run identifier for output paths
    base_drive : str
        Base drive path
    save_outputs : bool
        Save intermediate and final outputs
    
    Returns:
    --------
    dict with all processed datasets
    """
    
    # Setup configuration
    setup_processing_config(reference_port, puerto_coords, run_path, base_drive)
    
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if save_outputs:
        os.makedirs(ProcessingConfig.OUTPUT_DIR, exist_ok=True)
    
    # STEP 1: Load and validate data
    df_swan_daily, df_waverys_daily = load_and_validate_daily_data(
        swan_daily_path, waverys_daily_path, reference_port
    )
    
    if df_swan_daily.empty:
        print("ERROR: No SWAN daily data available!")
        return results
    
    # STEP 2: Detrend and deseasonalize
    print(f"\n{'='*60}")
    df_processed = detrend_and_deseasonalize_reference_point(
        df_swan_daily, 
        apply_detrending=ProcessingConfig.APPLY_DETRENDING,
        apply_deseasonalizing=ProcessingConfig.APPLY_DESEASONALIZING
    )
    results['processed_reference_point'] = df_processed
    
    if save_outputs:
        print(f"Saving processed data ({df_processed.shape})...")
        processed_file = os.path.join(ProcessingConfig.OUTPUT_DIR, f"processed_{reference_port}_{timestamp}")
        
        # Save as Parquet (much faster than CSV)
        df_processed.to_parquet(f"{processed_file}.parquet", index=False)
        print(f"✓ Saved: {processed_file}.parquet")
    
    # STEP 3: Enhanced feature engineering for both processed and raw features
    print(f"\n{'='*60}")
    
    for use_processed in [True, False]:
        feature_type = "processed" if use_processed else "raw"
        
        print(f"\nCreating enhanced features using {feature_type} features...")
        df_enhanced = create_enhanced_features_reference_point(
            df_processed, use_processed_features=use_processed
        )
        
        # Store results
        key = f'enhanced_{reference_port}_{feature_type}'
        results[key] = df_enhanced
        
        if save_outputs:
            print(f"  Saving enhanced features ({df_enhanced.shape})...")
            enhanced_file = os.path.join(ProcessingConfig.OUTPUT_DIR, 
                                       f"enhanced_{reference_port}_{feature_type}_{timestamp}")
            
            # Save as Parquet (10-50x faster than CSV for large files)
            df_enhanced.to_parquet(f"{enhanced_file}.parquet", index=False)
            print(f"  ✓ Saved: {enhanced_file}.parquet")
    
    # STEP 4: Merge with WAVERYS data
    if not df_waverys_daily.empty:
        print(f"\n{'='*60}")
        print("STEP 4: MERGING WITH WAVERYS DATA")
        
        # Create separate dictionary to avoid iteration issues
        combined_results = {}
        
        for key, df_enhanced in results.items():
            if key.startswith('enhanced_'):
                # Merge on date and event_dummy_1 if available
                merge_cols = ['date']
                if 'event_dummy_1' in df_enhanced.columns and 'event_dummy_1' in df_waverys_daily.columns:
                    merge_cols.append('event_dummy_1')
                
                df_combined = df_enhanced.merge(df_waverys_daily, on=merge_cols, how='left')
                
                # Only keep rows where we have WAVERYS data
                if 'port_name' in df_combined.columns:
                    df_combined.dropna(subset=['port_name'], inplace=True)
                
                combined_key = f'combined_{key.replace("enhanced_", "")}'
                combined_results[combined_key] = df_combined
                
                print(f"Created {combined_key}: {df_combined.shape}")
                
                if save_outputs:
                    print(f"  Saving combined data ({df_combined.shape})...")
                    combined_file = os.path.join(ProcessingConfig.OUTPUT_DIR, 
                                               f"combined_{key.replace('enhanced_', '')}_{timestamp}")
                    
                    # Save as Parquet (much faster)
                    df_combined.to_parquet(f"{combined_file}.parquet", index=False)
                    print(f"  ✓ Saved: {combined_file}.parquet")
        
        # Add combined results to main results
        results.update(combined_results)
    else:
        print(f"\nWarning: No WAVERYS data available for {reference_port}")
    
    # Save processing summary
    if save_outputs:
        summary = {
            'timestamp': timestamp,
            'reference_port': reference_port,
            'puerto_coords': puerto_coords,
            'run_path': run_path,
            'config': {
                'detrending': ProcessingConfig.APPLY_DETRENDING,
                'deseasonalizing': ProcessingConfig.APPLY_DESEASONALIZING,
                'reference_port': ProcessingConfig.REFERENCE_PORT,
                'puerto_coords': ProcessingConfig.PUERTO_COORDS
            },
            'datasets_created': {k: v.shape for k, v in results.items()},
            'processing_order': [
                "1. Load and validate daily data",
                "2. Detrend & Deseasonalize (reference point)",
                "3. Enhanced Feature Engineering",
                "4. Merge with WAVERYS data"
            ]
        }
        
        import json
        summary_file = os.path.join(ProcessingConfig.OUTPUT_DIR, f"processing_summary_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Saved processing summary: {summary_file}")
    
    print(f"\n{'='*80}")
    print("PROCESSING PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"Reference port: {reference_port}")
    print(f"Datasets created: {len(results)}")
    for name, df in results.items():
        print(f"  {name}: {df.shape}")
    
    return results

# ============================================================================
# EXECUTION WRAPPER
# ============================================================================

def execute_processing_pipeline(run_path, reference_port, reference_latitude, reference_longitude, base_drive):
    """
    Execute the processing pipeline that reads single-point component files and merges them
    
    Parameters:
    -----------
    run_path : str
        Current run path identifier
    reference_port : str 
        Current reference port name
    reference_latitude : float
        Latitude of reference port
    reference_longitude : float
        Longitude of reference port
    base_drive : str
        Base drive path
    """
    
    # Set file paths for the component files created in Step 1
    swan_out_path = f'{base_drive}/swan/out/{run_path}'
    swan_daily_path = f'{swan_out_path}/df_swan_daily_new_colnames.csv'
    waverys_daily_path = f'{swan_out_path}/df_waverys_daily_new_colnames.csv'
    
    print(f"Reading component files:")
    print(f"  SWAN daily: {swan_daily_path}")
    print(f"  WAVERYS daily: {waverys_daily_path}")
    
    # Read the component files (already single point each)
    df_swan_daily = pd.read_csv(swan_daily_path)
    df_waverys_daily = pd.read_csv(waverys_daily_path)
    
    print(f"✅ Loaded SWAN data: {df_swan_daily.shape}")
    print(f"✅ Loaded WAVERYS data: {df_waverys_daily.shape}")
    
    # MERGE: Combine SWAN and WAVERYS data
    print(f"\n🔗 Merging SWAN and WAVERYS data...")
    
    # Convert date columns to datetime
    df_swan_daily['date'] = pd.to_datetime(df_swan_daily['date'])
    df_waverys_daily['date'] = pd.to_datetime(df_waverys_daily['date'])
    
    # Merge on date (and event_dummy_1 if available in both)
    merge_cols = ['date']
    if 'event_dummy_1' in df_swan_daily.columns and 'event_dummy_1' in df_waverys_daily.columns:
        merge_cols.append('event_dummy_1')
    
    df_final = df_swan_daily.merge(df_waverys_daily, on=merge_cols, how='inner')
    
    print(f"✅ Final merged dataset: {df_final.shape}")
    
    # Save the final dataset (equivalent to df_final_40km.csv)
    output_dir = f'{base_drive}/swan/out/{run_path}'
    final_file = f'{output_dir}/df_final_{reference_port}_aggregated.csv'
    
    df_final.to_csv(final_file, index=False)
    print(f"✅ Saved final dataset: {final_file}")
    
    return df_final, df_swan_daily, df_waverys_daily

"""
DATA PROCESSING PIPELINE - MODIFIED FOR CSV REFERENCE PORT APPROACH
===================================================================

This script processes daily wave data for a single reference point approach:
1. Detrend and deseasonalize (single reference point) 
2. Enhanced feature engineering (on reference point data)
3. Combine with WAVERYS data

Assumes you already have:
- CSV file with daily wave statistics for reference point
- WAVERYS data for the reference port
- Reference port configuration

Author: Wave Analysis Team
Date: 2024
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class ProcessingConfig:
    """Configuration for the processing pipeline"""
    
    # File paths - will be set dynamically based on run_path
    OUTPUT_DIR = None
    
    # Port configuration - will be set dynamically
    REFERENCE_PORT = None
    PUERTO_COORDS = None
    
    # Processing options
    APPLY_DETRENDING = True
    APPLY_DESEASONALIZING = True
    
    # Feature engineering windows
    PERSISTENCE_WINDOWS = [2, 3, 5, 7, 14]
    TREND_WINDOWS = [3, 5, 7, 14] 
    CHANGE_WINDOWS = [3, 5, 7, 14]
    LAG_WINDOWS = [1, 3, 5, 7, 14]
    
    # Data quality thresholds
    MIN_DATA_POINTS = 30  # Minimum days needed for detrending
    MAX_MISSING_PCT = 0.3  # Maximum missing data allowed

def setup_processing_config(reference_port, puerto_coords, run_path, base_drive):
    """Setup configuration for the processing pipeline"""
    ProcessingConfig.REFERENCE_PORT = reference_port
    ProcessingConfig.PUERTO_COORDS = puerto_coords
    ProcessingConfig.OUTPUT_DIR = f"{base_drive}/suyana/peru/processed_data/{run_path}/"
    
    print("="*80)
    print("WAVE DATA PROCESSING PIPELINE - REFERENCE PORT APPROACH")
    print("="*80)
    print(f"Reference port: {reference_port}")
    print(f"Coordinates: {puerto_coords}")
    print(f"Run path: {run_path}")
    print("Processing order:")
    print("1. Load and validate daily data")  
    print("2. Detrend & Deseasonalize (reference point)")
    print("3. Enhanced Feature Engineering")
    print("4. Merge with WAVERYS data")
    print("="*80)

# ============================================================================
# STEP 1: LOAD AND VALIDATE DATA
# ============================================================================

def load_and_validate_daily_data(swan_daily_path, waverys_daily_path, reference_port):
    """
    Load and validate daily SWAN and WAVERYS data
    
    Parameters:
    -----------
    swan_daily_path : str
        Path to SWAN daily CSV file
    waverys_daily_path : str  
        Path to WAVERYS daily CSV file
    reference_port : str
        Name of reference port
        
    Returns:
    --------
    tuple : (df_swan_daily, df_waverys_daily)
    """
    
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
            
            # Filter for reference port
            df_waverys_daily = df_waverys_daily[
                df_waverys_daily['port_name'] == reference_port
            ].copy()
            
            print(f"✓ Loaded WAVERYS data for {reference_port}: {df_waverys_daily.shape}")
        else:
            print(f"✗ WAVERYS daily file not found: {waverys_daily_path}")
            df_waverys_daily = pd.DataFrame()
    except Exception as e:
        print(f"✗ Error loading WAVERYS data: {e}")
        df_waverys_daily = pd.DataFrame()
    
    # Validation
    if not df_swan_daily.empty:
        print(f"SWAN data date range: {df_swan_daily['date'].min().date()} to {df_swan_daily['date'].max().date()}")
        
    if not df_waverys_daily.empty:
        print(f"WAVERYS data date range: {df_waverys_daily['date'].min().date()} to {df_waverys_daily['date'].max().date()}")
        print(f"WAVERYS ports: {df_waverys_daily['port_name'].unique()}")
    
    return df_swan_daily, df_waverys_daily

# ============================================================================
# STEP 2: DETREND AND DESEASONALIZE (REFERENCE POINT)
# ============================================================================

def detrend_and_deseasonalize_reference_point(df_daily, apply_detrending=True, apply_deseasonalizing=True):
    """
    Apply detrending and deseasonalizing to reference point data
    
    Parameters:
    -----------
    df_daily : DataFrame
        Daily data for reference point
    apply_detrending : bool
        Whether to remove long-term trends
    apply_deseasonalizing : bool
        Whether to remove seasonal cycles
    
    Returns:
    --------
    DataFrame with processed features added
    """
    
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
    
    # Get wave feature columns (exclude metadata)
    wave_features = [col for col in df_daily.columns 
                    if any(wave_type in col for wave_type in ['swh', 'swe']) 
                    and col not in ['date', 'event_dummy_1', 'total_obs_sw', 'port_name', 'year']]
    
    print(f"Processing {len(wave_features)} wave features...")
    print(f"Example features: {wave_features[:5]}")
    
    if len(df_processed) < ProcessingConfig.MIN_DATA_POINTS:
        print(f"Warning: Insufficient data ({len(df_processed)} days) for processing")
        return df_processed
    
    # Create day of year once if deseasonalizing
    if apply_deseasonalizing:
        df_processed['day_of_year'] = df_processed['date'].dt.dayofyear
    
    # Vectorized processing of ALL wave features at once
    valid_features = [f for f in wave_features if f in df_processed.columns]
    
    if len(valid_features) == 0:
        print("No valid wave features found for processing")
        return df_processed
    
    print(f"Vectorized processing of {len(valid_features)} features...")
    
    # Step 2a: Vectorized deseasonalization for all features
    if apply_deseasonalizing:
        # Calculate seasonal climatology for all features at once
        seasonal_avgs = df_processed.groupby('day_of_year')[valid_features].transform('mean')
        feature_means = df_processed[valid_features].mean()
        
        # Vectorized deseasonalization: remove seasonal cycle, preserve mean
        deseasonalized = df_processed[valid_features] - seasonal_avgs + feature_means
        
        # Store deseasonalized versions
        for feature in valid_features:
            df_processed[f"{feature}_deseasonalized"] = deseasonalized[feature]
    else:
        # If not deseasonalizing, copy original features
        for feature in valid_features:
            df_processed[f"{feature}_deseasonalized"] = df_processed[feature]
    
    # Step 2b: Vectorized detrending for all features
    if apply_detrending:
        # Get the deseasonalized features for detrending
        deseason_features = [f"{f}_deseasonalized" for f in valid_features]
        deseason_data = df_processed[deseason_features]
        
        # Fill missing values for detrending (vectorized)
        filled_data = deseason_data.fillna(deseason_data.mean())
        
        # Vectorized detrending for all features at once
        detrended_array = signal.detrend(filled_data.values, axis=0, type='linear')
        
        # Add back means and convert to DataFrame
        means = filled_data.mean().values
        detrended_df = pd.DataFrame(
            detrended_array + means, 
            index=filled_data.index, 
            columns=deseason_features
        )
        
        # Store processed and raw versions
        for i, feature in enumerate(valid_features):
            df_processed[f"{feature}_processed"] = detrended_df[f"{feature}_deseasonalized"]
            df_processed[f"{feature}_raw"] = df_processed[feature]
    else:
        # If not detrending, use deseasonalized as processed
        for feature in valid_features:
            df_processed[f"{feature}_processed"] = df_processed[f"{feature}_deseasonalized"]
            df_processed[f"{feature}_raw"] = df_processed[feature]
        
        # Clean up temporary deseasonalized columns
        temp_cols = [f"{f}_deseasonalized" for f in valid_features]
        df_processed.drop(columns=temp_cols, inplace=True)
    
    # Count processed features
    processed_features = [col for col in df_processed.columns if col.endswith('_processed')]
    raw_features = [col for col in df_processed.columns if col.endswith('_raw')]
    
    print(f"Processing complete:")
    print(f"  Processed features created: {len(processed_features)}")
    print(f"  Raw features preserved: {len(raw_features)}")
    
    return df_processed

# ============================================================================
# STEP 3: ENHANCED FEATURE ENGINEERING
# ============================================================================

def create_enhanced_features_reference_point(df_processed, use_processed_features=True):
    """
    Create enhanced features on the reference point time series
    
    Parameters:
    -----------
    df_processed : DataFrame
        Processed daily time series for reference point
    use_processed_features : bool
        Use processed features (True) or raw features (False)
    
    Returns:
    --------
    DataFrame with enhanced features added
    """
    
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
    for window in ProcessingConfig.PERSISTENCE_WINDOWS:
        # Direct column assignment instead of concat (memory efficient)
        for feature in valid_base_features:
            col_name = f'{feature}_persistence_{window}'
            df_enhanced[col_name] = df_enhanced[feature].rolling(window, min_periods=1).mean()
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # 3b: Memory-efficient trend features 
    print("  Creating trend features (memory-efficient)...")
    for window in ProcessingConfig.TREND_WINDOWS:
        # Direct column assignment for rolling mean
        for feature in valid_base_features:
            mean_col = f'{feature}_rolling_mean_{window}'
            df_enhanced[mean_col] = df_enhanced[feature].rolling(window, min_periods=2).mean()
        
        # Memory-efficient slope calculation
        print(f"    Computing slopes for window {window}...")
        
        # Process in smaller chunks to reduce memory usage
        chunk_size = 10  # Process 10 features at a time
        
        for i in range(0, len(valid_base_features), chunk_size):
            chunk_features = valid_base_features[i:i+chunk_size]
            
            # Get data for this chunk only
            chunk_data = df_enhanced[chunk_features].values
            n_rows, n_features = chunk_data.shape
            
            # Pre-allocate for this chunk only
            slopes = np.full((n_rows, n_features), np.nan)
            
            # Slope calculation for chunk
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
            
            # Assign directly to DataFrame (memory efficient)
            for j, feature in enumerate(chunk_features):
                col_name = f'{feature}_trend_{window}'
                df_enhanced[col_name] = slopes[:, j]
        
        # Force garbage collection after each window
        gc.collect()
    
    # 3c: Memory-efficient change features
    print("  Creating change features (memory-efficient)...")
    
    for window in ProcessingConfig.CHANGE_WINDOWS:
        # Direct column assignment instead of large array operations
        for feature in valid_base_features:
            # Absolute change
            abs_change_col = f'{feature}_abs_change_{window}'
            df_enhanced[abs_change_col] = df_enhanced[feature] - df_enhanced[feature].shift(window)
            
            # Relative change
            rel_change_col = f'{feature}_rel_change_{window}'
            past_values = df_enhanced[feature].shift(window)
            df_enhanced[rel_change_col] = np.where(past_values != 0,
                                                  ((df_enhanced[feature] - past_values) / past_values) * 100,
                                                  0)
        
        # Garbage collection after each window
        gc.collect()
    
    # 3d: Memory-efficient lag features
    print("  Creating lag features (memory-efficient)...")
    
    # Get all newly created feature names (but only ones that exist)
    all_features_to_lag = valid_base_features.copy()
    
    # Add engineered features that actually exist in the dataframe
    for feature in valid_base_features:
        for window in ProcessingConfig.PERSISTENCE_WINDOWS:
            col_name = f'{feature}_persistence_{window}'
            if col_name in df_enhanced.columns:
                all_features_to_lag.append(col_name)
                
        for window in ProcessingConfig.TREND_WINDOWS:
            for suffix in ['_trend_', '_rolling_mean_']:
                col_name = f'{feature}{suffix}{window}'
                if col_name in df_enhanced.columns:
                    all_features_to_lag.append(col_name)
                    
        for window in ProcessingConfig.CHANGE_WINDOWS:
            for suffix in ['_abs_change_', '_rel_change_']:
                col_name = f'{feature}{suffix}{window}'
                if col_name in df_enhanced.columns:
                    all_features_to_lag.append(col_name)
    
    print(f"    Creating lags for {len(all_features_to_lag)} features...")
    
    # Process lags one at a time to avoid memory explosion
    for lag in ProcessingConfig.LAG_WINDOWS:
        print(f"      Processing lag {lag}...")
        
        # Process features in chunks to manage memory
        chunk_size = 50  # Process 50 features at a time
        
        for i in range(0, len(all_features_to_lag), chunk_size):
            chunk_features = all_features_to_lag[i:i+chunk_size]
            valid_chunk_features = [f for f in chunk_features if f in df_enhanced.columns]
            
            # Direct assignment instead of concat
            for feature in valid_chunk_features:
                lag_col = f'{feature}_lag_{lag}'
                df_enhanced[lag_col] = df_enhanced[feature].shift(lag)
        
        # Force garbage collection after each lag
        gc.collect()
    
    print("  Lag features complete!")
    
    # Count new features
    original_features = len(df_processed.columns)
    enhanced_features = len(df_enhanced.columns)
    new_features = enhanced_features - original_features
    
    print(f"Enhanced feature engineering complete:")
    print(f"  Original features: {original_features}")
    print(f"  Enhanced features: {enhanced_features}")
    print(f"  New features created: {new_features}")
    print(f"  Output shape: {df_enhanced.shape}")
    
    return df_enhanced

# ============================================================================
# COMPLETE PIPELINE FUNCTION
# ============================================================================

def run_reference_port_processing_pipeline(swan_daily_path, waverys_daily_path, 
                                          reference_port, puerto_coords, run_path, 
                                          base_drive, save_outputs=True):
    """
    Run the complete processing pipeline for reference port approach
    
    Parameters:
    -----------
    swan_daily_path : str
        Path to SWAN daily CSV file
    waverys_daily_path : str
        Path to WAVERYS daily CSV file
    reference_port : str
        Name of reference port
    puerto_coords : tuple
        (latitude, longitude) of reference port
    run_path : str
        Run identifier for output paths
    base_drive : str
        Base drive path
    save_outputs : bool
        Save intermediate and final outputs
    
    Returns:
    --------
    dict with all processed datasets
    """
    
    # Setup configuration
    setup_processing_config(reference_port, puerto_coords, run_path, base_drive)
    
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if save_outputs:
        os.makedirs(ProcessingConfig.OUTPUT_DIR, exist_ok=True)
    
    # STEP 1: Load and validate data
    df_swan_daily, df_waverys_daily = load_and_validate_daily_data(
        swan_daily_path, waverys_daily_path, reference_port
    )
    
    if df_swan_daily.empty:
        print("ERROR: No SWAN daily data available!")
        return results
    
    # STEP 2: Detrend and deseasonalize
    print(f"\n{'='*60}")
    df_processed = detrend_and_deseasonalize_reference_point(
        df_swan_daily, 
        apply_detrending=ProcessingConfig.APPLY_DETRENDING,
        apply_deseasonalizing=ProcessingConfig.APPLY_DESEASONALIZING
    )
    results['processed_reference_point'] = df_processed
    
    if save_outputs:
        print(f"Saving processed data ({df_processed.shape})...")
        processed_file = os.path.join(ProcessingConfig.OUTPUT_DIR, f"processed_{reference_port}_{timestamp}")
        
        # Save as Parquet (much faster than CSV)
        df_processed.to_parquet(f"{processed_file}.parquet", index=False)
        print(f"✓ Saved: {processed_file}.parquet")
    
    # STEP 3: Enhanced feature engineering for both processed and raw features
    print(f"\n{'='*60}")
    
    for use_processed in [True, False]:
        feature_type = "processed" if use_processed else "raw"
        
        print(f"\nCreating enhanced features using {feature_type} features...")
        df_enhanced = create_enhanced_features_reference_point(
            df_processed, use_processed_features=use_processed
        )
        
        # Store results
        key = f'enhanced_{reference_port}_{feature_type}'
        results[key] = df_enhanced
        
        if save_outputs:
            print(f"  Saving enhanced features ({df_enhanced.shape})...")
            enhanced_file = os.path.join(ProcessingConfig.OUTPUT_DIR, 
                                       f"enhanced_{reference_port}_{feature_type}_{timestamp}")
            
            # Save as Parquet (10-50x faster than CSV for large files)
            df_enhanced.to_parquet(f"{enhanced_file}.parquet", index=False)
            print(f"  ✓ Saved: {enhanced_file}.parquet")
    
    # STEP 4: Merge with WAVERYS data
    if not df_waverys_daily.empty:
        print(f"\n{'='*60}")
        print("STEP 4: MERGING WITH WAVERYS DATA")
        
        # Create separate dictionary to avoid iteration issues
        combined_results = {}
        
        for key, df_enhanced in results.items():
            if key.startswith('enhanced_'):
                # Merge on date and event_dummy_1 if available
                merge_cols = ['date']
                if 'event_dummy_1' in df_enhanced.columns and 'event_dummy_1' in df_waverys_daily.columns:
                    merge_cols.append('event_dummy_1')
                
                df_combined = df_enhanced.merge(df_waverys_daily, on=merge_cols, how='left')
                
                # Only keep rows where we have WAVERYS data
                if 'port_name' in df_combined.columns:
                    df_combined.dropna(subset=['port_name'], inplace=True)
                
                combined_key = f'combined_{key.replace("enhanced_", "")}'
                combined_results[combined_key] = df_combined
                
                print(f"Created {combined_key}: {df_combined.shape}")
                
                if save_outputs:
                    print(f"  Saving combined data ({df_combined.shape})...")
                    combined_file = os.path.join(ProcessingConfig.OUTPUT_DIR, 
                                               f"combined_{key.replace('enhanced_', '')}_{timestamp}")
                    
                    # Save as Parquet (much faster)
                    df_combined.to_parquet(f"{combined_file}.parquet", index=False)
                    print(f"  ✓ Saved: {combined_file}.parquet")
        
        # Add combined results to main results
        results.update(combined_results)
    else:
        print(f"\nWarning: No WAVERYS data available for {reference_port}")
    
    # Save processing summary
    if save_outputs:
        summary = {
            'timestamp': timestamp,
            'reference_port': reference_port,
            'puerto_coords': puerto_coords,
            'run_path': run_path,
            'config': {
                'detrending': ProcessingConfig.APPLY_DETRENDING,
                'deseasonalizing': ProcessingConfig.APPLY_DESEASONALIZING,
                'reference_port': ProcessingConfig.REFERENCE_PORT,
                'puerto_coords': ProcessingConfig.PUERTO_COORDS
            },
            'datasets_created': {k: v.shape for k, v in results.items()},
            'processing_order': [
                "1. Load and validate daily data",
                "2. Detrend & Deseasonalize (reference point)",
                "3. Enhanced Feature Engineering",
                "4. Merge with WAVERYS data"
            ]
        }
        
        import json
        summary_file = os.path.join(ProcessingConfig.OUTPUT_DIR, f"processing_summary_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Saved processing summary: {summary_file}")
    
    print(f"\n{'='*80}")
    print("PROCESSING PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"Reference port: {reference_port}")
    print(f"Datasets created: {len(results)}")
    for name, df in results.items():
        print(f"  {name}: {df.shape}")
    
    return results

# ============================================================================
# EXECUTION WRAPPER
# ============================================================================

def execute_processing_pipeline(run_path, reference_port, reference_latitude, reference_longitude, base_drive):
    """
    Execute the processing pipeline that reads single-point component files, merges them, and creates enhanced features
    
    Parameters:
    -----------
    run_path : str
        Current run path identifier
    reference_port : str 
        Current reference port name
    reference_latitude : float
        Latitude of reference port
    reference_longitude : float
        Longitude of reference port
    base_drive : str
        Base drive path
    """
    
    # Set file paths for the component files created in Step 1
    swan_out_path = f'{base_drive}/swan/out/{run_path}'
    swan_daily_path = f'{swan_out_path}/df_swan_daily_new_colnames.csv'
    waverys_daily_path = f'{swan_out_path}/df_waverys_daily_new_colnames.csv'
    
    print(f"Reading component files:")
    print(f"  SWAN daily: {swan_daily_path}")
    print(f"  WAVERYS daily: {waverys_daily_path}")
    
    # Read the component files (already single point each)
    df_swan_daily = pd.read_csv(swan_daily_path)
    df_waverys_daily = pd.read_csv(waverys_daily_path)
    
    print(f"✅ Loaded SWAN data: {df_swan_daily.shape}")
    print(f"✅ Loaded WAVERYS data: {df_waverys_daily.shape}")
    
    # MERGE: Combine SWAN and WAVERYS data
    print(f"\n🔗 Merging SWAN and WAVERYS data...")
    
    # Convert date columns to datetime
    df_swan_daily['date'] = pd.to_datetime(df_swan_daily['date'])
    df_waverys_daily['date'] = pd.to_datetime(df_waverys_daily['date'])
    
    # Merge on date (and event_dummy_1 if available in both)
    merge_cols = ['date']
    if 'event_dummy_1' in df_swan_daily.columns and 'event_dummy_1' in df_waverys_daily.columns:
        merge_cols.append('event_dummy_1')
    
    df_final = df_swan_daily.merge(df_waverys_daily, on=merge_cols, how='inner')
    
    print(f"✅ Final merged dataset: {df_final.shape}")
    
    # ENHANCED FEATURE ENGINEERING: Create trends, lags, persistence, etc.
    print(f"\n🛠️ Creating enhanced features (trends, lags, persistence)...")
    
    df_final_enhanced = create_enhanced_features_reference_point(df_final)
    
    print(f"✅ Enhanced features created: {df_final_enhanced.shape}")
    
    # Save the final enhanced dataset
    output_dir = f'{base_drive}/swan/out/{run_path}'
    final_file = f'{output_dir}/df_final_{reference_port}_aggregated.csv'
    
    df_final_enhanced.to_csv(final_file, index=False)
    print(f"✅ Saved enhanced dataset: {final_file}")
    
    return df_final_enhanced

def create_enhanced_features_reference_point(df_final):
    """
    Create enhanced features (trends, lags, persistence, changes) on the merged reference point dataset
    
    Parameters:
    -----------
    df_final : DataFrame
        Merged SWAN + WAVERYS dataset for reference point
    
    Returns:
    --------
    DataFrame with enhanced features added
    """
    
    print(f"  Input shape: {df_final.shape}")
    
    df_enhanced = df_final.copy()
    df_enhanced['date'] = pd.to_datetime(df_enhanced['date'])
    df_enhanced = df_enhanced.sort_values('date').reset_index(drop=True)
    
    # Get base wave features (both _sw and _wa features)
    base_features = [col for col in df_enhanced.columns 
                    if (col.endswith('_sw') or col.endswith('_wa') or col.startswith('pct_'))
                    and col not in ['date', 'event_dummy_1', 'port_name', 'year']]
    
    print(f"  Processing {len(base_features)} base features...")
    
    # Configuration for feature engineering
    PERSISTENCE_WINDOWS = [2, 3, 5, 7, 14]
    TREND_WINDOWS = [3, 5, 7, 14] 
    CHANGE_WINDOWS = [3, 5, 7, 14]
    LAG_WINDOWS = [1, 3, 5, 7, 14]
    
    # Memory-efficient feature creation
    import gc
    
    # 1. Persistence features (rolling averages)
    print("    Creating persistence features...")
    for window in PERSISTENCE_WINDOWS:
        for feature in base_features:
            if feature in df_enhanced.columns:
                col_name = f'{feature}_persistence_{window}'
                df_enhanced[col_name] = df_enhanced[feature].rolling(window, min_periods=1).mean()
        gc.collect()
    
    # 2. Trend features (rolling slopes and means)
    print("    Creating trend features...")
    for window in TREND_WINDOWS:
        # Rolling means (fast)
        for feature in base_features:
            if feature in df_enhanced.columns:
                mean_col = f'{feature}_rolling_mean_{window}'
                df_enhanced[mean_col] = df_enhanced[feature].rolling(window, min_periods=2).mean()
        
        # Rolling slopes (memory-efficient)
        print(f"      Computing slopes for window {window}...")
        
        # Process features in chunks
        chunk_size = 10
        valid_base_features = [f for f in base_features if f in df_enhanced.columns]
        
        for i in range(0, len(valid_base_features), chunk_size):
            chunk_features = valid_base_features[i:i+chunk_size]
            
            for feature in chunk_features:
                trend_col = f'{feature}_trend_{window}'
                
                # Simple slope calculation using numpy
                feature_data = df_enhanced[feature].values
                n_rows = len(feature_data)
                slopes = np.full(n_rows, 0.0)
                
                x = np.arange(window)
                x_mean = x.mean()
                x_centered = x - x_mean
                x_var = np.sum(x_centered ** 2)
                
                for row_idx in range(window-1, n_rows):
                    start_idx = row_idx - window + 1
                    y_window = feature_data[start_idx:row_idx+1]
                    
                    if not np.isnan(y_window).all():
                        y_mean = np.nanmean(y_window)
                        y_centered = y_window - y_mean
                        numerator = np.nansum(x_centered * y_centered)
                        slopes[row_idx] = numerator / x_var if x_var > 0 else 0
                
                df_enhanced[trend_col] = slopes
        
        gc.collect()
    
    # 3. Change features (absolute and relative)
    print("    Creating change features...")
    for window in CHANGE_WINDOWS:
        for feature in base_features:
            if feature in df_enhanced.columns:
                # Absolute change
                abs_change_col = f'{feature}_abs_change_{window}'
                df_enhanced[abs_change_col] = df_enhanced[feature] - df_enhanced[feature].shift(window)
                
                # Relative change
                rel_change_col = f'{feature}_rel_change_{window}'
                past_values = df_enhanced[feature].shift(window)
                df_enhanced[rel_change_col] = np.where(past_values != 0,
                                                      ((df_enhanced[feature] - past_values) / past_values) * 100,
                                                      0)
        gc.collect()
    
    # 4. Lag features
    print("    Creating lag features...")
    
    # Get all newly created feature names
    all_features_to_lag = base_features.copy()
    
    # Add engineered features that exist in the dataframe
    for feature in base_features:
        if feature in df_enhanced.columns:
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
    
    print(f"      Creating lags for {len(all_features_to_lag)} features...")
    
    # Process lags in chunks to manage memory
    for lag in LAG_WINDOWS:
        print(f"        Processing lag {lag}...")
        
        chunk_size = 50
        for i in range(0, len(all_features_to_lag), chunk_size):
            chunk_features = all_features_to_lag[i:i+chunk_size]
            valid_chunk_features = [f for f in chunk_features if f in df_enhanced.columns]
            
            for feature in valid_chunk_features:
                lag_col = f'{feature}_lag_{lag}'
                df_enhanced[lag_col] = df_enhanced[feature].shift(lag)
        
        gc.collect()
    
    # Count new features
    original_features = len(df_final.columns)
    enhanced_features = len(df_enhanced.columns)
    new_features = enhanced_features - original_features
    
    print(f"  Enhanced feature engineering complete:")
    print(f"    Original features: {original_features}")
    print(f"    Enhanced features: {enhanced_features}")
    print(f"    New features created: {new_features}")
    print(f"    Output shape: {df_enhanced.shape}")
    
    return df_enhanced

import re
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Apple Silicon optimizations for interactive environments
os.environ['OPENBLAS_NUM_THREADS'] = '10'  # Adjust based on your CPU cores
os.environ['MKL_NUM_THREADS'] = '10'
os.environ['VECLIB_MAXIMUM_THREADS'] = '10'
os.environ['NUMEXPR_NUM_THREADS'] = '10'

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm.auto import tqdm  # For progress bars in notebooks

def get_top_features_ml(selected_features, pattern, top_n=15):
    """Extract top features matching pattern"""
    filtered = [f for f in selected_features if re.search(pattern, f)]
    if len(filtered) < top_n:
        extras = [f for f in selected_features if f not in filtered]
        filtered += extras[:top_n - len(filtered)]
    return filtered[:top_n]

def get_lagged_feature_names(base, lags, suffix):
    """Generate lagged feature names"""
    return [f"{base}_lag_{i}{suffix}" for i in range(1, lags+1)]

def run_cv_with_threshold_search_interactive(X, y, model_class, model_kwargs=None, 
                                           n_splits=6, model_name='model_ml', verbose=True):
    """CV with threshold search optimized for interactive environments"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics = []
    model_kwargs = model_kwargs or {}
    
    # Use tqdm for progress tracking
    fold_iterator = enumerate(tscv.split(X))
    if verbose:
        fold_iterator = tqdm(fold_iterator, total=n_splits, desc=f"{model_name}")
    
    for fold, (train_idx, test_idx) in fold_iterator:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        
        try:
            base_model = model_class(**model_kwargs)
            # Use fewer CV folds for speed in interactive environment
            model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:,1]
            
            best_f1 = -1
            best_thresh = 0.5
            best_metrics = {}
            
            # Coarser threshold grid for interactive speed
            thresholds = np.arange(0.001, 0.995, 0.001)
            
            for thresh in thresholds:
                y_pred = (y_prob >= thresh).astype(int)
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
                    best_metrics = {
                        'fold': fold+1,
                        'threshold_ml': thresh,
                        'accuracy_ml': acc,
                        'precision_ml': prec,
                        'recall_ml': rec,
                        'f1_ml': f1,
                        'model_ml': model_name,
                        'calibration_ml': 'sigmoid'
                    }
            
            metrics.append(best_metrics)
            
        except Exception as e:
            if verbose:
                print(f"Fold {fold+1} failed: {e}")
            continue
    
    df_metrics = pd.DataFrame(metrics)
    if len(df_metrics) > 0:
        mean_metrics = df_metrics.mean(numeric_only=True).to_dict()
        std_metrics = df_metrics.std(numeric_only=True).to_dict()
        min_metrics = df_metrics.min(numeric_only=True).to_dict()
        max_metrics = df_metrics.max(numeric_only=True).to_dict()
        summary = {k+'_mean_ml': v for k, v in mean_metrics.items()}
        summary.update({k+'_std_ml': v for k, v in std_metrics.items()})
        summary.update({k+'_min_ml': v for k, v in min_metrics.items()})
        summary.update({k+'_max_ml': v for k, v in max_metrics.items()})
        # Add totals for confusion matrix elements (skipping NaNs)
        for cm in ['tp_ml', 'fp_ml', 'tn_ml', 'fn_ml']:
            if cm in df_metrics.columns:
                summary[cm + '_total'] = np.nansum(df_metrics[cm].values)
    else:
        summary = {}
    
    return df_metrics, summary

def run_interactive_ml_pipeline(prediction_data, selected_features, quick_mode=False):
    """
    Interactive ML pipeline optimized for Jupyter/IPython
    
    Parameters:
    -----------
    prediction_data : pd.DataFrame
        Your dataset with features and 'event_dummy_1' target
    selected_features : list
        List of selected feature names
    quick_mode : bool
        If True, uses fewer models and faster settings
    """
    
    print("🚀 Starting Interactive ML Pipeline for Apple Silicon")
    print("=" * 60)
    
    # Check data
    print(f"📊 Dataset shape: {prediction_data.shape}")
    print(f"🎯 Target distribution:")
    print(prediction_data['event_dummy_1'].value_counts())
    print(f"📋 Selected features: {len(selected_features)}")
    
    # Extract feature sets
    print("\n🔍 Extracting feature sets...")
    top15_swan_ml = get_top_features_ml(selected_features, r'_sw$', 15)
    top15_waverys_ml = get_top_features_ml(selected_features, r'_wa$', 15)
    top15_global_ml = selected_features[:15]
    
    print(f"   • SWAN features: {len(top15_swan_ml)}")
    print(f"   • WAVERYS features: {len(top15_waverys_ml)}")
    print(f"   • Global top features: {len(top15_global_ml)}")
    
    # Prepare DataFrames
    swan_df_ml = prediction_data[top15_swan_ml + ['event_dummy_1']].copy()
    waverys_df_ml = prediction_data[top15_waverys_ml + ['event_dummy_1']].copy()
    global_df_ml = prediction_data[top15_global_ml + ['event_dummy_1']].copy()
    
    # Feature sets for specific variables
    swan_features_ml = [
        'swh_p80_sw', 'clima_swh_sw', 'anom_swh_p80_sw'
    ] + get_lagged_feature_names('swh_p80', 2, '_sw')
    
    waverys_features_ml = [
        'swh_p80_wa', 'clima_swh_wa', 'anom_swh_p80_wa'
    ] + get_lagged_feature_names('swh_p80', 2, '_wa')
    
    waverys_swe_features_ml = [
        'swe_p80_wa', 'clima_swe_wa', 'anom_swe_p80_wa'
    ] + get_lagged_feature_names('swe_p80', 2, '_wa')
    
    # Create specific feature DataFrames
    available_swan = [f for f in swan_features_ml if f in prediction_data.columns]
    available_waverys = [f for f in waverys_features_ml if f in prediction_data.columns]
    available_waverys_swe = [f for f in waverys_swe_features_ml if f in prediction_data.columns]
    
    swan_set_df_ml = prediction_data[available_swan + ['event_dummy_1']].copy() if available_swan else None
    waverys_set_df_ml = prediction_data[available_waverys + ['event_dummy_1']].copy() if available_waverys else None
    waverys_swe_set_df_ml = prediction_data[available_waverys_swe + ['event_dummy_1']].copy() if available_waverys_swe else None
    
    # Model variants (optimized for interactive use)
    if quick_mode:
        print("\n⚡ Quick mode: Using fast model subset")
        ml_model_variants = [
            (LogisticRegression, {'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 1000, 'n_jobs': -1}, 'logit_l2'),
            (RandomForestClassifier, {'n_estimators': 50, 'max_depth': 3, 'n_jobs': -1, 'random_state': 42}, 'rf_fast'),
        ]
    else:
        print("\n🔬 Full mode: Using all model variants")
        ml_model_variants = [
            (LogisticRegression, {'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 2000, 'n_jobs': -1}, 'logit_l2'),
            (LogisticRegression, {'penalty': 'l1', 'solver': 'liblinear', 'max_iter': 2000}, 'logit_l1'),
            (LogisticRegression, {'penalty': 'none', 'solver': 'lbfgs', 'max_iter': 2000, 'n_jobs': -1}, 'logit_none'),
            (DecisionTreeClassifier, {'max_depth': 3, 'random_state': 42}, 'tree_depth3'),
            (RandomForestClassifier, {'n_estimators': 100, 'max_depth': 3, 'n_jobs': -1, 'random_state': 42}, 'rf_depth3'),
        ]
    
    # Try to add XGBoost
    try:
        from xgboost import XGBClassifier
        ml_model_variants.append(
            (XGBClassifier, {
                'max_depth': 3, 
                'n_estimators': 100 if not quick_mode else 50,
                'use_label_encoder': False, 
                'eval_metric': 'logloss', 
                'n_jobs': -1,
                'random_state': 42
            }, 'xgb')
        )
        print("   • XGBoost available ✓")
    except ImportError:
        print("   • XGBoost not available (install with: pip install xgboost)")
    
    # Feature sets to evaluate
    ml_feature_sets = [
        (swan_df_ml, top15_swan_ml, 'SWAN'),
        (waverys_df_ml, top15_waverys_ml, 'WAVERYS'),
        (global_df_ml, top15_global_ml, 'GLOBAL'),
    ]
    
    # Add specific feature sets if available
    if swan_set_df_ml is not None and len(available_swan) > 0:
        ml_feature_sets.append((swan_set_df_ml, available_swan, 'SWAN_SET'))
    if waverys_set_df_ml is not None and len(available_waverys) > 0:
        ml_feature_sets.append((waverys_set_df_ml, available_waverys, 'WAVERYS_SET'))
    if waverys_swe_set_df_ml is not None and len(available_waverys_swe) > 0:
        ml_feature_sets.append((waverys_swe_set_df_ml, available_waverys_swe, 'WAVERYS_SWE_SET'))
    
    print(f"\n📈 Will evaluate {len(ml_feature_sets)} feature sets × {len(ml_model_variants)} models = {len(ml_feature_sets) * len(ml_model_variants)} combinations")
    
    # Run experiments
    all_ml_results = []
    start_time = time.time()
    
    total_combinations = len(ml_feature_sets) * len(ml_model_variants)
    combination_count = 0
    
    for df_ml, features_ml, set_name in ml_feature_sets:
        print(f"\n🔄 Processing feature set: {set_name}")
        print(f"   Features: {len(features_ml)}, Samples: {len(df_ml)}")
        
        X = df_ml[features_ml]
        y = df_ml['event_dummy_1']
        
        for model_class, model_kwargs, model_label in ml_model_variants:
            combination_count += 1
            print(f"\n   📊 [{combination_count}/{total_combinations}] {set_name} | {model_label}")
            
            try:
                cv_df, summary = run_cv_with_threshold_search_interactive(
                    X, y, model_class, model_kwargs, n_splits=5, 
                    model_name=f"{set_name}_{model_label}", verbose=False
                )
                
                summary_row = {'feature_set_ml': set_name, 'model_ml': model_label}
                summary_row.update(summary)
                all_ml_results.append(summary_row)
                
                # Show quick results
                if 'f1_ml_mean_ml' in summary:
                    f1_score = summary['f1_ml_mean_ml']
                    f1_std = summary.get('f1_ml_std_ml', 0)
                    print(f"      ✓ F1: {f1_score:.3f} ± {f1_std:.3f}")
                
            except Exception as e:
                print(f"      ✗ Failed: {e}")
                continue
    
    # Create results summary
    ml_results_summary = pd.DataFrame(all_ml_results)
    
    elapsed_time = time.time() - start_time
    print(f"\n🎉 Analysis Complete! Time: {elapsed_time:.1f}s")
    print("=" * 60)
    
    if len(ml_results_summary) > 0:
        # Show top results
        print("\n🏆 TOP 5 MODELS BY F1 SCORE:")
        if 'f1_ml_mean_ml' in ml_results_summary.columns:
            top_models = ml_results_summary.nlargest(5, 'f1_ml_mean_ml')
            for idx, row in top_models.iterrows():
                f1_mean = row['f1_ml_mean_ml']
                f1_std = row.get('f1_ml_std_ml', 0)
                print(f"   {row['feature_set_ml']:15} | {row['model_ml']:12} | F1: {f1_mean:.3f} ± {f1_std:.3f}")
        
        print(f"\n📋 Full results shape: {ml_results_summary.shape}")
        return ml_results_summary
    else:
        print("❌ No successful results!")
        return pd.DataFrame()

# --- Enhanced Feature Selection Methods ---
def enhanced_random_forest_selection(X_train, y_train, top_k=200):
    """Enhanced Random Forest feature selection"""
    print("🌲 Enhanced Random Forest Feature Importance")
    feature_names = X_train.columns.tolist()
    rf_configs = [
        {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5, 'random_state': 42},
        {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 3, 'random_state': 123},
        {'n_estimators': 150, 'max_depth': 12, 'min_samples_split': 8, 'random_state': 456},
        {'n_estimators': 250, 'max_depth': 18, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'random_state': 789},
        {'n_estimators': 180, 'max_depth': None, 'min_samples_split': 10, 'max_features': 0.8, 'random_state': 987}
    ]
    all_importances = []
    for i, config in enumerate(rf_configs):
        print(f"  Running RF config {i+1}/{len(rf_configs)}...")
        try:
            rf = RandomForestClassifier(**config, n_jobs=-1)
            rf.fit(X_train, y_train)
            all_importances.append(rf.feature_importances_)
        except Exception as e:
            print(f"    RF config {i+1} failed: {e}")
    if len(all_importances) == 0:
        return [], pd.DataFrame()
    mean_importances = np.mean(all_importances, axis=0)
    std_importances = np.std(all_importances, axis=0)
    stability_scores = mean_importances / (std_importances + 1e-6)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': mean_importances,
        'importance_std': std_importances,
        'stability': stability_scores,
    })
    importance_df['combined_score'] = (
        0.7 * importance_df['importance_mean'] + 
        0.3 * (importance_df['stability'] / importance_df['stability'].max())
    )
    importance_df = importance_df.sort_values('combined_score', ascending=False)
    selected_features = importance_df.head(top_k)['feature'].tolist()
    print(f"  ✅ Selected {len(selected_features)} features")
    return selected_features, importance_df

def enhanced_extra_trees_selection(X_train, y_train, top_k=200):
    """Enhanced Extra Trees feature selection"""
    print("🌳 Enhanced Extra Trees Feature Importance")
    feature_names = X_train.columns.tolist()
    et_configs = [
        {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 3, 'random_state': 42},
        {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 5, 'random_state': 123},
        {'n_estimators': 150, 'max_depth': None, 'min_samples_leaf': 3, 'random_state': 456},
        {'n_estimators': 250, 'max_depth': 25, 'max_features': 0.7, 'random_state': 789}
    ]
    all_importances = []
    for i, config in enumerate(et_configs):
        print(f"  Running ET config {i+1}/{len(et_configs)}...")
        try:
            et = ExtraTreesClassifier(**config, n_jobs=-1)
            et.fit(X_train, y_train)
            all_importances.append(et.feature_importances_)
        except Exception as e:
            print(f"    ET config {i+1} failed: {e}")
    if len(all_importances) == 0:
        return [], pd.DataFrame()
    mean_importances = np.mean(all_importances, axis=0)
    stability = mean_importances / (np.std(all_importances, axis=0) + 1e-6)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_importances,
        'stability': stability,
        'weighted_score': mean_importances * (1 + 0.2 * stability / stability.max())
    }).sort_values('weighted_score', ascending=False)
    selected_features = importance_df.head(top_k)['feature'].tolist()
    print(f"  ✅ Selected {len(selected_features)} features")
    return selected_features, importance_df

def enhanced_mutual_info_selection(X_train, y_train, top_k=200):
    """Enhanced Mutual Information selection"""
    print("🔗 Enhanced Mutual Information")
    feature_names = X_train.columns.tolist()
    try:
        mi_scores_list = []
        random_states = [42, 123, 456, 789, 987]
        for rs in random_states:
            mi_scores = mutual_info_classif(X_train, y_train, random_state=rs)
            mi_scores_list.append(mi_scores)
        mean_mi_scores = np.mean(mi_scores_list, axis=0)
        std_mi_scores = np.std(mi_scores_list, axis=0)
        stability = mean_mi_scores / (std_mi_scores + 1e-6)
        mi_df = pd.DataFrame({
            'feature': feature_names,
            'mi_score': mean_mi_scores,
            'mi_stability': stability,
            'weighted_mi': mean_mi_scores * (1 + 0.15 * stability / stability.max())
        }).sort_values('weighted_mi', ascending=False)
        selected_features = mi_df.head(top_k)['feature'].tolist()
        print(f"  ✅ Selected {len(selected_features)} features")
        return selected_features, mi_df
    except Exception as e:
        print(f"  ❌ Mutual Information failed: {e}")
        return [], pd.DataFrame()

def enhanced_f_test_selection(X_train, y_train, top_k=200):
    """Enhanced F-Test selection"""
    print("📊 Enhanced F-Test (ANOVA)")
    feature_names = X_train.columns.tolist()
    try:
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X_train, y_train)
        f_scores = selector.scores_
        p_values = selector.pvalues_
        neg_log_p = -np.log(p_values + 1e-10)
        normalized_f = f_scores / (f_scores.max() + 1e-6)
        normalized_p = neg_log_p / (neg_log_p.max() + 1e-6)
        combined_score = 0.7 * normalized_f + 0.3 * normalized_p
        f_df = pd.DataFrame({
            'feature': feature_names,
            'f_score': f_scores,
            'p_value': p_values,
            'combined_f_score': combined_score
        }).sort_values('combined_f_score', ascending=False)
        selected_features = f_df.head(top_k)['feature'].tolist()
        print(f"  ✅ Selected {len(selected_features)} features")
        return selected_features, f_df
    except Exception as e:
        print(f"  ❌ F-Test failed: {e}")
        return [], pd.DataFrame()

def domain_specific_selection(X_train, y_train, top_k=200):
    """Domain-specific feature selection for oceanographic data"""
    print("🌊 Domain-Specific Oceanographic Selection")
    feature_names = X_train.columns.tolist()
    domain_scores = {}
    for feature in feature_names:
        try:
            base_correlation = abs(X_train[feature].fillna(0).corr(y_train))
            if pd.isna(base_correlation):
                base_correlation = 0
            boost_factor = 1.0
            matched_patterns = []
            for pattern_name, (regex, boost) in EnhancedFeatureSelectionConfig.DOMAIN_PATTERNS.items():
                if re.search(regex, feature):
                    boost_factor *= boost
                    matched_patterns.append(pattern_name)
            missing_pct = X_train[feature].isna().mean()
            has_variance = X_train[feature].std() > 1e-6
            quality_penalty = 1.0 if (missing_pct < 0.3 and has_variance) else 0.5
            type_bonus = 1.0
            if '_max_' in feature or '_p9' in feature:
                type_bonus = 1.2
            elif '_trend_' in feature or '_change_' in feature:
                type_bonus = 1.15
            elif '_persistence_' in feature:
                type_bonus = 1.1
            final_score = base_correlation * boost_factor * quality_penalty * type_bonus
            domain_scores[feature] = {
                'base_correlation': base_correlation,
                'boost_factor': boost_factor,
                'quality_penalty': quality_penalty,
                'type_bonus': type_bonus,
                'final_score': final_score,
                'matched_patterns': matched_patterns
            }
        except Exception as e:
            domain_scores[feature] = {
                'base_correlation': 0, 'boost_factor': 1.0, 'quality_penalty': 0,
                'type_bonus': 1.0, 'final_score': 0, 'matched_patterns': []
            }
    domain_df = pd.DataFrame.from_dict(domain_scores, orient='index')
    domain_df['feature'] = domain_df.index
    domain_df = domain_df.sort_values('final_score', ascending=False)
    selected_features = domain_df.head(top_k)['feature'].tolist()
    print(f"  ✅ Selected {len(selected_features)} features")
    return selected_features, domain_df

# --- Enhanced Ensemble Voting for Feature Selection ---
def enhanced_ensemble_voting(method_results, final_top_k=150):
    """Enhanced ensemble voting system"""
    print("🗳️ ENHANCED ENSEMBLE VOTING")
    print("="*60)
    feature_votes = {}
    method_names = ['random_forest', 'extra_trees', 'mutual_info', 'f_test', 'domain_selection']
    # Collect votes with enhanced scoring
    for method_name, (features, scores_df) in zip(method_names, method_results):
        if len(features) == 0:
            continue
        weight = EnhancedFeatureSelectionConfig.METHOD_WEIGHTS.get(method_name, 1.0)
        print(f"📋 {method_name}: {len(features)} features, weight: {weight}")
        for i, feature in enumerate(features):
            position_score = np.exp(-i / len(features)) if len(features) > 0 else 0
            if feature not in feature_votes:
                feature_votes[feature] = {
                    'total_score': 0,
                    'method_count': 0,
                    'methods': [],
                    'positions': [],
                    'best_position': float('inf')
                }
            feature_votes[feature]['total_score'] += position_score * weight
            feature_votes[feature]['method_count'] += 1
            feature_votes[feature]['methods'].append(method_name)
            feature_votes[feature]['positions'].append(i + 1)
            feature_votes[feature]['best_position'] = min(feature_votes[feature]['best_position'], i + 1)
    # Filter by minimum votes
    qualified_features = {}
    for feature, info in feature_votes.items():
        if info['method_count'] >= EnhancedFeatureSelectionConfig.MIN_VOTES:
            if info['method_count'] == EnhancedFeatureSelectionConfig.MIN_VOTES:
                print(f"[DEBUG] Feature '{feature}' just qualified with {info['method_count']} votes.")
            qualified_features[feature] = info
    print(f"✅ {len(qualified_features)} features qualified")
    # Enhanced scoring
    for feature, info in qualified_features.items():
        consensus_score = info['method_count'] / len(method_names)
        quality_score = info['total_score']
        info['enhanced_score'] = quality_score * (1 + 0.2 * consensus_score)
        info['consensus_score'] = consensus_score
    # Sort by enhanced score
    sorted_features = sorted(qualified_features.items(), 
                           key=lambda x: x[1]['enhanced_score'], reverse=True)
    final_features = [feature for feature, info in sorted_features[:final_top_k]]
    # Create summary
    voting_summary = []
    for feature, info in sorted_features[:final_top_k]:
        voting_summary.append({
            'feature': feature,
            'enhanced_score': info['enhanced_score'],
            'method_count': info['method_count'],
            'consensus_score': info['consensus_score'],
            'methods': ', '.join(info['methods']),
            'best_position': info['best_position'],
            'avg_position': np.mean(info['positions'])
        })
    voting_df = pd.DataFrame(voting_summary)
    print(f"🏆 ENHANCED SELECTION: {len(final_features)} features")
    return final_features, voting_df

# --- Enhanced Feature Selection Pipeline ---
def enhanced_feature_selection_pipeline(X_train, y_train, top_k_per_method=200, final_top_k=150):
    """Complete enhanced feature selection pipeline"""
    print("🚀 ENHANCED FEATURE SELECTION PIPELINE")
    print("="*80)
    print(f"Input: {X_train.shape[1]} features, {X_train.shape[0]} samples")
    print(f"Target: Select {final_top_k} features using 5 enhanced methods")
    print("="*80)
    method_results = []
    # Method 1: Enhanced Random Forest
    try:
        rf_features, rf_scores = enhanced_random_forest_selection(X_train, y_train, top_k_per_method)
        method_results.append((rf_features, rf_scores))
    except Exception as e:
        print(f"❌ Enhanced Random Forest failed: {e}")
        method_results.append(([], pd.DataFrame()))
    # Method 2: Enhanced Extra Trees
    try:
        et_features, et_scores = enhanced_extra_trees_selection(X_train, y_train, top_k_per_method)
        method_results.append((et_features, et_scores))
    except Exception as e:
        print(f"❌ Enhanced Extra Trees failed: {e}")
        method_results.append(([], pd.DataFrame()))
    # Method 3: Enhanced Mutual Information
    try:
        mi_features, mi_scores = enhanced_mutual_info_selection(X_train, y_train, top_k_per_method)
        method_results.append((mi_features, mi_scores))
    except Exception as e:
        print(f"❌ Enhanced Mutual Information failed: {e}")
        method_results.append(([], pd.DataFrame()))
    # Method 4: Enhanced F-Test
    try:
        f_features, f_scores = enhanced_f_test_selection(X_train, y_train, top_k_per_method)
        method_results.append((f_features, f_scores))
    except Exception as e:
        print(f"❌ Enhanced F-Test failed: {e}")
        method_results.append(([], pd.DataFrame()))
    # Method 5: Domain-Specific Selection
    try:
        domain_features, domain_scores = domain_specific_selection(X_train, y_train, top_k_per_method)
        method_results.append((domain_features, domain_scores))
    except Exception as e:
        print(f"❌ Domain-Specific Selection failed: {e}")
        method_results.append(([], pd.DataFrame()))
    successful_methods = sum(1 for features, _ in method_results if len(features) > 0)
    if successful_methods == 0:
        print("❌ All enhanced methods failed!")
        return [], pd.DataFrame(), {}
    print(f"\n✅ {successful_methods}/5 enhanced methods completed")
    final_features, voting_summary = enhanced_ensemble_voting(method_results, final_top_k)
    enhanced_report = {
        'total_input_features': X_train.shape[1],
        'successful_methods': successful_methods,
        'final_features_selected': len(final_features),
    }
    print(f"\n🎯 ENHANCED PIPELINE COMPLETE")
    print(f"  📊 Final features selected: {len(final_features)}")
    print(f"  📈 Success rate: {successful_methods}/5 methods")
    return final_features, voting_summary, enhanced_report

# --- Enhanced Rule Combination Creation ---
def create_enhanced_cv_rule_combinations(features, max_combinations=250):
    """Create enhanced rule combinations"""
    print("🔧 Creating enhanced rule combinations...")
    rules = []
    # 1. Single feature rules
    for feature in features:
        rules.append({
            'type': 'single',
            'feature': feature,
            'name': f'{feature} > threshold'
        })
    print(f"✅ Added {len(features)} single-feature rules")
    # 2. Smart feature pairing
    two_feature_count = 0
    max_two_feature = min(80, max_combinations // 3)
    extreme_features = [f for f in features if re.search(r'_max_|_p(85|90|95)', f)][:8]
    temporal_features = [f for f in features if re.search(r'_trend_|_persistence_|_change_', f)][:8]
    anomaly_features = [f for f in features if 'anom_' in f][:5]
    source_features = [f for f in features if re.search(r'_sw$|_wa', f)][:8]
    pairings = [
        (extreme_features, temporal_features, "extreme + temporal"),
        (extreme_features, anomaly_features, "extreme + anomaly"),
        (source_features[:4], temporal_features[:4], "source + temporal"),
        (anomaly_features, temporal_features[:5], "anomaly + temporal")
    ]
    for group1, group2, description in pairings:
        for f1 in group1:
            for f2 in group2:
                if two_feature_count >= max_two_feature:
                    break
                if f1 != f2:
                    # AND rule
                    rules.append({
                        'type': 'and_2',
                        'feature1': f1,
                        'feature2': f2,
                        'name': f'{f1} > t1 AND {f2} > t2'
                    })
                    two_feature_count += 1
                    # OR rule (fewer of these)
                    if two_feature_count < max_two_feature and np.random.random() < 0.3:
                        rules.append({
                            'type': 'or_2',
                            'feature1': f1,
                            'feature2': f2,
                            'name': f'{f1} > t1 OR {f2} > t2'
                        })
                        two_feature_count += 1
    print(f"✅ Added {two_feature_count} two-feature rules")
    # 3. Limited three-feature combinations
    three_feature_count = 0
    max_three_feature = min(25, max_combinations // 8)
    top_features = features[:12]
    for i in range(0, min(9, len(top_features)), 3):
        if three_feature_count >= max_three_feature:
            break
        if i+2 < len(top_features):
            f1, f2, f3 = top_features[i], top_features[i+1], top_features[i+2]
            rules.append({
                'type': 'majority_3',
                'feature1': f1,
                'feature2': f2,
                'feature3': f3,
                'name': f'Majority of {f1}, {f2}, {f3}'
            })
            three_feature_count += 1
    print(f"✅ Added {three_feature_count} three-feature rules")
    print(f"🎯 Total enhanced rules: {len(rules)}")
    return rules

# --- Enhanced Rule CV Evaluation Helper Functions ---

def find_optimal_threshold(df_train, df_test, feature):
    """Find threshold that maximizes F1 score for a feature"""
    if feature not in df_test.columns or feature not in df_train.columns:
        return None
    train_feature = df_train[feature].dropna()
    test_feature = df_test[feature].dropna()
    if len(train_feature) < 5 or len(test_feature) < 5:
        return None
    if train_feature.std() < 1e-6:
        return None
    percentiles = list(range(10, 95, 5))  # 10% to 90% by 5%
    try:
        thresholds = [np.percentile(train_feature, p) for p in percentiles]
        thresholds = sorted(list(set(thresholds)))
    except:
        return None
    if len(thresholds) < 2:
        return None
    best_f1 = -1
    best_threshold = None
    y_true = df_test['event_dummy_1']
    valid_predictions = 0
    for threshold in thresholds:
        try:
            y_pred = (df_test[feature] > threshold).astype(int)
            if y_pred.sum() == 0:
                continue
            accuracy = accuracy_score(y_true, y_pred)
            if accuracy > 0.3:  # More lenient
                f1 = f1_score(y_true, y_pred, zero_division=0)
                valid_predictions += 1
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
        except:
            continue
    if best_threshold is None and len(thresholds) > 0:
        print(f"    DEBUG: Feature {feature[:30]}... - {len(thresholds)} thresholds tested, {valid_predictions} valid predictions, best_f1={best_f1:.3f}")
    return best_threshold

def evaluate_rule_cv_with_thresholds(rule, df, cv_splits):
    """CV evaluation with threshold tracking"""
    fold_results = []
    fold_thresholds = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        df_train_fold = df.iloc[train_idx]
        df_test_fold = df.iloc[test_idx]
        try:
            thresholds_used = {}
            if rule['type'] == 'single':
                threshold = find_optimal_threshold(df_train_fold, df_test_fold, rule['feature'])
                if threshold is None:
                    print(f"[DEBUG] Skipping fold {fold_idx} for rule {rule['name']} (no valid threshold found for {rule['feature']})")
                    continue
                thresholds_used[rule['feature']] = threshold
                y_pred_series = apply_single_condition(df_test_fold, rule['feature'], threshold)
            elif rule['type'] == 'and_2':
                threshold1 = find_optimal_threshold(df_train_fold, df_test_fold, rule['feature1'])
                threshold2 = find_optimal_threshold(df_train_fold, df_test_fold, rule['feature2'])
                if threshold1 is None or threshold2 is None:
                    continue
                thresholds_used[rule['feature1']] = threshold1
                thresholds_used[rule['feature2']] = threshold2
                y_pred_series = apply_and_condition(df_test_fold, rule['feature1'], threshold1, rule['feature2'], threshold2)
            elif rule['type'] == 'or_2':
                threshold1 = find_optimal_threshold(df_train_fold, df_test_fold, rule['feature1'])
                threshold2 = find_optimal_threshold(df_train_fold, df_test_fold, rule['feature2'])
                if threshold1 is None or threshold2 is None:
                    continue
                thresholds_used[rule['feature1']] = threshold1
                thresholds_used[rule['feature2']] = threshold2
                y_pred_series = apply_or_condition(df_test_fold, rule['feature1'], threshold1, rule['feature2'], threshold2)
            elif rule['type'] == 'majority_3':
                threshold1 = find_optimal_threshold(df_train_fold, df_test_fold, rule['feature1'])
                threshold2 = find_optimal_threshold(df_train_fold, df_test_fold, rule['feature2'])
                threshold3 = find_optimal_threshold(df_train_fold, df_test_fold, rule['feature3'])
                if any(t is None for t in [threshold1, threshold2, threshold3]):
                    print(f"[DEBUG] Skipping fold {fold_idx} for rule {rule['name']} (no valid threshold for one of {rule['feature1']}, {rule['feature2']}, {rule['feature3']})")
                    continue
                thresholds_used[rule['feature1']] = threshold1
                thresholds_used[rule['feature2']] = threshold2
                thresholds_used[rule['feature3']] = threshold3
                y_pred_series = apply_majority_condition(df_test_fold, rule['feature1'], threshold1, rule['feature2'], threshold2, rule['feature3'], threshold3)
            else:
                continue
            # Calculate metrics
            y_pred = y_pred_series.astype(int)
            y_true = df_test_fold['event_dummy_1']
            if len(y_true) > 0:
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                try:
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                except:
                    tn, fp, fn, tp = 0, 0, 0, 0
                fold_results.append({
                    'fold': fold_idx,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'accuracy': accuracy,
                    'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
                })
                fold_thresholds.append({
                    'fold': fold_idx,
                    'rule_type': rule['type'],
                    'rule_name': rule['name'],
                    'f1_score': f1,
                    **thresholds_used
                })
        except Exception as e:
            print(f"[DEBUG] Exception in fold {fold_idx} for rule {rule['name']}: {e}")
            continue
    # Aggregate results across folds
    if len(fold_results) == 0:
        return None, []
    fold_df = pd.DataFrame(fold_results)
    cv_result = {
        'rule_type': rule['type'],
        'rule_name': rule['name'],
        'folds': len(fold_results),
        'f1_mean': fold_df['f1_score'].mean(),
        'f1_std': fold_df['f1_score'].std(),
        'precision_mean': fold_df['precision'].mean(),
        'recall_mean': fold_df['recall'].mean(),
        'accuracy_mean': fold_df['accuracy'].mean(),
        'tp_total': fold_df['tp'].sum(),
        'fp_total': fold_df['fp'].sum(),
        'tn_total': fold_df['tn'].sum(),
        'fn_total': fold_df['fn'].sum(),
        'n_folds_successful': len(fold_results),
    }
    return cv_result, fold_thresholds

def apply_single_condition(df, feature, threshold):
    if feature not in df.columns or threshold is None:
        return pd.Series(False, index=df.index)
    return df[feature] > threshold

def apply_and_condition(df, feature1, threshold1, feature2, threshold2):
    if any(f not in df.columns for f in [feature1, feature2]):
        return pd.Series(False, index=df.index)
    if threshold1 is None or threshold2 is None:
        return pd.Series(False, index=df.index)
    return (df[feature1] > threshold1) & (df[feature2] > threshold2)

def apply_or_condition(df, feature1, threshold1, feature2, threshold2):
    if any(f not in df.columns for f in [feature1, feature2]):
        return pd.Series(False, index=df.index)
    if threshold1 is None or threshold2 is None:
        return pd.Series(False, index=df.index)
    return (df[feature1] > threshold1) | (df[feature2] > threshold2)

def apply_majority_condition(df, feature1, threshold1, feature2, threshold2, feature3, threshold3):
    if any(f not in df.columns for f in [feature1, feature2, feature3]):
        return pd.Series(False, index=df.index)
    if any(t is None for t in [threshold1, threshold2, threshold3]):
        return pd.Series(False, index=df.index)
    cond1 = (df[feature1] > threshold1).astype(int)
    cond2 = (df[feature2] > threshold2).astype(int)
    cond3 = (df[feature3] > threshold3).astype(int)
    return (cond1 + cond2 + cond3) >= 2

# --- Enhanced Rule CV Evaluation with Threshold Tracking ---
def run_cv_evaluation_with_threshold_tracking(rules, df, cv_splits):
    print("🚀 Starting CV...")
    results = []
    all_fold_thresholds = []
    best_f1_mean = 0
    for i, rule in enumerate(rules):
        if i % 20 == 0:
            print(f"📊 Progress: {i}/{len(rules)} rules evaluated (Best mean F1: {best_f1_mean:.3f})")
        result, fold_thresholds = evaluate_rule_cv_with_thresholds(rule, df, cv_splits)
        if result is not None and result['n_folds_successful'] >= 3:
            results.append(result)
            all_fold_thresholds.extend(fold_thresholds)
            if result['f1_mean'] > best_f1_mean:
                best_f1_mean = result['f1_mean']
                print(f"  🎯 New best mean F1: {best_f1_mean:.3f} (±{result['f1_std']:.3f}) - {result['rule_name'][:50]}...")
        else:
            if result is None:
                print(f"[DEBUG] Rule {rule['name']} excluded: no valid folds (all folds skipped or errored)")
            elif result['n_folds_successful'] < 3:
                print(f"[DEBUG] Rule {rule['name']} excluded: only {result['n_folds_successful']} valid folds (needs >= 3)")
            elif result['f1_mean'] == 0:
                print(f"[DEBUG] Rule {rule['name']} excluded: mean F1=0 over {result['n_folds_successful']} folds")
    return results, all_fold_thresholds

# --- Enhanced Rule Evaluation Pipeline ---
def run_enhanced_cv_pipeline_fast(prediction_data):
    """
    FAST enhanced CV pipeline for port closure prediction.
    Pipeline order:
      1. Enhanced Feature Selection
      2. Rule Generation
      3. Rule Evaluation (CV)
    Returns:
      cv_df, stable_rules, selected_features, all_fold_thresholds
    """
    print("🚀 FAST ENHANCED CV PIPELINE")
    print("="*80)
    # --- STEP 1: Prepare data ---
    df = prediction_data.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"📊 Dataset: {len(df)} samples, {df['event_dummy_1'].sum()} events ({df['event_dummy_1'].mean()*100:.1f}%)")
    exclude_cols = ['date', 'port_name', 'event_dummy_1', 'total_obs']
    # Configuration
    N_FOLDS = 6
    USE_TIME_SERIES_CV = True
    TOP_K_FEATURES = 450
    MAX_COMBINATIONS = 900
    # Create CV splits
    print(f"\n{'='*60}")
    print("CREATING CROSS-VALIDATION SPLITS")
    print("="*60)
    if USE_TIME_SERIES_CV:
        tscv = TimeSeriesSplit(n_splits=N_FOLDS)
        cv_splits = list(tscv.split(df))
        print(f"📅 Using TimeSeriesSplit with {N_FOLDS} folds")
    else:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        cv_splits = list(skf.split(df, df['event_dummy_1']))
        print(f"🎯 Using StratifiedKFold with {N_FOLDS} folds")
    # In the enhanced pipeline, before feature selection:
    print("🔍 FEATURE COMPARISON DEBUG:")
    print(f"Total columns in dataset: {len(df.columns)}")
    print(f"Excluded columns: {exclude_cols}")
    # Show what we're actually using
    actual_features = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
    print(f"Features being used: {len(actual_features)}")
    # Show first 20 features
    print("First 20 features:")
    for i, feat in enumerate(actual_features[:20]):
        print(f"  {i+1:2d}. {feat}")
    # --- STEP 2: FEATURE SELECTION ---
    print(f"\n{'='*60}")
    print("ENHANCED FEATURE SELECTION (SINGLE RUN)")
    print("="*60)
    try:
        # Use larger training set for feature selection
        combined_train_idx = []
        for i in range(min(3, len(cv_splits))):
            train_idx, _ = cv_splits[i]
            combined_train_idx.extend(train_idx)
        combined_train_idx = list(set(combined_train_idx))
        df_train_combined = df.iloc[combined_train_idx]
        print(f"📊 Using {len(df_train_combined)} samples for feature selection...")
        # Prepare features
        exclude_cols = [
            'date', 'port_name', 'event_dummy_1', 'total_obs',
            'duracion', 'year', 'latitude', 'longitude'
        ]
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols 
                       and df[col].dtype in ['float64', 'int64']]
        X_train = df_train_combined[feature_cols].fillna(0)
        y_train = df_train_combined['event_dummy_1']
        print(f"📊 Evaluating {len(feature_cols)} candidate features...")
        # Run enhanced feature selection ONCE
        selected_features, voting_summary, report = enhanced_feature_selection_pipeline(
            X_train, y_train, 
            top_k_per_method=350,
            final_top_k=TOP_K_FEATURES
        )
        print(f"✅ Enhanced feature selection complete: {len(selected_features)} features")
        # Check if the best original features made it through selection
        best_original_features = ['swh_median_wa', 'swh_p80_wa', 'swh_max_wa', 'anom_swh_p25_sw']
        print("🔍 Did best original features survive selection?")
        for feat in best_original_features:
            if feat in selected_features:
                print(f"  ✅ {feat} - SELECTED")
            else:
                print(f"  ❌ {feat} - EXCLUDED!")
    except Exception as e:
        print(f"❌ Enhanced feature selection failed: {e}")
        print("Falling back to quick correlation selection...")
        # Quick fallback
        exclude_cols = [
            'date', 'port_name', 'event_dummy_1', 'total_obs',
            'duracion', 'year', 'latitude', 'longitude'
        ]
        all_features = [col for col in df.columns 
                       if col not in exclude_cols 
                       and df[col].dtype in ['float64', 'int64']]
        # Domain boost patterns for quick selection
        boost_patterns = {
            'max_waves': (r'swh_max_sw|swe_max_sw|swh_max_wa|swe_max_wa', 4.0),
            'swan_features': (r'_sw', 3.5),
            'high_percentiles': (r'_p(80|90|95)', 3.0),
            'trends': (r'_trend_(3|5|7|14)', 2.5),
            'persistence': (r'_persistence_(5|7|14)', 2.3),
            'anomalies': (r'anom_', 3.5),
            'volatility': (r'_iqr|_std|_cv', 2.5)
        }
        feature_scores = []
        y = df['event_dummy_1']
        for feature in all_features:
            try:
                corr = abs(df[feature].fillna(0).corr(y))
                if pd.isna(corr):
                    continue
                boost = 1.0
                for pattern_name, (regex, boost_factor) in boost_patterns.items():
                    if re.search(regex, feature):
                        boost *= boost_factor
                final_score = corr * boost
                feature_scores.append((feature, final_score))
            except:
                continue
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        selected_features = [f for f, _ in feature_scores[:TOP_K_FEATURES]]
        print(f"✅ Quick selection: {len(selected_features)} features")
    # --- STEP 3: RULE GENERATION ---
    print(f"\n{'='*60}")
    print("CREATING RULE COMBINATIONS")
    print("="*60)
    rules = create_enhanced_cv_rule_combinations(selected_features, MAX_COMBINATIONS)
    print(f"✅ Created {len(rules)} rules from selected features")

    # --- STEP 4: RULE EVALUATION (CROSS-VALIDATION) ---
    print(f"\n{'='*60}")
    print("RUNNING CROSS-VALIDATION EVALUATION")
    print("="*60)
    print(f"⏱️ Evaluating {len(rules)} rules across {N_FOLDS} folds...")
    cv_results, all_fold_thresholds = run_cv_evaluation_with_threshold_tracking(
        rules, df, cv_splits
    )
    # --- STEP 5: PROCESS RULE EVALUATION RESULTS ---
    if len(cv_results) == 0:
        print("❌ No valid rules found!")
        return pd.DataFrame(), pd.DataFrame(), [], []
    cv_df = pd.DataFrame(cv_results)
    cv_df = cv_df.sort_values('f1_mean', ascending=False)
    print(f"✅ Evaluated {len(cv_df)} rules successfully")
    print(f"📊 Best F1 score: {cv_df['f1_mean'].max():.3f}")
    # Get stable high-performers
    TARGET_F1 = 0.6
    stable_rules = cv_df[
        (cv_df['f1_mean'] >= TARGET_F1) & 
        (cv_df['f1_std'] <= 0.2) &
        (cv_df['n_folds_successful'] >= 4)
    ].copy()
    if len(stable_rules) == 0:
        stable_rules = cv_df.head(10).copy()
    print(f"\n✅ FAST ENHANCED PIPELINE COMPLETE")
    print(f"📊 Final results: {len(cv_df)} rules evaluated, {len(stable_rules)} stable rules")
    return cv_df, stable_rules, selected_features, all_fold_thresholds

# Convenience functions for interactive use
def quick_ml_analysis(prediction_data, selected_features):
    """Quick analysis with fast models only"""
    return run_interactive_ml_pipeline(prediction_data, selected_features, quick_mode=True)

def full_ml_analysis(prediction_data, selected_features):
    """Full analysis with all models"""
    return run_interactive_ml_pipeline(prediction_data, selected_features, quick_mode=False)


def save_best_lr_predictions_2024(prediction_data, selected_features, output_path='best_lr_predictions_2024.csv'):
    """
    Find the best logistic regression model and save predictions for 2024 data
    
    Parameters:
    -----------
    prediction_data : pd.DataFrame
        Your full dataset with date column and features
    selected_features : list
        List of selected feature names
    output_path : str
        Path to save the CSV file
    
    Returns:
    --------
    pd.DataFrame with date, estimated_probability, calibrated_probability
    """
    
    print("🔍 Finding Best Logistic Regression Model for 2024 Predictions")
    print("=" * 60)
    
    # Ensure date column is datetime
    if 'date' in prediction_data.columns:
        prediction_data['date'] = pd.to_datetime(prediction_data['date'])
    else:
        print("ERROR: 'date' column not found in prediction_data")
        return None
    
    # Filter for 2024 data
    data_2024 = prediction_data[prediction_data['date'].dt.year == 2024].copy()
    if len(data_2024) == 0:
        print("ERROR: No 2024 data found")
        return None
    
    print(f"📊 Dataset info:")
    print(f"   Total data: {len(prediction_data)} rows")
    print(f"   2024 data: {len(data_2024)} rows")
    print(f"   Date range 2024: {data_2024['date'].min()} to {data_2024['date'].max()}")
    print(f"   Selected features: {len(selected_features)}")
    
    if 'event_dummy_1' not in prediction_data.columns:
        print("ERROR: Target variable 'event_dummy_1' not found")
        return None
    
    # Prepare feature sets to test
    feature_sets = {
        'SWAN': [f for f in selected_features if f.endswith('_sw')][:15],
        'WAVERYS': [f for f in selected_features if f.endswith('_wa')][:15],
        'GLOBAL': selected_features[:15]
    }
    
    # Add specific feature sets if they exist
    swan_specific = ['swh_p80_sw', 'clima_swh_sw', 'anom_swh_p80_sw'] + \
                   [f'swh_p80_lag_{i}_sw' for i in range(1, 3)]
    waverys_specific = ['swh_p80_wa', 'clima_swh_wa', 'anom_swh_p80_wa'] + \
                      [f'swh_p80_lag_{i}_wa' for i in range(1, 3)]
    
    if all(f in prediction_data.columns for f in swan_specific):
        feature_sets['SWAN_SPECIFIC'] = swan_specific
    
    if all(f in prediction_data.columns for f in waverys_specific):
        feature_sets['WAVERYS_SPECIFIC'] = waverys_specific
    
    print(f"🎯 Testing {len(feature_sets)} feature sets")
    
    # Logistic regression variants to test
    lr_variants = {
        'L2': {'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 2000, 'C': 1.0},
        'L1': {'penalty': 'l1', 'solver': 'liblinear', 'max_iter': 2000, 'C': 1.0},
        'L2_Strong': {'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 2000, 'C': 0.1},
        'L2_Weak': {'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 2000, 'C': 10.0}
    }
    
    print(f"🧪 Testing {len(lr_variants)} logistic regression variants")
    
    # Find best model through cross-validation
    best_f1 = -1
    best_model_info = None
    all_results = []
    
    for feature_name, features in feature_sets.items():
        # Filter features that actually exist
        available_features = [f for f in features if f in prediction_data.columns]
        if len(available_features) == 0:
            print(f"   ⚠️  {feature_name}: No features available")
            continue
            
        print(f"\n📈 Testing {feature_name} ({len(available_features)} features)")
        
        # Prepare data
        X = prediction_data[available_features].fillna(0)
        y = prediction_data['event_dummy_1']
        from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix

        for lr_name, lr_params in lr_variants.items():
            print(f"   🔬 {feature_name} | {lr_name}")
            
            try:
                # Cross-validation to find best threshold
                tscv = TimeSeriesSplit(n_splits=5)
                cv_f1_scores = []
                cv_thresholds = []
                successful_folds = 0
                
                for train_idx, val_idx in tscv.split(X):
                    try:
                        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                        
                        # Check if both classes are present in training data
                        if len(np.unique(y_train)) < 2:
                            print(f"      ⚠️  Fold skipped: only one class in training data")
                            continue
                            
                        # Check if validation set has any positive cases for meaningful F1
                        if y_val.sum() == 0:
                            print(f"      ⚠️  Fold skipped: no positive cases in validation")
                            continue
                        
                        # Train and calibrate model
                        base_model = LogisticRegression(**lr_params)
                        calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=2)
                        calibrated_model.fit(X_train, y_train)
                        
                        # Get probabilities
                        y_prob = calibrated_model.predict_proba(X_val)[:, 1]
                        
                        # Find best threshold with constraint that both precision and recall > 0
                        best_thresh = 0.5
                        best_val_f1 = 0
                        found_valid_threshold = False
                        
                        # First pass: find thresholds where both precision and recall > 0
                        valid_thresholds = []
                        for thresh in np.arange(0.01, 0.99, 0.001):  # Finer grid, starting lower
                            y_pred = (y_prob >= thresh).astype(int)
                            if y_pred.sum() == 0:  # No positive predictions
                                continue
                            val_precision = precision_score(y_val, y_pred, zero_division=0)
                            val_recall = recall_score(y_val, y_pred, zero_division=0)
                            val_f1 = f1_score(y_val, y_pred, zero_division=0)
                            
                            # Only consider thresholds where both precision and recall > 0
                            if val_precision > 0 and val_recall > 0:
                                valid_thresholds.append((thresh, val_f1, val_precision, val_recall))
                        
                        # Select best F1 among valid thresholds
                        if valid_thresholds:
                            best_thresh, best_val_f1, best_prec, best_rec = max(valid_thresholds, key=lambda x: x[1])
                            found_valid_threshold = True
                        else:
                            # Fallback: find threshold that maximizes recall (at least catch some positives)
                            best_recall = 0
                            for thresh in np.arange(0.01, 0.99, 0.01):
                                y_pred = (y_prob >= thresh).astype(int)
                                if y_pred.sum() == 0:
                                    continue
                                val_recall = recall_score(y_val, y_pred, zero_division=0)
                                if val_recall > best_recall:
                                    best_recall = val_recall
                                    best_thresh = thresh
                                    best_val_f1 = f1_score(y_val, y_pred, zero_division=0)
                        
                        cv_f1_scores.append(best_val_f1)
                        cv_thresholds.append(best_thresh)
                        successful_folds += 1
                        
                        # Print fold details for debugging
                        status = "✓ Valid" if found_valid_threshold else "⚠ Fallback"
                        print(f"         Fold {successful_folds}: {status} - Thresh={best_thresh:.3f}, F1={best_val_f1:.3f}")
                        
                    except Exception as fold_e:
                        print(f"      ⚠️  Fold failed: {fold_e}")
                        continue
                
                # Only proceed if we have at least 2 successful folds
                if successful_folds < 2:
                    print(f"      ❌ Insufficient successful folds ({successful_folds}/5)")
                    continue
                
                # Average CV performance
                mean_f1 = np.mean(cv_f1_scores)
                mean_threshold = np.mean(cv_thresholds)
                
                result = {
                    'feature_set': feature_name,
                    'lr_variant': lr_name,
                    'mean_f1': mean_f1,
                    'std_f1': np.std(cv_f1_scores),
                    'mean_threshold': mean_threshold,
                    'features': available_features,
                    'lr_params': lr_params
                }
                all_results.append(result)
                
                print(f"      ✅ F1: {mean_f1:.3f} ± {np.std(cv_f1_scores):.3f}, Threshold: {mean_threshold:.3f} ({successful_folds} folds)")
                
                # Track best model
                if mean_f1 > best_f1:
                    best_f1 = mean_f1
                    best_model_info = result
                    
            except Exception as e:
                print(f"      ❌ Failed: {e}")
                continue
    
    if best_model_info is None:
        print("ERROR: No successful models found")
        return None
    
    print(f"\n🏆 BEST MODEL FOUND:")
    print(f"   Feature Set: {best_model_info['feature_set']}")
    print(f"   LR Variant: {best_model_info['lr_variant']}")
    print(f"   CV F1 Score: {best_model_info['mean_f1']:.3f} ± {best_model_info['std_f1']:.3f}")
    print(f"   Optimal Threshold: {best_model_info['mean_threshold']:.3f}")
    print(f"   Features: {len(best_model_info['features'])}")
    
    # Train final model on all training data (pre-2024)
    print(f"\n🚀 Training final model on all pre-2024 data...")
    
    train_data = prediction_data[prediction_data['date'].dt.year < 2024]
    if len(train_data) == 0:
        print("WARNING: No pre-2024 training data, using all data for training")
        train_data = prediction_data
    
    print(f"   Training data: {len(train_data)} rows")
    
    # Check if training data has both classes
    train_target = train_data['event_dummy_1']
    print(f"   Training target distribution: {train_target.value_counts().to_dict()}")
    
    if len(np.unique(train_target)) < 2:
        print("ERROR: Training data contains only one class - cannot train classifier")
        return None
    
    # Prepare training data
    X_train = train_data[best_model_info['features']].fillna(0)
    y_train = train_data['event_dummy_1']
    
    try:
        # Train final calibrated model
        final_base_model = LogisticRegression(**best_model_info['lr_params'])
        final_model = CalibratedClassifierCV(final_base_model, method='sigmoid', cv=3)
        final_model.fit(X_train, y_train)
        
        print(f"   ✅ Final model trained successfully")
    except Exception as e:
        print(f"ERROR: Failed to train final model: {e}")
        return None
    
    # Generate predictions for 2024
    print(f"\n🔮 Generating predictions for 2024...")
    
    X_2024 = data_2024[best_model_info['features']].fillna(0)
    
    try:
        # Get both raw and calibrated probabilities
        raw_probabilities = final_base_model.fit(X_train, y_train).predict_proba(X_2024)[:, 1]
        calibrated_probabilities = final_model.predict_proba(X_2024)[:, 1]
        
        print(f"   ✅ Predictions generated for {len(X_2024)} days")
    except Exception as e:
        print(f"ERROR: Failed to generate predictions: {e}")
        return None
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'date': data_2024['date'],
        'estimated_probability': raw_probabilities,
        'calibrated_probability': calibrated_probabilities,
        'optimal_threshold': best_model_info['mean_threshold'],
        'predicted_event': (calibrated_probabilities >= best_model_info['mean_threshold']).astype(int),
        'observed_event': data_2024['event_dummy_1'].values  # Always include observed events
    })

    # Print confusion matrix for ML predictions using optimal threshold
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(results_df['observed_event'], results_df['predicted_event'])
    print(f"\n📊 2024 Confusion Matrix (Optimal Threshold={best_model_info['mean_threshold']:.3f}):")
    print(f"                 Predicted")
    print(f"                 0    1")
    print(f"   Actual   0   {cm[0,0]:3d}  {cm[0,1]:3d}")
    print(f"            1   {cm[1,0]:3d}  {cm[1,1]:3d}")
    print(f"   TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

    # Optionally, save the optimal threshold for downstream use
    output_threshold_path = output_path.replace('.csv', '_optimal_threshold.txt')
    with open(output_threshold_path, 'w') as f:
        f.write(str(best_model_info['mean_threshold']))
    print(f"Optimal threshold saved to: {output_threshold_path}")
    
    # Calculate performance metrics if observed events are available
    if 'event_dummy_1' in data_2024.columns:
        
        # Calculate 2024 performance metrics
        y_pred_2024 = results_df['predicted_event']
        y_true_2024 = results_df['observed_event']
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        accuracy = accuracy_score(y_true_2024, y_pred_2024)
        precision = precision_score(y_true_2024, y_pred_2024, zero_division=0)
        recall = recall_score(y_true_2024, y_pred_2024, zero_division=0)
        f1_2024 = f1_score(y_true_2024, y_pred_2024, zero_division=0)
        
        print(f"\n📊 2024 Performance Metrics:")
        print(f"   Accuracy:  {accuracy:.3f}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall:    {recall:.3f}")
        print(f"   F1 Score:  {f1_2024:.3f}")
        print(f"   Predicted Events: {y_pred_2024.sum()}/{len(y_pred_2024)}")
        print(f"   Actual Events: {y_true_2024.sum()}/{len(y_true_2024)}")
    
    # Add model metadata
    results_df['feature_set'] = best_model_info['feature_set']
    results_df['lr_variant'] = best_model_info['lr_variant']
    results_df['cv_f1_score'] = best_model_info['mean_f1']
    
    # Save to disk
    results_df.to_csv(output_path, index=False)
    print(f"\n💾 Results saved to: {output_path}")
    print(f"   Shape: {results_df.shape}")
    print(f"   Columns: {list(results_df.columns)}")
    
    # Save model info separately
    model_info_path = output_path.replace('.csv', '_model_info.txt')
    with open(model_info_path, 'w') as f:
        f.write("BEST LOGISTIC REGRESSION MODEL INFO\n")
        f.write("="*40 + "\n\n")
        f.write(f"Feature Set: {best_model_info['feature_set']}\n")
        f.write(f"LR Variant: {best_model_info['lr_variant']}\n")
        f.write(f"CV F1 Score: {best_model_info['mean_f1']:.3f} ± {best_model_info['std_f1']:.3f}\n")
        f.write(f"Optimal Threshold: {best_model_info['mean_threshold']:.3f}\n")
        f.write(f"LR Parameters: {best_model_info['lr_params']}\n")
        f.write(f"Features ({len(best_model_info['features'])}):\n")
        for feature in best_model_info['features']:
            f.write(f"  - {feature}\n")
    
    print(f"   Model info saved to: {model_info_path}")
    
    # Display sample of results
    print(f"\n📋 Sample Results:")
    print(results_df[['date', 'estimated_probability', 'calibrated_probability', 'predicted_event', 'observed_event']].head(10))
    
    return results_df

# Quick usage function
def quick_save_lr_2024(prediction_data, selected_features, filename=None):
    """Quick function to save best LR predictions for 2024"""
    if filename is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"best_lr_predictions_2024_{timestamp}.csv"
    
    return save_best_lr_predictions_2024(prediction_data, selected_features, filename)


# --- MAIN EXECUTION ---
def main():
    print("\n🔧 RULE_EVALUATION.PY - CV Pipeline")
    print(f"Run: {config.RUN_PATH}")
    print(f"Reference port: {config.reference_port}")

    # Get input and output paths
    input_files = get_input_files()
    output_files = get_output_files()

    # Load merged features file (output of data_preparation_1.py)
    merged_path = os.path.join(config.run_output_dir, 'df_swan_waverys_merged.csv')
    print(f"🔎 Looking for merged features at: {merged_path}")
    if not os.path.exists(merged_path):
        print(f"❌ Input file not found: {merged_path}")
        return
    df = pd.read_csv(merged_path)
    print(f"✅ Loaded merged features: {merged_path} ({df.shape})")

    # --- STEP 1: Feature Selection and Rule Evaluation ---
# ============================================================================
    print("\n🚦 Running enhanced rule evaluation pipeline ...")
    cv_df, stable_rules, selected_features_rule, all_fold_thresholds = run_enhanced_cv_pipeline_fast(df)
    # Define output paths
    rule_cv_path = os.path.join(config.results_output_dir, 'rule_cv_results.csv')
    stable_rules_path = os.path.join(config.results_output_dir, 'stable_rules.csv')

    # Ensure output directory exists
    os.makedirs(config.results_output_dir, exist_ok=True)

    if not cv_df.empty:
        cv_df.to_csv(rule_cv_path, index=False)
        print(f"✅ Rule CV results saved: {rule_cv_path}")

    # Save per-fold thresholds for compatibility with AEP
    fold_thresholds_path = os.path.join(config.results_output_dir, f'fold_thresholds_{config.RUN_PATH}.csv')
    if all_fold_thresholds:
        pd.DataFrame(all_fold_thresholds).to_csv(fold_thresholds_path, index=False)
        print(f"✅ Fold thresholds saved: {fold_thresholds_path}")
    else:
        print("⚠️ No fold thresholds to save.")

    if isinstance(stable_rules, pd.DataFrame) and not stable_rules.empty:
        stable_rules.to_csv(stable_rules_path, index=False)
        print(f"✅ Stable rules saved: {stable_rules_path}")
    print("\n🚦 Running and saving ML probability predictions for 2024 ...")
    ml_probs_path = os.path.join(config.results_output_dir, 'ML_probs_2024.csv')
    save_best_lr_predictions_2024(df, selected_features_rule, output_path=ml_probs_path)
    print(f"✅ ML probabilities saved: {ml_probs_path}")

    print("\n🎉 Rule evaluation completed!")

if __name__ == "__main__":
    main()
else:
    main()
