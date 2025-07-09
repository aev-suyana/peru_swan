"""
MASTER CONFIGURATION AND EXECUTION SCRIPT
=========================================

GitHub repo version - all paths are local to the repository.
External data should be placed in the data/ directory.

Author: Wave Analysis Team
Date: 2024
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import importlib.util

class Config:
    # AEP pipeline parameters
    MIN_DAYS = 1
    N_SIMULATIONS = 4000
    BLOCK_LENGTH = 7
    WINDOW_DAYS = 20
    """Master configuration for the entire project"""
    
    # Base paths (local to repository)
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    
    # Local data directories
    DATA_DIR = os.path.join(PROJECT_ROOT, 'wave_analysis_pipeline', 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    FINAL_DATA_DIR = os.path.join(DATA_DIR, 'final')
    
    # Project structure paths
    SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'scripts')
    RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
    CV_RESULTS_DIR = os.path.join(RESULTS_DIR, 'cv_results')
    RULES_DIR = os.path.join(RESULTS_DIR, 'rules')
    AEP_DIR = os.path.join(RESULTS_DIR, 'aep')
    LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
    
    # Current run selection - CHANGE THIS TO SWITCH BETWEEN RUNS
    RUN_PATH = 'run_g7'

    # Reference ports dictionary
    REFERENCE_PORTS = {
    'CALETA_GRAU': {'latitude': -3.583, 'longitude': -80.733, 'region': 'g1'},  # Original coordinates appear correct
    'CALETA_ORGANOS': {'latitude': -4.176, 'longitude': -81.127, 'region': 'g2'},  # Original coordinates appear correct
    'CALETA_TIERRA_COLORADA': {'latitude': -5.897, 'longitude': -80.681, 'region': 'g3'},  # Original coordinates appear correct
    'PUERTO_ETEN': {'latitude': -6.933, 'longitude': -79.855, 'region': 'g4'},  # CORRECTED: Was -6.908, -79.864
    'CALETA_GRAMITA': {'latitude': -6.908, 'longitude': -79.864, 'region': 'g5'},  # Original coordinates appear correct
    'ANCON': {'latitude': -11.828, 'longitude': -77.131, 'region': 'g6'},  # CORRECTED: Was -11.770, -77.180
    'CALETA_SAN_ANDRES': {'latitude': -13.710, 'longitude': -76.220, 'region': 'g8'},  # CORRECTED: Was -14.830, -75.090
    'CALETA_NAZCA': {'latitude': -14.830, 'longitude': -75.090, 'region': 'g8'},  # Original coordinates appear correct for Nazca region
    'CALETA_ATICO': {'latitude': -16.232, 'longitude': -73.612, 'region': 'g9'},  # CORRECTED: Was -16.226, -73.690
    'DPA_VILA_VILA': {'latitude': -18.110, 'longitude': -70.726, 'region': 'g10'},  # CORRECTED: Was -17.700, -71.330
    }
    
    # Simplified run to port mapping
    RUN_TO_PORT_MAPPING = {
    'run_g1': 'CALETA_GRAU',
    'run_g2': 'CALETA_ORGANOS',
    'run_g3': 'CALETA_TIERRA_COLORADA',
    'run_g4': 'PUERTO_ETEN',
    'run_g5': 'CALETA_GRAMITA',
    'run_g6': 'ANCON',
    'run_g7': 'CALETA_SAN_ANDRES',
    'run_g8': 'CALETA_NAZCA',
    'run_g9': 'CALETA_ATICO',
    'run_g10': 'DPA_VILA_VILA',
}

    # Simplified CSV file mapping
    CSV_FILE_MAPPING = {
    'run_g1': 'g1_wave_height.csv',
    'run_g2': 'g2_wave_height.csv',
    'run_g3': 'g3_wave_height.csv',
    'run_g4': 'g4_wave_height.csv',
    'run_g5': 'g5_wave_height.csv',
    'run_g6': 'g6_wave_height.csv',
    'run_g7': 'g7_wave_height.csv',
    'run_g8': 'g8_wave_height.csv',
    'run_g9': 'g9_wave_height.csv',
    'run_g10': 'g10_wave_height.csv',
}
    
    # Feature engineering settings
    PERSISTENCE_WINDOWS = [2, 3, 5, 7, 14]
    TREND_WINDOWS = [3, 5, 7, 14] 
    CHANGE_WINDOWS = [3, 5, 7, 14]
    LAG_WINDOWS = [1, 3, 5, 7, 14]
    
    # CV pipeline settings
    N_FOLDS = 6
    USE_TIME_SERIES_CV = True
    TOP_K_FEATURES = 450
    MAX_COMBINATIONS = 900
    TARGET_F1 = 0.6
    
    @property
    def reference_port(self):
        return self.RUN_TO_PORT_MAPPING.get(self.RUN_PATH)
    
    @property
    def port_info(self):
        port = self.reference_port
        if port and port in self.REFERENCE_PORTS:
            return self.REFERENCE_PORTS[port]
        return None
    
    @property
    def reference_latitude(self):
        info = self.port_info
        return info['latitude'] if info else None
    
    @property
    def reference_longitude(self):
        info = self.port_info
        return info['longitude'] if info else None
    
    @property
    def csv_filename(self):
        return self.CSV_FILE_MAPPING.get(self.RUN_PATH, 'wave_height.csv')
    
    @property
    def csv_path(self):
        return os.path.join(self.RAW_DATA_DIR, self.csv_filename)
    
    @property
    def run_output_dir(self):
        return os.path.join(self.PROCESSED_DATA_DIR, self.RUN_PATH)
    
    @property
    def results_output_dir(self):
        return os.path.join(self.CV_RESULTS_DIR, self.RUN_PATH)

# Create global config instance
config = Config()

def get_input_files():
    """Get standard input file paths for current configuration"""
    return {
        'swan_daily': os.path.join(config.run_output_dir, 'df_swan_daily_new_colnames.csv'),
        'waverys_daily': os.path.join(config.run_output_dir, 'df_waverys_daily_new_colnames.csv'),
        'swan_hourly': os.path.join(config.run_output_dir, 'df_swan_hourly.csv'),
        'final_aggregated': os.path.join(config.run_output_dir, f'df_final_{config.reference_port}_aggregated.csv'),
        'closed_ports': os.path.join(config.RAW_DATA_DIR, 'closed_ports_consolidated_2024_2025.csv'),
        'wave_height_csv': config.csv_path
    }

def get_output_files():
    """Get standard output file paths for current configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        'cv_results': os.path.join(config.results_output_dir, f'cv_results_{config.reference_port}_{timestamp}.csv'),
        'best_rules': os.path.join(config.results_output_dir, f'best_rules_{config.reference_port}_{timestamp}.csv'),
        'selected_features': os.path.join(config.results_output_dir, f'selected_features_{config.reference_port}_{timestamp}.txt'),
        'aep_results': os.path.join(config.AEP_DIR, f'aep_results_{config.reference_port}_{timestamp}.csv')
    }

def validate_configuration():
    """Validate the current configuration"""
    print("🔍 Validating configuration...")
    
    if not config.reference_port:
        print(f"❌ Error: Unknown run path '{config.RUN_PATH}'")
        print(f"Available runs: {list(config.RUN_TO_PORT_MAPPING.keys())}")
        return False
    
    if not config.port_info:
        print(f"❌ Error: Port info not found for '{config.reference_port}'")
        return False
    
    # Create directories if they don't exist
    for directory in [config.DATA_DIR, config.RAW_DATA_DIR, config.results_output_dir]:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Configuration validated!")
    return True

def print_configuration():
    """Print current configuration"""
    print("="*80)
    print("CURRENT CONFIGURATION (GitHub Repo Version)")
    print("="*80)
    print(f"Project root: {config.PROJECT_ROOT}")
    print(f"Run path: {config.RUN_PATH}")
    print(f"Reference port: {config.reference_port}")
    print(f"Coordinates: ({config.reference_latitude}, {config.reference_longitude})")
    print(f"Data directory: {config.DATA_DIR}")
    print(f"Results directory: {config.RESULTS_DIR}")
    print("="*80)

def run_script(script_name, description):
    """Run a script from the scripts directory"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Script: {script_name}")
    print("="*60)
    
    script_path = os.path.join(config.SCRIPTS_DIR, script_name)
    
    if not os.path.exists(script_path):
        print(f"❌ Error: Script not found: {script_path}")
        return False
    
    try:
        spec = importlib.util.spec_from_file_location("script_module", script_path)
        script_module = importlib.util.module_from_spec(spec)
        script_module.config = config
        spec.loader.exec_module(script_module)
        print(f"✅ {description} completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_complete_pipeline():
    """Run the complete pipeline in order"""
    print("🚀 STARTING COMPLETE PIPELINE")
    print("="*80)
    
    if not validate_configuration():
        print("❌ Configuration validation failed!")
        return False
    
    print_configuration()
    
    pipeline_steps = [
        ("data_preparation_0.py", "Data Preparation Step 0 - Initial Processing"),
        ("data_preparation_1.py", "Data Preparation Step 1 - Enhanced Feature Engineering"), 
        ("rule_evaluation.py", "Rule Evaluation - CV Pipeline and Feature Selection"),
        ("aep_calculation.py", "AEP Calculation - Final Analysis")
    ]
    
    start_time = datetime.now()
    
    for i, (script_name, description) in enumerate(pipeline_steps, 1):
        print(f"\n🔄 STEP {i}/{len(pipeline_steps)}: {description}")
        
        if not run_script(script_name, description):
            print(f"❌ Pipeline failed at step {i}")
            return False
        
        print(f"✅ Step {i} completed successfully!")
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*80}")
    print("🎉 COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    print("="*80)
    print(f"⏱️ Total execution time: {duration}")
    print(f"📊 Run: {config.RUN_PATH}")
    print(f"🏠 Port: {config.reference_port}")
    print(f"📁 Results: {config.results_output_dir}")
    print("="*80)
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Wave Analysis Pipeline")
    parser.add_argument('--script', help='Run individual script')
    parser.add_argument('--validate', action='store_true', help='Only validate configuration')
    
    args = parser.parse_args()
    
    if args.validate:
        validate_configuration()
        print_configuration()
    elif args.script:
        if validate_configuration():
            run_script(args.script, f"Individual script: {args.script}")
    else:
        run_complete_pipeline()
