"""
WAVE ANALYSIS PIPELINE - PROJECT SETUP SCRIPT
==============================================

This script creates the complete project structure for the wave analysis pipeline.
Run this once to set up everything from scratch with local GitHub-ready paths.

Usage:
    python setup_project.py
    python setup_project.py --name custom_project_name

Author: Wave Analysis Team
Date: 2024
"""

import os
import shutil
from datetime import datetime

def create_directory_structure(project_root):
    """Create the complete directory structure"""
    
    print("📁 Creating project directory structure...")
    
    directories = [
        "scripts",
        "data",
        "data/raw",
        "data/processed", 
        "data/final",
        "results",
        "results/cv_results",
        "results/rules",
        "results/aep",
        "docs",
        "logs"
    ]
    
    for directory in directories:
        full_path = os.path.join(project_root, directory)
        os.makedirs(full_path, exist_ok=True)
        print(f"  📂 {directory}/")
    
    print("✅ Directory structure created!")

def create_config_file(project_root):
    """Create the master config.py file with local paths"""
    
    print("⚙️ Creating config.py...")
    
    config_content = '''"""
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
    """Master configuration for the entire project"""
    
    # Base paths (local to repository)
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    
    # Local data directories
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
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
    RUN_PATH = 'run_G4_san_jose_to_eten'
    
    # Reference ports dictionary
    REFERENCE_PORTS = {
        'CALETA_TIERRA_COLORADA': {'latitude': -5.897, 'longitude': -80.681, 'region': 'G3_colan_to_bayovar'},
        'CALETA_ORGANOS': {'latitude': -4.176, 'longitude': -81.127, 'region': 'G2_punta_de_sal_to_cabo_blanco'},
        'CALETA_GRAU': {'latitude': -3.583, 'longitude': -80.733, 'region': 'G1_puerto_pizarro_to_caleta_cancas'},
        'ANCON': {'latitude': -11.770, 'longitude': -77.180, 'region': 'G4_ancon_to_callao'},
        'DPA_CHORRILLOS': {'latitude': -12.170, 'longitude': -77.020, 'region': 'G5'},
        'CALETA_NAZCA': {'latitude': -14.830, 'longitude': -75.090, 'region': 'G8'},
        'CALETA_ATICO': {'latitude': -16.226, 'longitude': -73.690, 'region': 'G9'},
        'DPA_VILA_VILA': {'latitude': -17.700, 'longitude': -71.330, 'region': 'G10'},
        'PUERTO_ETEN': {'latitude': -6.908, 'longitude': -79.864, 'region': 'G4_san_jose_to_eten'},
        'PUERTO_DE_PIMENTEL': {'latitude': -6.837, 'longitude': -79.934, 'region': 'G4_san_jose_to_eten'},
    }
    
    # Run to port mapping
    RUN_TO_PORT_MAPPING = {
        'run_G3_colan_to_bayovar': 'CALETA_TIERRA_COLORADA',
        'run_G2_punta_de_sal_to_cabo_blanco': 'CALETA_ORGANOS',
        'run_G1_puerto_pizarro_to_caleta_cancas': 'CALETA_GRAU',
        '06_run_G1_puerto_pizarro_to_caleta_cancas': 'CALETA_GRAU',
        'run_G4_ancon_to_callao': 'ANCON',
        'run_G4_san_jose_to_eten': 'PUERTO_ETEN',
        'run_G5': 'DPA_CHORRILLOS',
        'run_G8': 'CALETA_NAZCA',
        'run_G9': 'CALETA_ATICO',
        'run_G10': 'DPA_VILA_VILA',
        'run_san_jose_to_eten_long_run': 'PUERTO_ETEN',
    }
    
    # CSV file mapping (files should be in data/raw/)
    CSV_FILE_MAPPING = {
        'run_G4_ancon_to_callao': 'g4_ancon_callao_wave_height.csv',
        'run_G3_colan_to_bayovar': 'g3_colan_bayovar_wave_height.csv',
        'run_G2_punta_de_sal_to_cabo_blanco': 'g2_punta_sal_cabo_blanco_wave_height.csv',
        'run_G1_puerto_pizarro_to_caleta_cancas': 'g1_puerto_pizarro_cancas_wave_height.csv',
        '06_run_G1_puerto_pizarro_to_caleta_cancas': 'g1_puerto_pizarro_cancas_wave_height.csv',
        'run_G5': 'g5_wave_height.csv',
        'run_G8': 'g8_wave_height.csv',
        'run_G9': 'g9_wave_height.csv',
        'run_G10': 'g10_wave_height.csv',
        'run_G4_san_jose_to_eten': 'g4_wave_height.csv',
        'run_san_jose_to_eten_long_run': 'g4_wave_height.csv',
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
    print(f"\\n{'='*60}")
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
        ("data_prep_0.py", "Data Preparation Step 0 - Initial Processing"),
        ("data_prep_1.py", "Data Preparation Step 1 - Enhanced Feature Engineering"), 
        ("rule_evaluation.py", "Rule Evaluation - CV Pipeline and Feature Selection"),
        ("aep_calculation.py", "AEP Calculation - Final Analysis")
    ]
    
    start_time = datetime.now()
    
    for i, (script_name, description) in enumerate(pipeline_steps, 1):
        print(f"\\n🔄 STEP {i}/{len(pipeline_steps)}: {description}")
        
        if not run_script(script_name, description):
            print(f"❌ Pipeline failed at step {i}")
            return False
        
        print(f"✅ Step {i} completed successfully!")
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\\n{'='*80}")
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
'''
    
    config_path = os.path.join(project_root, 'config.py')
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Created config.py")

def create_script_templates(project_root):
    """Create template script files"""
    
    print("📝 Creating script templates...")
    
    scripts = {
        'data_prep_0.py': '''"""
DATA PREPARATION STEP 0 - INITIAL PROCESSING
============================================

Replace this template with your actual data preparation logic.
This script should read raw data and create initial processed files.

Author: Wave Analysis Team
Date: 2024
"""

from config import config, get_input_files, get_output_files
import pandas as pd
import os

def main():
    print("🔧 DATA_PREP_0.PY - Initial Processing")
    print(f"Run: {config.RUN_PATH}")
    print(f"Reference port: {config.reference_port}")
    
    # TODO: Add your data_prep_0 logic here
    # Expected outputs:
    # - df_swan_daily_new_colnames.csv
    # - df_waverys_daily_new_colnames.csv  
    # - df_swan_hourly.csv
    
    print("✅ data_prep_0.py template completed!")

if __name__ == "__main__":
    main()
else:
    main()
''',
            
        'data_prep_1.py': '''"""
DATA PREPARATION STEP 1 - ENHANCED FEATURE ENGINEERING
======================================================

Replace this template with your actual feature engineering logic.
This script should read component files and create enhanced features.

Author: Wave Analysis Team
Date: 2024
"""

from config import config, get_input_files, get_output_files
import pandas as pd
import os

def main():
    print("🔧 DATA_PREP_1.PY - Enhanced Feature Engineering")
    print(f"Run: {config.RUN_PATH}")
    print(f"Reference port: {config.reference_port}")
    
    # TODO: Add your data_prep_1 logic here
    # Expected output:
    # - df_final_{reference_port}_aggregated.csv
    
    print("✅ data_prep_1.py template completed!")

if __name__ == "__main__":
    main()
else:
    main()
''',
            
        'rule_evaluation.py': '''"""
RULE EVALUATION - CV PIPELINE AND FEATURE SELECTION
===================================================

Replace this template with your actual CV pipeline logic.
This script should run feature selection and rule evaluation.

Author: Wave Analysis Team
Date: 2024
"""

from config import config, get_input_files, get_output_files
import pandas as pd
import os

def main():
    print("🔧 RULE_EVALUATION.PY - CV Pipeline")
    print(f"Run: {config.RUN_PATH}")
    print(f"Reference port: {config.reference_port}")
    
    # TODO: Add your rule evaluation logic here
    # Expected outputs:
    # - CV results
    # - Best rules
    # - Selected features
    
    print("✅ rule_evaluation.py template completed!")

if __name__ == "__main__":
    main()
else:
    main()
''',
            
        'aep_calculation.py': '''"""
AEP CALCULATION - FINAL ANALYSIS
================================

Replace this template with your actual AEP calculation logic.
This script should perform final analysis and calculations.

Author: Wave Analysis Team  
Date: 2024
"""

from config import config, get_input_files, get_output_files
import pandas as pd
import os

def main():
    print("🔧 AEP_CALCULATION.PY - Final Analysis")
    print(f"Run: {config.RUN_PATH}")
    print(f"Reference port: {config.reference_port}")
    
    # TODO: Add your AEP calculation logic here
    
    print("✅ aep_calculation.py template completed!")

if __name__ == "__main__":
    main()
else:
    main()
'''
    }
    
    scripts_dir = os.path.join(project_root, 'scripts')
    for script_name, content in scripts.items():
        script_path = os.path.join(scripts_dir, script_name)
        with open(script_path, 'w') as f:
            f.write(content)
        print(f"  📄 {script_name}")
    
    print("✅ Script templates created!")

def create_interactive_notebook(project_root):
    """Create the interactive Jupyter notebook"""
    
    print("📓 Creating interactive notebook...")
    
    notebook_content = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Wave Analysis Pipeline - Interactive Execution\\n",
    "\\n",
    "This notebook provides interactive execution of the wave analysis pipeline.\\n",
    "Each script can be run independently in its own cell."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🔧 Setup and Configuration"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import required libraries\\n",
    "import os\\n",
    "import sys\\n",
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "from datetime import datetime\\n",
    "import importlib.util\\n",
    "import warnings\\n",
    "warnings.filterwarnings('ignore')\\n",
    "\\n",
    "# Import configuration\\n",
    "from config import config, get_input_files, get_output_files, validate_configuration, print_configuration\\n",
    "\\n",
    "print('✅ Configuration imported successfully!')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## ⚙️ Configuration and Validation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Validate and display current configuration\\n",
    "print('🔍 CONFIGURATION VALIDATION')\\n",
    "print('='*50)\\n",
    "\\n",
    "if validate_configuration():\\n",
    "    print_configuration()\\n",
    "    \\n",
    "    # Show file paths\\n",
    "    input_files = get_input_files()\\n",
    "    output_files = get_output_files()\\n",
    "    \\n",
    "    print(f'\\\\n📂 INPUT FILES:')\\n",
    "    for name, path in input_files.items():\\n",
    "        exists = '✅' if os.path.exists(path) else '❌'\\n",
    "        print(f'  {exists} {name}: {path}')\\n",
    "    \\n",
    "    print('\\\\n🎯 Ready to run pipeline!')\\n",
    "    \\n",
    "else:\\n",
    "    print('❌ Configuration validation failed!')\\n",
    "    print('Please check config.py settings')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 📊 Step 1: Data Preparation 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Execute data_prep_0.py\\n",
    "print('🚀 EXECUTING: data_prep_0.py')\\n",
    "print('='*50)\\n",
    "\\n",
    "script_path = os.path.join('scripts', 'data_prep_0.py')\\n",
    "\\n",
    "if os.path.exists(script_path):\\n",
    "    exec(open(script_path).read())\\n",
    "else:\\n",
    "    print(f'❌ Script not found: {script_path}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🛠️ Step 2: Data Preparation 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Execute data_prep_1.py\\n",
    "print('🚀 EXECUTING: data_prep_1.py')\\n",
    "print('='*50)\\n",
    "\\n",
    "script_path = os.path.join('scripts', 'data_prep_1.py')\\n",
    "\\n",
    "if os.path.exists(script_path):\\n",
    "    exec(open(script_path).read())\\n",
    "else:\\n",
    "    print(f'❌ Script not found: {script_path}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🎯 Step 3: Rule Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Execute rule_evaluation.py\\n",
    "print('🚀 EXECUTING: rule_evaluation.py')\\n",
    "print('='*50)\\n",
    "\\n",
    "script_path = os.path.join('scripts', 'rule_evaluation.py')\\n",
    "\\n",
    "if os.path.exists(script_path):\\n",
    "    exec(open(script_path).read())\\n",
    "else:\\n",
    "    print(f'❌ Script not found: {script_path}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 📈 Step 4: AEP Calculation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Execute aep_calculation.py\\n",
    "print('🚀 EXECUTING: aep_calculation.py')\\n",
    "print('='*50)\\n",
    "\\n",
    "script_path = os.path.join('scripts', 'aep_calculation.py')\\n",
    "\\n",
    "if os.path.exists(script_path):\\n",
    "    exec(open(script_path).read())\\n",
    "else:\\n",
    "    print(f'❌ Script not found: {script_path}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 📊 Results Summary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Display results summary\\n",
    "print('📊 PIPELINE RESULTS SUMMARY')\\n",
    "print('='*50)\\n",
    "\\n",
    "print(f'🎉 Pipeline completed for: {config.reference_port}')\\n",
    "print(f'📊 Run: {config.RUN_PATH}')\\n",
    "\\n",
    "# Check output files\\n",
    "output_files = get_output_files()\\n",
    "print(f'\\\\n📁 Output Files:')\\n",
    "for name, path in output_files.items():\\n",
    "    if os.path.exists(path):\\n",
    "        size = os.path.getsize(path) / (1024*1024)  # MB\\n",
    "        print(f'  ✅ {name}: {size:.1f} MB')\\n",
    "    else:\\n",
    "        print(f'  ❌ {name}: Not created')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''
    
    notebook_path = os.path.join(project_root, 'interactive_pipeline.ipynb')
    with open(notebook_path, 'w') as f:
        f.write(notebook_content)
    
    print("✅ Created interactive_pipeline.ipynb")

def create_readme(project_root, project_name):
    """Create README.md file"""
    
    print("📚 Creating README.md...")
    
    readme_content = f'''# Wave Analysis Pipeline

Automated pipeline for wave data analysis and port closure prediction.

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd {project_name}
```

### 2. Add Your Data
Place your data files in `data/raw/`:
- `closed_ports_consolidated_2024_2025.csv`
- `g4_wave_height.csv` (and other wave height CSVs)
- `waverys/` directory with WAVERYS data

### 3. Run Pipeline
```bash
# Interactive (recommended)
jupyter notebook interactive_pipeline.ipynb

# Command line
python config.py --validate
python config.py
```

## 📁 Project Structure

```
{project_name}/
├── config.py                 # Master configuration
├── interactive_pipeline.ipynb # Interactive notebook
├── scripts/
│   ├── data_prep_0.py        # Initial data preparation
│   ├── data_prep_1.py        # Feature engineering
│   ├── rule_evaluation.py    # CV pipeline
│   └── aep_calculation.py    # Final analysis
├── data/
│   ├── raw/                  # 📁 PUT YOUR DATA HERE
│   ├── processed/            # Intermediate files
│   └── final/                # Final outputs
└── results/
    ├── cv_results/           # Cross-validation results
    ├── rules/                # Best rules
    └── aep/                  # AEP calculations
```

## ⚙️ Configuration

Change runs by editing `config.py`:
```python
RUN_PATH = 'run_G4_san_jose_to_eten'  # Change this line
```

Available runs:
- `run_G4_san_jose_to_eten` → PUERTO_ETEN
- `run_G4_ancon_to_callao` → ANCON  
- `run_G3_colan_to_bayovar` → CALETA_TIERRA_COLORADA
- `run_G2_punta_de_sal_to_cabo_blanco` → CALETA_ORGANOS
- `run_G1_puerto_pizarro_to_caleta_cancas` → CALETA_GRAU
- `run_G5` → DPA_CHORRILLOS
- `run_G8` → CALETA_NAZCA
- `run_G9` → CALETA_ATICO
- `run_G10` → DPA_VILA_VILA

## 🔧 Features

- ✅ **GitHub-ready**: All paths relative to repository
- ✅ **Interactive execution**: Jupyter notebook interface  
- ✅ **Centralized configuration**: Single point of control
- ✅ **Clean structure**: Organized directories
- ✅ **Template scripts**: Easy to customize
- ✅ **Local data storage**: No external dependencies

## 📊 Pipeline Steps

1. **data_prep_0.py**: Load and prepare raw data
2. **data_prep_1.py**: Create enhanced features
3. **rule_evaluation.py**: Run CV pipeline and feature selection  
4. **aep_calculation.py**: Calculate final metrics

## 🛠️ Development

Replace the template scripts in `scripts/` with your actual code. All scripts use:

```python
from config import config, get_input_files, get_output_files
```

## Authors

Wave Analysis Team, 2024
'''
    
    readme_path = os.path.join(project_root, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print("✅ README.md created!")

def create_gitignore(project_root):
    """Create .gitignore file"""
    
    print("🔧 Creating .gitignore...")
    
    gitignore_content = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.ipynb_checkpoints/

# Data files (keep structure, ignore content)
data/raw/*
data/processed/*
data/final/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/final/.gitkeep

# Results (keep structure, ignore content)
results/cv_results/*
results/rules/*
results/aep/*
!results/cv_results/.gitkeep
!results/rules/.gitkeep
!results/aep/.gitkeep

# Logs
logs/*
!logs/.gitkeep

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Large files
*.csv
*.parquet
*.xlsx
*.nc
*.hdf5
'''
    
    gitignore_path = os.path.join(project_root, '.gitignore')
    with open(gitignore_path, 'w') as f:
        f.write(gitignore_content)
    
    print("✅ .gitignore created!")

def create_gitkeep_files(project_root):
    """Create .gitkeep files to preserve empty directories"""
    
    print("📝 Creating .gitkeep files...")
    
    gitkeep_dirs = [
        'data/raw',
        'data/processed', 
        'data/final',
        'results/cv_results',
        'results/rules',
        'results/aep',
        'logs'
    ]
    
    for directory in gitkeep_dirs:
        gitkeep_path = os.path.join(project_root, directory, '.gitkeep')
        with open(gitkeep_path, 'w') as f:
            f.write('# This file keeps the directory in git\n')
    
    print("✅ .gitkeep files created!")

def setup_complete_project(project_name):
    """Run complete project setup"""
    
    project_root = os.path.abspath(project_name)
    
    print("🚀 WAVE ANALYSIS PIPELINE - PROJECT SETUP")
    print("="*60)
    print(f"Creating project: {project_name}")
    print(f"Location: {project_root}")
    print("="*60)
    
    # Check if project already exists
    if os.path.exists(project_root):
        response = input(f"Project '{project_name}' already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("❌ Setup cancelled!")
            return False
        
        # Remove existing directory
        shutil.rmtree(project_root)
        print(f"🗑️ Removed existing project directory")
    
    # Create everything
    create_directory_structure(project_root)
    create_config_file(project_root)
    create_script_templates(project_root)
    create_interactive_notebook(project_root)
    create_readme(project_root, project_name)
    create_gitignore(project_root)
    create_gitkeep_files(project_root)
    
    print(f"\n{'='*60}")
    print("🎉 PROJECT SETUP COMPLETE!")
    print("="*60)
    print(f"📁 Project created: {project_root}")
    print(f"\n📋 Next steps:")
    print(f"  1. cd {project_name}")
    print(f"  2. Place your data files in data/raw/")
    print(f"  3. Replace script templates with your actual code")
    print(f"  4. Test: python config.py --validate")
    print(f"  5. Run: jupyter notebook interactive_pipeline.ipynb")
    print("="*60)
    
    return True

# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Wave Analysis Pipeline Project")
    parser.add_argument('--name', default='wave_analysis_pipeline', 
                       help='Project name (default: wave_analysis_pipeline)')
    
    args = parser.parse_args()
    
    # Run setup
    success = setup_complete_project(args.name)
    
    if success:
        print("\n✅ Setup completed successfully!")
        print(f"📁 Navigate to: cd {args.name}")
        print("🚀 Start with: jupyter notebook interactive_pipeline.ipynb")
    else:
        print("\n❌ Setup failed!")