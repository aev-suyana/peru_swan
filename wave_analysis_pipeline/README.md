# Wave Analysis Pipeline

Automated pipeline for wave data analysis and port closure prediction.

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd wave_analysis_pipeline
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
wave_analysis_pipeline/
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
