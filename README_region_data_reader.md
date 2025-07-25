# Interactive Region Data Reader

A simple interactive script to explore the merged wave data for any region in the Peru Swan project.

## Quick Start

```bash
# Run the interactive script
python scripts/read_region_data_interactive.py
```

## What it does

1. **Lists available regions** - Shows all 10 coastal regions with their port names
2. **Loads merged data** - Reads the `df_swan_waverys_merged.csv` file for your selected region
3. **Displays data summary** - Shows comprehensive information about the data structure
4. **Creates overview plots** - Generates 4-panel visualization of the data
5. **Interactive explorer** - Provides menu-driven data exploration

## Available Regions

| Region | Port Name |
|--------|-----------|
| run_g1 | CALETA_GRAU |
| run_g2 | CALETA_ORGANOS |
| run_g3 | CALETA_TIERRA_COLORADA |
| run_g4 | PUERTO_ETEN |
| run_g5 | CALETA_GRAMITA |
| run_g6 | ANCON |
| run_g7 | CALETA_SAN_ANDRES |
| run_g8 | CALETA_NAZCA |
| run_g9 | CALETA_ATICO |
| run_g10 | DPA_VILA_VILA |

## Data Summary Features

- **Basic info**: Shape, date range, total days
- **Column details**: Data types, null counts, missing data analysis
- **Event data**: Summary of event columns (if available)
- **Wave height**: Statistical summary of wave data

## Overview Plots

1. **Wave height time series** - Daily wave height over time
2. **Wave height distribution** - Histogram of wave heights
3. **Event time series** - Daily event occurrences (0/1)
4. **Monthly averages** - Monthly mean wave heights

## Interactive Explorer Options

1. **Show first/last 10 rows** - Quick data preview
2. **Data types** - Column type information
3. **Summary statistics** - Statistical summary of numeric columns
4. **Correlation matrix** - Correlation heatmap of numeric columns
5. **Search columns** - Find columns by name pattern
6. **Filter by date** - Subset data by date range
7. **Unique values** - Show unique values in any column
8. **Custom plots** - Create time series or distribution plots
9. **Exit** - Close the explorer

## Example Usage

```bash
$ python scripts/read_region_data_interactive.py

Interactive Region Data Reader
========================================
Available regions:
   1. run_g1 (CALETA_GRAU)
   2. run_g2 (CALETA_ORGANOS)
   ...
  10. run_g10 (DPA_VILA_VILA)

Select a region (1-10) or enter region name: 4

Selected region: run_g4
Port: PUERTO_ETEN

Loading data from: /path/to/wave_analysis_pipeline/data/processed/run_g4/df_swan_waverys_merged.csv

============================================================
DATA SUMMARY FOR RUN_G4
============================================================
Shape: (3650, 45)
Date range: 2014-01-01 to 2023-12-31
Total days: 3650

Columns (45):
   1. swh_max_swan                    | float64   |   3650 non-null ( 0.0% null)
   2. swh_mean_swan                   | float64   |   3650 non-null ( 0.0% null)
   ...
```

## Output Files

- **Overview plot**: `region_data_overview_{region}.png` - 4-panel data visualization
- **Console output**: Comprehensive data summary and interactive exploration

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn

## Data Structure

The script expects merged data files at:
```
wave_analysis_pipeline/data/processed/{region}/df_swan_waverys_merged.csv
```

Each file should contain:
- Date column (converted to index)
- Wave height columns (SWAN model data)
- Event columns (port closure events)
- Additional features and metadata

## Troubleshooting

**No regions found**: Check that the data directory exists and contains region subdirectories.

**File not found**: Ensure the merged CSV file exists for your selected region.

**Import errors**: Make sure all required packages are installed.

**Plot issues**: Check that matplotlib backend is properly configured for your environment. 