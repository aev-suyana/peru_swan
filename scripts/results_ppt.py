import os
import glob
import pandas as pd

# Directory containing all run subfolders (update if needed)
BASE_RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'cv_results')

# Pattern to find all multi-threshold summary CSVs in each run (update if needed)
SUMMARY_PATTERN = 'aep_multi_threshold_full_summary_*.csv'

# Find all run directories
run_dirs = [os.path.join(BASE_RESULTS_DIR, d) for d in os.listdir(BASE_RESULTS_DIR)
            if os.path.isdir(os.path.join(BASE_RESULTS_DIR, d))]

best_rows = []

for run_dir in run_dirs:
    summary_files = sorted(glob.glob(os.path.join(run_dir, SUMMARY_PATTERN)))
    if not summary_files:
        print(f"No summary file found in {run_dir}")
        continue
    # Use the latest summary file if multiple exist
    summary_path = summary_files[-1]
    df = pd.read_csv(summary_path)
    # Sort by absolute mean_loss_minus_obs (best match to observed loss)
    if 'mean_loss_minus_obs' in df.columns:
        df_sorted = df.reindex(df['mean_loss_minus_obs'].abs().sort_values().index)
    else:
        df_sorted = df
    best_row = df_sorted.iloc[0]
    best_row['run_dir'] = os.path.basename(run_dir)
    best_rows.append(best_row)

# Combine into a DataFrame (should have 10 rows if 10 runs)
best_df = pd.DataFrame(best_rows)

print(f"\nBest results from each run ({len(best_df)} rows):")
print(best_df.head(10))

# Optionally save to CSV
out_path = os.path.join(BASE_RESULTS_DIR, 'aep_best_per_run_summary.csv')
best_df.to_csv(out_path, index=False)
print(f"\n✅ Saved best-per-run summary: {out_path}")
