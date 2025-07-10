#%%
import os
import glob
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

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

import numpy as np

# Calculate observed metrics
best_df['obs_precision'] = best_df['obs_tp'] / (best_df['obs_tp'] + best_df['obs_fp'])
best_df['obs_recall'] = best_df['obs_tp'] / (best_df['obs_tp'] + best_df['obs_fn'])
best_df['obs_accuracy'] = (best_df['obs_tp'] + best_df['obs_tn']) / (best_df['obs_tp'] + best_df['obs_fp'] + best_df['obs_tn'] + best_df['obs_fn'])
best_df['obs_f1'] = 2 * best_df['obs_precision'] * best_df['obs_recall'] / (best_df['obs_precision'] + best_df['obs_recall'])

# Calculate mean (simulated) metrics
best_df['mean_precision'] = best_df['mean_tp'] / (best_df['mean_tp'] + best_df['mean_fp'])
best_df['mean_recall'] = best_df['mean_tp'] / (best_df['mean_tp'] + best_df['mean_fn'])
best_df['mean_accuracy'] = (best_df['mean_tp'] + best_df['mean_tn']) / (best_df['mean_tp'] + best_df['mean_fp'] + best_df['mean_tn'] + best_df['mean_fn'])
best_df['mean_f1'] = 2 * best_df['mean_precision'] * best_df['mean_recall'] / (best_df['mean_precision'] + best_df['mean_recall'])

# Handle division by zero and invalid results
def clean_metric(col):
    best_df[col] = best_df[col].replace([np.inf, -np.inf], np.nan)

# Sort by run number if run_dir is like 'run_g1', 'run_g2', ..., 'run_g10'
def extract_run_number(run_dir):
    import re
    m = re.search(r'(\d+)', str(run_dir))
    return int(m.group(1)) if m else 0

best_df = best_df.copy()
best_df['run_number'] = best_df['run_dir'].apply(extract_run_number)
best_df = best_df.sort_values('run_number').reset_index(drop=True)
best_df = best_df.drop(columns=['run_number'])

# Add totals row for numeric columns BEFORE calculating performance
numeric_cols = best_df.select_dtypes(include=[np.number]).columns
summary_row = best_df[numeric_cols].sum().to_dict()
summary_row.update({col: 'TOTAL' for col in best_df.columns if col not in numeric_cols})
best_df = pd.concat([best_df, pd.DataFrame([summary_row])], ignore_index=True)

# Calculate observed metrics (skip the totals row for now)
import numpy as np
per_run = best_df.iloc[:-1].copy()
per_run['obs_precision'] = per_run['obs_tp'] / (per_run['obs_tp'] + per_run['obs_fp'])
per_run['obs_recall'] = per_run['obs_tp'] / (per_run['obs_tp'] + per_run['obs_fn'])
per_run['obs_accuracy'] = (per_run['obs_tp'] + per_run['obs_tn']) / (per_run['obs_tp'] + per_run['obs_fp'] + per_run['obs_tn'] + per_run['obs_fn'])
per_run['obs_f1'] = 2 * per_run['obs_precision'] * per_run['obs_recall'] / (per_run['obs_precision'] + per_run['obs_recall'])
per_run['mean_precision'] = per_run['mean_tp'] / (per_run['mean_tp'] + per_run['mean_fp'])
per_run['mean_recall'] = per_run['mean_tp'] / (per_run['mean_tp'] + per_run['mean_fn'])
per_run['mean_accuracy'] = (per_run['mean_tp'] + per_run['mean_tn']) / (per_run['mean_tp'] + per_run['mean_fp'] + per_run['mean_tn'] + per_run['mean_fn'])
per_run['mean_f1'] = 2 * per_run['mean_precision'] * per_run['mean_recall'] / (per_run['mean_precision'] + per_run['mean_recall'])

def clean_metric(col):
    per_run[col] = per_run[col].replace([np.inf, -np.inf], np.nan)
for col in ['obs_precision', 'obs_recall', 'obs_accuracy', 'obs_f1',
            'mean_precision', 'mean_recall', 'mean_accuracy', 'mean_f1']:
    clean_metric(col)

# Compute totals row metrics from summed confusion matrix
last = best_df.iloc[-1]
totals_metrics = {}
try:
    obs_tp, obs_fp, obs_tn, obs_fn = [float(last.get(x, np.nan)) for x in ['obs_tp','obs_fp','obs_tn','obs_fn']]
    totals_metrics['obs_precision'] = obs_tp / (obs_tp + obs_fp) if (obs_tp + obs_fp) > 0 else np.nan
    totals_metrics['obs_recall'] = obs_tp / (obs_tp + obs_fn) if (obs_tp + obs_fn) > 0 else np.nan
    totals_metrics['obs_accuracy'] = (obs_tp + obs_tn) / (obs_tp + obs_fp + obs_tn + obs_fn) if (obs_tp + obs_fp + obs_tn + obs_fn) > 0 else np.nan
    totals_metrics['obs_f1'] = 2 * totals_metrics['obs_precision'] * totals_metrics['obs_recall'] / (totals_metrics['obs_precision'] + totals_metrics['obs_recall']) if (totals_metrics['obs_precision'] + totals_metrics['obs_recall']) > 0 else np.nan
except Exception:
    totals_metrics['obs_precision'] = totals_metrics['obs_recall'] = totals_metrics['obs_accuracy'] = totals_metrics['obs_f1'] = np.nan
try:
    mean_tp, mean_fp, mean_tn, mean_fn = [float(last.get(x, np.nan)) for x in ['mean_tp','mean_fp','mean_tn','mean_fn']]
    totals_metrics['mean_precision'] = mean_tp / (mean_tp + mean_fp) if (mean_tp + mean_fp) > 0 else np.nan
    totals_metrics['mean_recall'] = mean_tp / (mean_tp + mean_fn) if (mean_tp + mean_fn) > 0 else np.nan
    totals_metrics['mean_accuracy'] = (mean_tp + mean_tn) / (mean_tp + mean_fp + mean_tn + mean_fn) if (mean_tp + mean_fp + mean_tn + mean_fn) > 0 else np.nan
    totals_metrics['mean_f1'] = 2 * totals_metrics['mean_precision'] * totals_metrics['mean_recall'] / (totals_metrics['mean_precision'] + totals_metrics['mean_recall']) if (totals_metrics['mean_precision'] + totals_metrics['mean_recall']) > 0 else np.nan
except Exception:
    totals_metrics['mean_precision'] = totals_metrics['mean_recall'] = totals_metrics['mean_accuracy'] = totals_metrics['mean_f1'] = np.nan

# Reattach metrics to DataFrame
for col in ['obs_precision', 'obs_recall', 'obs_accuracy', 'obs_f1',
            'mean_precision', 'mean_recall', 'mean_accuracy', 'mean_f1']:
    best_df.loc[:len(per_run)-1, col] = per_run[col].values
    best_df.at[len(best_df)-1, col] = totals_metrics[col]

print(f"\nBest results from each run ({len(best_df)-1} rows + totals):")
print(best_df)

# Save to CSV
out_path = os.path.join(BASE_RESULTS_DIR, 'aep_best_per_run_summary.xlsx')
best_df.to_excel(out_path, index=False)
print(f"\n✅ Saved best-per-run summary: {out_path}")

# %%
