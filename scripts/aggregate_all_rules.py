# Cell 1: Imports and setup
#%%
import os
import glob
import pandas as pd

# Cell 2: Directory and run detection
#%%
try:
    script_dir = os.path.dirname(__file__)
except NameError:
    script_dir = os.getcwd()

top_results_dir = os.path.abspath(os.path.join(script_dir, '..', 'results', 'cv_results'))
run_dirs = [os.path.join(top_results_dir, d) for d in os.listdir(top_results_dir)
            if os.path.isdir(os.path.join(top_results_dir, d)) and d.startswith('run_g')]
rule_file_patterns = [
    'rule_cv_results*.csv',
    'stable_rules*.csv',
    'enhanced_multi_rule_summary*.csv',
    'fast_multi_rule_summary*.csv',
    'multi_rule_aep_summary*.csv',
    'enhanced_multi_rule_complete_summary*.csv',
]

# Cell 3: Aggregate all rules into a single DataFrame
#%%
# Aggregate single-condition and multi-condition rules separately
single_rule_patterns = [
    'rule_cv_results*.csv',
    'stable_rules*.csv',
]
multi_rule_patterns = [
    'enhanced_multi_rule_summary*.csv',
    'fast_multi_rule_summary*.csv',
    'multi_rule_aep_summary*.csv',
    'enhanced_multi_rule_complete_summary*.csv',
]

#%%
import re
from collections import defaultdict

# Helper to extract timestamp from filename (if present)
def extract_timestamp(filename):
    match = re.search(r'(\d{8}_\d{6})', filename)
    if match:
        return match.group(1)
    return None

# Aggregate single-condition rules (latest file per pattern per run)
single_rules = []
for run_dir in run_dirs:
    run_name = os.path.basename(run_dir)
    for pattern in single_rule_patterns:
        files = glob.glob(os.path.join(run_dir, pattern))
        if not files:
            continue
        # Pick latest file by modification time (or by timestamp if present)
        latest_file = max(files, key=lambda f: os.path.getmtime(f))
        try:
            df = pd.read_csv(latest_file)
            df['run'] = run_name
            df['source_file'] = os.path.basename(latest_file)
            single_rules.append(df)
        except Exception as e:
            print(f"Could not read {latest_file}: {e}")
if single_rules:
    single_rules_df = pd.concat(single_rules, ignore_index=True)
    print(f"Aggregated {len(single_rules_df)} single-condition rules (latest files only).")
else:
    print("❌ No single-condition rule files found to aggregate.")
    single_rules_df = None

#%%
# Aggregate multi-condition rules (latest file per pattern per run)
multi_rules = []
for run_dir in run_dirs:
    run_name = os.path.basename(run_dir)
    for pattern in multi_rule_patterns:
        files = glob.glob(os.path.join(run_dir, pattern))
        if not files:
            continue
        # Pick latest file by modification time (or by timestamp if present)
        latest_file = max(files, key=lambda f: os.path.getmtime(f))
        try:
            df = pd.read_csv(latest_file)
            df['run'] = run_name
            df['source_file'] = os.path.basename(latest_file)
            multi_rules.append(df)
        except Exception as e:
            print(f"Could not read {latest_file}: {e}")
if multi_rules:
    multi_rules_df = pd.concat(multi_rules, ignore_index=True)
    print(f"Aggregated {len(multi_rules_df)} multi-condition rules (latest files only).")
else:
    print("❌ No multi-condition rule files found to aggregate.")
    multi_rules_df = None

#%%
# Preview single-condition rules
if single_rules_df is not None:
    print("Single-condition rules columns:", list(single_rules_df.columns))
    print(single_rules_df.head(10))

#%%
# Preview multi-condition rules
if multi_rules_df is not None:
    print("Multi-condition rules columns:", list(multi_rules_df.columns))
    print(multi_rules_df.head(10))

#%%
# Extract top 10 single-condition rules per run by recall_mean (filter rule_type == 'single')
single_rules_top10_by_run = None
if single_rules_df is not None and 'run' in single_rules_df.columns and 'recall_mean' in single_rules_df.columns:
    filtered_single_rules_df = single_rules_df[single_rules_df['rule_type'] == 'single'] if 'rule_type' in single_rules_df.columns else single_rules_df
    single_rules_top10_by_run = (
        filtered_single_rules_df.sort_values(by=["run", "recall_mean"], ascending=[True, False])
        .groupby("run", group_keys=False)
        .head(10)
    )
    print("Top 10 single-condition rules per run by recall_mean (rule_type == 'single'):")
    print(single_rules_top10_by_run)
else:
    print("Column 'run' or 'recall_mean' not found in single_rules_df.")

#%%
# Extract top 10 multi-condition rules per run by F1-score
multi_rules_top10_by_run = None
if multi_rules_df is not None and 'run' in multi_rules_df.columns:
    if 'f1_score' in multi_rules_df.columns:
        multi_rules_top10_by_run = (
            multi_rules_df.sort_values(by=["run", "f1_score"], ascending=[True, False])
            .groupby("run", group_keys=False)
            .head(10)
        )
        print("Top 10 multi-condition rules per run by f1_score:")
        print(multi_rules_top10_by_run)
    else:
        print("Column 'f1_score' not found in multi_rules_df. Cannot extract top 10 multi-condition rules.")
else:
    print("Column 'run' not found in multi_rules_df.")

# %%

# === Find common rules (single or multi) that perform well across all runs using recall ===
def get_recall_col(df):
    # Try common recall column names
    for col in ['recall_mean', 'recall', 'recall_score', 'obs_recall']:
        if df is not None and col in df.columns:
            return col
    return None

single_recall_col = get_recall_col(single_rules_df) if single_rules_df is not None else None
multi_recall_col = get_recall_col(multi_rules_df) if multi_rules_df is not None else None

# Prepare DataFrames for combination
single_rules_common = None
if single_rules_df is not None and single_recall_col:
    # Use 'rule_name' as description for single rules, features/thresholds may not exist
    single_rules_common = single_rules_df[['run', 'rule_name', single_recall_col]].copy()
    single_rules_common['description'] = single_rules_common['rule_name']
    single_rules_common['features'] = ''
    single_rules_common['thresholds'] = ''
    single_rules_common['rule_type'] = 'single'
    single_rules_common = single_rules_common.rename(columns={single_recall_col: 'recall'})
    single_rules_common = single_rules_common[['run', 'description', 'features', 'thresholds', 'recall', 'rule_type']]

multi_rules_common = None
if multi_rules_df is not None and multi_recall_col:
    print(f"Loaded {len(multi_rules_df)} multi-condition rules.")
    print("Multi-condition rules columns:", list(multi_rules_df.columns))
    print("Sample multi-condition rules:")
    print(multi_rules_df.head(10))
    
    # Debug: Check what recall column we found
    print(f"Using recall column: '{multi_recall_col}'")
    print(f"Recall column values sample: {multi_rules_df[multi_recall_col].head(10).tolist()}")
    
    # Multi rules should have description, features, thresholds
    needed_cols = ['run', 'description', 'features', 'thresholds', multi_recall_col]
    existing_cols = [col for col in needed_cols if col in multi_rules_df.columns]
    missing_cols = [col for col in needed_cols if col not in multi_rules_df.columns]
    print(f"Existing columns: {existing_cols}")
    print(f"Missing columns: {missing_cols}")
    
    multi_rules_common = multi_rules_df[existing_cols].copy()
    # Add missing columns as empty string
    for col in ['description', 'features', 'thresholds']:
        if col not in multi_rules_common.columns:
            multi_rules_common[col] = ''
            print(f"Added missing column: '{col}'")
    
    multi_rules_common['rule_type'] = 'multi'
    multi_rules_common = multi_rules_common.rename(columns={multi_recall_col: 'recall'})
    multi_rules_common = multi_rules_common[['run', 'description', 'features', 'thresholds', 'recall', 'rule_type']]
    
    # Debug: Check for NaN values in recall
    print(f"Multi-rules recall stats: min={multi_rules_common['recall'].min()}, max={multi_rules_common['recall'].max()}, mean={multi_rules_common['recall'].mean()}")
    print(f"Multi-rules with NaN recall: {multi_rules_common['recall'].isna().sum()}")
    
    # Print unique multi-rule identities
    multi_rules_common['rule_id'] = multi_rules_common.apply(
        lambda row: f"{row['rule_type']}|{row['description']}|{row['features']}|{row['thresholds']}", axis=1
    )
    print("Sample multi-rule identities:")
    print(multi_rules_common['rule_id'].head(10).to_list())
else:
    print("Multi-rules not processed because:")
    print(f"  multi_rules_df is None: {multi_rules_df is None}")
    print(f"  multi_recall_col is None: {multi_recall_col is None}")
    if multi_rules_df is not None:
        print(f"  Available columns: {list(multi_rules_df.columns)}")
        print(f"  Recall columns found: {[col for col in multi_rules_df.columns if 'recall' in col.lower()]}")

# Combine both
print(f"\nCombining rules:")
print(f"  Single rules common: {len(single_rules_common) if single_rules_common is not None else 0}")
print(f"  Multi rules common: {len(multi_rules_common) if multi_rules_common is not None else 0}")

combined_rules = pd.concat([df for df in [single_rules_common, multi_rules_common] if df is not None], ignore_index=True)
print(f"  Combined total: {len(combined_rules)}")

# Debug: Check rule types in combined data
if len(combined_rules) > 0:
    print(f"  Rule types in combined data: {combined_rules['rule_type'].value_counts().to_dict()}")
    print(f"  Recall stats by rule type:")
    for rule_type in combined_rules['rule_type'].unique():
        type_data = combined_rules[combined_rules['rule_type'] == rule_type]
        print(f"    {rule_type}: min={type_data['recall'].min():.3f}, max={type_data['recall'].max():.3f}, mean={type_data['recall'].mean():.3f}")

# Define a unique rule identity
# For multi-rules, use only description and features (ignore thresholds) to group similar rules across regions
def create_rule_id(row):
    if row['rule_type'] == 'multi':
        # For multi-rules, ignore thresholds to group similar rules across regions
        return f"{row['rule_type']}|{row['description']}|{row['features']}"
    else:
        # For single rules, keep the full identity
        return f"{row['rule_type']}|{row['description']}|{row['features']}|{row['thresholds']}"

combined_rules['rule_id'] = combined_rules.apply(create_rule_id, axis=1)

# Aggregate performance across runs
print(f"\nAggregating rules by rule_id:")
print(f"  Total unique rule_ids: {combined_rules['rule_id'].nunique()}")
print(f"  Rule_ids by type: {combined_rules.groupby('rule_type')['rule_id'].nunique().to_dict()}")

agg = (combined_rules
       .groupby('rule_id')
       .agg(
           mean_recall=('recall', 'mean'),
           min_recall=('recall', 'min'),
           runs_present=('run', 'nunique'),
           description=('description', 'first'),
           features=('features', 'first'),
           thresholds=('thresholds', 'first'),
           rule_type=('rule_type', 'first')
       )
       .reset_index())

print(f"  Aggregated rules by type: {agg['rule_type'].value_counts().to_dict()}")
print(f"  Multi-rules runs_present stats: {agg[agg['rule_type'] == 'multi']['runs_present'].describe()}")
print(f"  Multi-rules mean_recall stats: {agg[agg['rule_type'] == 'multi']['mean_recall'].describe()}")

# Keep rules present in at least 6 out of 10 runs (60% coverage)
num_runs = combined_rules['run'].nunique()
min_runs_required = max(6, int(num_runs * 0.6))  # At least 6 runs or 60% of total runs

print(f"\nCoverage filtering:")
print(f"  Total runs: {num_runs}")
print(f"  Min runs required: {min_runs_required}")
print(f"  Rules by type before coverage filter:")
for rule_type in agg['rule_type'].unique():
    type_rules = agg[agg['rule_type'] == rule_type]
    print(f"    {rule_type}: {len(type_rules)} rules")
    if len(type_rules) > 0:
        print(f"      Runs present range: {type_rules['runs_present'].min()}-{type_rules['runs_present'].max()}")
        print(f"      Rules with >= {min_runs_required} runs: {len(type_rules[type_rules['runs_present'] >= min_runs_required])}")

common_rules = agg[agg['runs_present'] >= min_runs_required]
print(f"  Rules after coverage filter: {len(common_rules)}")
print(f"  Rules by type after coverage filter: {common_rules['rule_type'].value_counts().to_dict()}")

# Filter for reasonably good recall (e.g., mean_recall >= 0.55 for multi-rules, 0.6 for single-rules)
print(f"\nFiltering by recall:")
print(f"  Rules before recall filter: {len(common_rules)}")

# Different recall thresholds for different rule types
single_rules_filtered = common_rules[(common_rules['rule_type'] == 'single') & (common_rules['mean_recall'] >= 0.6)]
multi_rules_filtered = common_rules[(common_rules['rule_type'] == 'multi') & (common_rules['mean_recall'] >= 0.55)]

print(f"  Single rules with mean_recall >= 0.6: {len(single_rules_filtered)}")
print(f"  Multi rules with mean_recall >= 0.55: {len(multi_rules_filtered)}")

print(f"  Rules by rule_type before recall filter:")
for rule_type in common_rules['rule_type'].unique():
    type_rules = common_rules[common_rules['rule_type'] == rule_type]
    print(f"    {rule_type}: {len(type_rules)} rules, mean recall range: {type_rules['mean_recall'].min():.3f}-{type_rules['mean_recall'].max():.3f}")

good_common_rules = pd.concat([single_rules_filtered, multi_rules_filtered], ignore_index=True)

print(f"\nFound {len(good_common_rules)} common rules present in at least {min_runs_required}/{num_runs} runs")
print(f"  Single rules: recall >= 0.6, Multi rules: recall >= 0.55")
print(f"Final rules by type: {good_common_rules['rule_type'].value_counts().to_dict()}")
print(f"Coverage threshold: {min_runs_required}/{num_runs} runs ({min_runs_required/num_runs:.1%})")

# Show coverage distribution
print(f"\nCoverage distribution of all rules:")
coverage_counts = agg['runs_present'].value_counts().sort_index()
for coverage, count in coverage_counts.items():
    print(f"  {coverage}/{num_runs} runs: {count} rules")

# Show rules by coverage level
print(f"\nRules by coverage level (sorted by mean recall):")
for coverage in sorted(agg['runs_present'].unique(), reverse=True):
    if coverage >= min_runs_required:
        coverage_rules = good_common_rules[good_common_rules['runs_present'] == coverage]
        if len(coverage_rules) > 0:
            print(f"\nRules present in {coverage}/{num_runs} runs:")
            print(coverage_rules.sort_values(by='mean_recall', ascending=False)[['description', 'mean_recall', 'min_recall', 'runs_present']])

print(f"\nTop rules overall:")
print(good_common_rules.sort_values(by='mean_recall', ascending=False)[['description', 'mean_recall', 'min_recall', 'runs_present']])
good_common_rules.sort_values(by='mean_recall', ascending=False)
# %%

# === Calculate AEP for a selected simple rule and run ===

import sys
try:
    script_dir = os.path.dirname(__file__)
except NameError:
    script_dir = os.getcwd()
sys.path.append(script_dir)
# --- Embedded AEP Calculation Functions (no config dependencies) ---
import numpy as np
import pandas as pd
from datetime import timedelta

def calculate_annual_loss_jit(predicted_events, N, W, min_days):
    if len(predicted_events) == 0:
        return 0
    total_loss = 0
    current_event_length = 0
    for i in range(len(predicted_events)):
        if predicted_events[i] == 1:
            current_event_length += 1
        else:
            if current_event_length >= min_days:
                total_loss += N * W * current_event_length
            current_event_length = 0
    if current_event_length >= min_days:
        total_loss += N * W * current_event_length
    return total_loss

def vectorized_block_bootstrap(daily_clean, n_simulations, block_length=7, window_days=20, days_per_year=365):
    from datetime import datetime
    available_years = sorted(daily_clean.index.year.unique())
    n_days = len(daily_clean)
    valid_starts_cache = {}
    for day_of_year in range(1, days_per_year + 1, block_length):
        ref_year = available_years[0]
        try:
            center_date = datetime(ref_year, 1, 1) + timedelta(days=day_of_year - 1)
        except:
            center_date = datetime(ref_year, 12, 31)
        valid_starts = []
        for year in available_years:
            try:
                year_center = datetime(year, center_date.month, center_date.day)
            except ValueError:
                if center_date.month == 2 and center_date.day == 29:
                    year_center = datetime(year, 2, 28)
                else:
                    continue
            window_start = year_center - timedelta(days=window_days//2)
            window_end = year_center + timedelta(days=window_days//2)
            year_mask = (daily_clean.index >= window_start) & (daily_clean.index <= window_end)
            year_indices = np.where(year_mask)[0]
            for start_idx in year_indices:
                end_idx = start_idx + block_length - 1
                if end_idx < n_days and daily_clean.index[end_idx] <= window_end:
                    valid_starts.append(start_idx)
        valid_starts_cache[day_of_year] = valid_starts
    all_simulation_indices = np.zeros((n_simulations, days_per_year), dtype=int)
    for sim in range(n_simulations):
        np.random.seed(sim)
        current_day = 1
        sim_indices = []
        while current_day <= days_per_year:
            days_remaining = days_per_year - current_day + 1
            actual_block_size = min(block_length, days_remaining)
            cache_day = ((current_day - 1) // block_length) * block_length + 1
            valid_starts = valid_starts_cache.get(cache_day, list(range(n_days)))
            if valid_starts:
                chosen_start = np.random.choice(valid_starts)
                block_indices = list(range(chosen_start, min(chosen_start + actual_block_size, n_days)))
                sim_indices.extend(block_indices)
            else:
                fallback_indices = [current_day % n_days for _ in range(actual_block_size)]
                sim_indices.extend(fallback_indices)
            current_day += actual_block_size
        all_simulation_indices[sim, :len(sim_indices[:days_per_year])] = sim_indices[:days_per_year]
    return all_simulation_indices

def calculate_cm_costs(predicted_events, observed_events, N, W, min_days):
    # FP: predicted=1, observed=0; TP: predicted=1, observed=1; FN: predicted=0, observed=1
    fp_loss = 0
    tp_loss = 0
    fn_loss = 0
    current_fp = current_tp = current_fn = 0
    for i in range(len(predicted_events)):
        p, o = predicted_events[i], observed_events[i]
        # False Positive event
        if p == 1 and o == 0:
            current_fp += 1
            if current_tp > 0:
                if current_tp >= min_days:
                    tp_loss += N * W * current_tp
                current_tp = 0
            if current_fn > 0:
                if current_fn >= min_days:
                    fn_loss += N * W * current_fn
                current_fn = 0
        # True Positive event
        elif p == 1 and o == 1:
            current_tp += 1
            if current_fp > 0:
                if current_fp >= min_days:
                    fp_loss += N * W * current_fp
                current_fp = 0
            if current_fn > 0:
                if current_fn >= min_days:
                    fn_loss += N * W * current_fn
                current_fn = 0
        # False Negative event
        elif p == 0 and o == 1:
            current_fn += 1
            if current_fp > 0:
                if current_fp >= min_days:
                    fp_loss += N * W * current_fp
                current_fp = 0
            if current_tp > 0:
                if current_tp >= min_days:
                    tp_loss += N * W * current_tp
                current_tp = 0
        else:
            # End any ongoing events
            if current_fp > 0:
                if current_fp >= min_days:
                    fp_loss += N * W * current_fp
                current_fp = 0
            if current_tp > 0:
                if current_tp >= min_days:
                    tp_loss += N * W * current_tp
                current_tp = 0
            if current_fn > 0:
                if current_fn >= min_days:
                    fn_loss += N * W * current_fn
                current_fn = 0
    # End of sequence: add any remaining events
    if current_fp >= min_days:
        fp_loss += N * W * current_fp
    if current_tp >= min_days:
        tp_loss += N * W * current_tp
    if current_fn >= min_days:
        fn_loss += N * W * current_fn
    return fp_loss, tp_loss, fn_loss

def process_simulation_batch_threaded(batch_indices, trigger_values_matrix, observed_matrix, N, W, min_days, trigger_threshold, has_observed):
    batch_losses = []
    batch_fp_losses = []
    batch_tp_losses = []
    batch_fn_losses = []
    for sim_indices in batch_indices:
        try:
            trigger_vals = trigger_values_matrix[sim_indices]
            predicted_events = (trigger_vals > trigger_threshold).astype(np.int32)
            if has_observed and observed_matrix is not None:
                obs_vals = observed_matrix[sim_indices]
                fp_loss, tp_loss, fn_loss = calculate_cm_costs(predicted_events, obs_vals, N, W, min_days)
                batch_losses.append(fp_loss + tp_loss)
                batch_fp_losses.append(fp_loss)
                batch_tp_losses.append(tp_loss)
                batch_fn_losses.append(fn_loss)
            else:
                total_loss = calculate_annual_loss_jit(predicted_events, N, W, min_days)
                batch_losses.append(total_loss)
                batch_fp_losses.append(np.nan)
                batch_tp_losses.append(np.nan)
                batch_fn_losses.append(np.nan)
        except Exception as e:
            batch_losses.append(0)
            batch_fp_losses.append(np.nan)
            batch_tp_losses.append(np.nan)
            batch_fn_losses.append(np.nan)
    return batch_losses, batch_fp_losses, batch_tp_losses, batch_fn_losses, []

def calculate_unified_aep_analysis_fast(swh_data, trigger_feature, trigger_threshold, N, W, min_days=None, n_simulations=1000, observed_events=None, block_length=7, window_days=20, n_jobs=1):
    print(f"AEP ANALYSIS (standalone, no config)")
    if trigger_feature not in swh_data.columns:
        print(f"ERROR: {trigger_feature} not found in input DataFrame!")
        return None
    daily_clean = swh_data.copy()
    if 'date' in daily_clean.columns:
        daily_clean['date'] = pd.to_datetime(daily_clean['date'])
        daily_clean = daily_clean.set_index('date')
    daily_clean = daily_clean.dropna(subset=[trigger_feature])
    if len(daily_clean) == 0:
        print("ERROR: No valid rows with trigger feature after cleaning!")
        return None
    print(f"  Using {len(daily_clean)} days for simulation.")
    trigger_values_matrix = daily_clean[trigger_feature].values.astype(np.float32)
    observed_matrix = None
    has_observed = False
    if observed_events is not None:
        # Align observed_events to daily_clean index if possible
        if hasattr(observed_events, 'reindex'):
            observed_matrix = observed_events.reindex(daily_clean.index).values.astype(np.int32)
        else:
            observed_matrix = np.array(observed_events).astype(np.int32)
        has_observed = True
        print("Observed events successfully aligned. Example:", observed_matrix[:10])
        print("Observed matrix shape:", observed_matrix.shape)
    else:
        print("Observed events not used in simulation.")
    all_simulation_indices = vectorized_block_bootstrap(
        daily_clean, n_simulations, block_length, window_days, days_per_year=365
    )
    batch_size = n_simulations
    simulation_batches = [all_simulation_indices]
    all_losses = []
    all_fp_losses = []
    all_tp_losses = []
    all_fn_losses = []
    for batch_indices in simulation_batches:
        batch_losses, batch_fp, batch_tp, batch_fn, _ = process_simulation_batch_threaded(
            batch_indices, trigger_values_matrix, observed_matrix, N, W, min_days, trigger_threshold, has_observed
        )
        all_losses.extend(batch_losses)
        all_fp_losses.extend(batch_fp)
        all_tp_losses.extend(batch_tp)
        all_fn_losses.extend(batch_fn)
    annual_losses = np.array(all_losses)
    fp_losses = np.array([x for x in all_fp_losses if x is not None and not np.isnan(x)])
    tp_losses = np.array([x for x in all_tp_losses if x is not None and not np.isnan(x)])
    fn_losses = np.array([x for x in all_fn_losses if x is not None and not np.isnan(x)])
    print(f"  Completed {len(annual_losses)} simulations successfully.")
    standard_summary = {
        'mean_loss': float(np.mean(annual_losses)),
        'std_loss': float(np.std(annual_losses)),
        'max_loss': float(np.max(annual_losses)),
        'zero_prob': float(np.mean(annual_losses == 0)),
        'method': 'standalone_block_bootstrap',
        'trigger_feature': trigger_feature,
        'trigger_threshold': trigger_threshold,
        'min_days': min_days,
        'n_fishermen': N,
        'daily_wage': W,
        'n_simulations': len(annual_losses),
    }
    if has_observed:
        standard_summary.update({
            'mean_fp_loss': float(np.mean(fp_losses)) if len(fp_losses) else np.nan,
            'std_fp_loss': float(np.std(fp_losses)) if len(fp_losses) else np.nan,
            'mean_tp_loss': float(np.mean(tp_losses)) if len(tp_losses) else np.nan,
            'std_tp_loss': float(np.std(tp_losses)) if len(tp_losses) else np.nan,
            'mean_fn_loss': float(np.mean(fn_losses)) if len(fn_losses) else np.nan,
            'std_fn_loss': float(np.std(fn_losses)) if len(fn_losses) else np.nan,
            # Add raw loss arrays for percentile calculations
            'raw_fp_losses': fp_losses.tolist() if len(fp_losses) > 0 else [],
            'raw_tp_losses': tp_losses.tolist() if len(tp_losses) > 0 else [],
            'raw_fn_losses': fn_losses.tolist() if len(fn_losses) > 0 else [],
        })
        print(f"    Mean FP loss: ${standard_summary['mean_fp_loss']:,.0f} ± ${standard_summary['std_fp_loss']:,.0f}")
        print(f"    Mean TP loss: ${standard_summary['mean_tp_loss']:,.0f} ± ${standard_summary['std_tp_loss']:,.0f}")
        print(f"    Mean FN loss: ${standard_summary['mean_fn_loss']:,.0f} ± ${standard_summary['std_fn_loss']:,.0f}")
    print(f"\nAEP Results:")
    print(f"  Mean annual loss: ${standard_summary['mean_loss']:,.0f}")
    print(f"  Max annual loss: ${standard_summary['max_loss']:,.0f}")
    print(f"  Zero loss probability: {standard_summary['zero_prob']:.1%}")
    return standard_summary

#%%

# Set these variables to select the rule and run

desired_run = 'run_g1'  # Change as needed

N = 3948
W = 12
min_days = 1
n_simulations = 1000
block_length = 7
window_days = 20
#%%
# Load the data for AEP calculation (simple rule): use df_swan_daily_features.csv
swh_data_path = f'/Users/ageidv/suyana/peru_swan/wave_analysis_pipeline/data/processed/{desired_run}/df_swan_waverys_merged.csv'
swh_data = pd.read_csv(swh_data_path)

data_dir = os.path.join(top_results_dir, desired_run)
cv_results_path = os.path.join(data_dir, 'rule_cv_results.csv')
folds_dir = data_dir
fold_thresholds_path = None
for fname in os.listdir(data_dir):
    if fname.startswith('fold_thresholds_') and fname.endswith('.csv'):
        fold_thresholds_path = os.path.join(data_dir, fname)
        break
if not fold_thresholds_path:
    raise FileNotFoundError(f"No fold_thresholds_*.csv file found in {data_dir}")

# Load CV results and fold thresholds
df_cv = pd.read_csv(cv_results_path)
df_folds = pd.read_csv(fold_thresholds_path)

# Debug: Show available rules in the desired run
print(f"\n=== Available rules in {desired_run} ===")
print(f"Total rules in CV results: {len(df_cv)}")

# Show single rules with 'anom_' prefix
anom_single_rules = df_cv[(df_cv['rule_type'] == 'single') & (df_cv['rule_name'].str.contains('anom_', na=False))]
print(f"\nSingle rules with 'anom_' prefix ({len(anom_single_rules)} found):")
if len(anom_single_rules) > 0:
    for idx, row in anom_single_rules.head(10).iterrows():
        print(f"  {row['rule_name']}")
    if len(anom_single_rules) > 20:
        print(f"  ... and {len(anom_single_rules) - 20} more")
else:
    print("  No single rules with 'anom_' prefix found")

# Show all single rules
all_single_rules = df_cv[df_cv['rule_type'] == 'single']
print(f"\nAll single rules ({len(all_single_rules)} found):")
for idx, row in all_single_rules.head(10).iterrows():
    print(f"  {row['rule_name']}")
if len(all_single_rules) > 20:
    print(f"  ... and {len(all_single_rules) - 20} more")

# Show multi rules with 'anom_' prefix
anom_multi_rules = df_cv[(df_cv['rule_type'] == 'multi') & (df_cv['rule_name'].str.contains('anom_', na=False))]
print(f"\nMulti rules with 'anom_' prefix ({len(anom_multi_rules)} found):")
if len(anom_multi_rules) > 0:
    for idx, row in anom_multi_rules.head(10).iterrows():
        print(f"  {row['rule_name']}")
    if len(anom_multi_rules) > 20:
        print(f"  ... and {len(anom_multi_rules) - 20} more")
else:
    print("  No multi rules with 'anom_' prefix found")

# Show top 10 anom_ rules by recall
print(f"\n=== Top 10 'anom_' rules by recall in {desired_run} ===")
if 'recall_mean' in df_cv.columns:
    recall_col = 'recall_mean'
elif 'recall' in df_cv.columns:
    recall_col = 'recall'
else:
    recall_col = None

if recall_col:
    # Get all anom_ rules and sort by recall
    anom_rules = df_cv[df_cv['rule_name'].str.contains('anom_', na=False)].copy()
    if len(anom_rules) > 0:
        anom_rules_sorted = anom_rules.sort_values(by=recall_col, ascending=False)
        print(f"Top 10 'anom_' rules by {recall_col}:")
        for idx, row in anom_rules_sorted.head(20).iterrows():
            rule_type = row.get('rule_type', 'unknown')
            recall_val = row[recall_col]
            print(f"  {recall_val:.3f} | {rule_type} | {row['rule_name']}")
    else:
        print("No 'anom_' rules found")
else:
    print("No recall column found in CV results")
#%%
# desired_rule_name = 'swh_max_swan > threshold'  # Change as needed
desired_rule_name = 'anom_swh_max_swan > threshold'  # Change as needed
#%%
# Extract the correct threshold for the selected rule and run
feature = desired_rule_name.replace(' > threshold', '').replace('Single: ', '').strip()
rule_row = df_cv[(df_cv['rule_type'] == 'single') & (df_cv['rule_name'] == desired_rule_name)]
if rule_row.empty:
    raise ValueError(f"Rule '{desired_rule_name}' not found in CV results for {desired_run}")
folds = df_folds[(df_folds['rule_type'] == 'single') & (df_folds['rule_name'] == desired_rule_name)]
if folds.empty:
    raise ValueError(f"No fold thresholds found for rule: {desired_rule_name} in {desired_run}")
if feature in folds.columns:
    threshold = folds[feature].mean()
else:
    raise ValueError(f"No threshold column for feature '{feature}' found in fold thresholds file. Columns are: {list(folds.columns)}")

# === Allow user to tweak threshold ===
threshold_multiplier = 0.85  # Set to e.g. 0.85 to lower the threshold by 15%
threshold = threshold * threshold_multiplier
print(f"Selected rule: {desired_rule_name} | Run: {desired_run} | Feature: {feature} | Threshold: {threshold} (multiplier: {threshold_multiplier})")


# Extract observed events from swh_data
if 'event_dummy_1' in swh_data.columns:
    observed_events = swh_data.set_index('date')['event_dummy_1'] if 'date' in swh_data.columns else swh_data['event_dummy_1']
    print(swh_data['event_dummy_1'].value_counts())
    print("Observed events head:", observed_events.head() if hasattr(observed_events, 'head') else observed_events[:10])
    print("Observed events dtype:", type(observed_events))
    # Ensure observed_events index is DatetimeIndex for alignment
    if hasattr(observed_events, 'index') and not isinstance(observed_events.index, pd.DatetimeIndex):
        observed_events.index = pd.to_datetime(observed_events.index)
else:
    observed_events = None
    print("Warning: 'event_dummy_1' column not found in swh_data. Confusion matrix AEP will not be computed.")
n_jobs = -1

# === Debug: Check overlap between predicted and observed events ===
predicted_events = (swh_data[feature] > threshold).astype(int)
if 'date' in swh_data.columns:
    predicted_events.index = pd.to_datetime(swh_data['date'])
    observed_events_aligned = observed_events.reindex(predicted_events.index)
else:
    observed_events_aligned = observed_events

print("Observed events after reindex (head):", observed_events_aligned.head(10))
print("Observed events after reindex (value counts):", observed_events_aligned.value_counts(dropna=False))
print("Predicted events (head):", predicted_events.head(10))
print("Predicted events index sample:", predicted_events.index[:10])
print("Observed events index sample:", observed_events_aligned.index[:10])
print("Predicted events index dtype:", predicted_events.index.dtype)
print("Observed events index dtype:", observed_events.index.dtype if hasattr(observed_events, 'index') else type(observed_events))
overlap = ((predicted_events == 1) & (observed_events_aligned == 1)).sum()
print(f"Number of days with both predicted and observed events (TP candidates): {overlap}")
print(f"Number of predicted events: {predicted_events.sum()}")
print(f"Number of observed events: {observed_events_aligned.sum()}")

# This function may require additional parameters depending on your pipeline
try:
    aep_result = calculate_unified_aep_analysis_fast(
        swh_data, feature, threshold, N, W, min_days=min_days, n_simulations=n_simulations,
        observed_events=observed_events, block_length=block_length, window_days=window_days, n_jobs=n_jobs
    )
    print("AEP calculation result:")
    print(aep_result)
    if aep_result is not None:
        if 'mean_tp_loss' in aep_result:
            print(f"TP AEP: ${aep_result['mean_tp_loss']:,.0f} ± ${aep_result['std_tp_loss']:,.0f}")
        if 'mean_fp_loss' in aep_result:
            print(f"FP AEP: ${aep_result['mean_fp_loss']:,.0f} ± ${aep_result['std_fp_loss']:,.0f}")
        if 'mean_fn_loss' in aep_result:
            print(f"FN AEP: ${aep_result['mean_fn_loss']:,.0f} ± ${aep_result['std_fn_loss']:,.0f}")
        # Save Excel summary with TP, FP, FN rows
        import pandas as pd
        # Try to get the raw loss arrays if available, otherwise use mean/std/max
        tp_losses = None
        fp_losses = None
        fn_losses = None
        # Extract raw loss arrays directly from AEP result
        tp_losses = aep_result.get('raw_tp_losses')
        fp_losses = aep_result.get('raw_fp_losses')
        fn_losses = aep_result.get('raw_fn_losses')
        # If still None, just use mean/std/max for all
        def get_stats(losses, mean, std):
            if losses is not None and hasattr(losses, '__len__') and len(losses) > 0:
                losses = pd.Series(losses)
                return {
                    'mean': losses.mean(),
                    'std': losses.std(),
                    'max': losses.max(),
                    'p95': losses.quantile(0.95),
                    'p99': losses.quantile(0.99)
                }
            else:
                # Only mean and std are reliable, others are not
                return {
                    'mean': mean,
                    'std': std,
                    'max': float('nan'),
                    'p95': float('nan'),
                    'p99': float('nan')
                }
        # --- DEBUG: Print loss arrays before stats extraction ---
        print(f"tp_losses: type={type(tp_losses)}, len={len(tp_losses) if tp_losses is not None else 'None'}")
        if tp_losses is not None:
            print(f"tp_losses sample: {str(tp_losses[:10])}")
        print(f"fp_losses: type={type(fp_losses)}, len={len(fp_losses) if fp_losses is not None else 'None'}")
        if fp_losses is not None:
            print(f"fp_losses sample: {str(fp_losses[:10])}")
        print(f"fn_losses: type={type(fn_losses)}, len={len(fn_losses) if fn_losses is not None else 'None'}")
        if fn_losses is not None:
            print(f"fn_losses sample: {str(fn_losses[:10])}")
        stats = {}
        stats['tp'] = get_stats(tp_losses, aep_result.get('mean_tp_loss', float('nan')), aep_result.get('std_tp_loss', float('nan')))
        stats['fp'] = get_stats(fp_losses, aep_result.get('mean_fp_loss', float('nan')), aep_result.get('std_fp_loss', float('nan')))
        stats['fn'] = get_stats(fn_losses, aep_result.get('mean_fn_loss', float('nan')), aep_result.get('std_fn_loss', float('nan')))

        # Calculate observed events and recall
        observed_events_count = observed_events.sum() if observed_events is not None else float('nan')
        tp = stats['tp']['mean']
        fn = stats['fn']['mean']
        recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')

        # --- Add extra event columns ---
        # Predicted events for 2024 (if available)
        mean_predicted_events = float('nan')
        n_predicted_2024 = float('nan')
        n_observed_2024 = float('nan')
        if 'predicted_events' in locals():
            # Restrict to 2024 if possible
            if hasattr(predicted_events, 'index') and hasattr(predicted_events, 'loc'):
                pred_2024 = predicted_events.loc[predicted_events.index.year == 2024]
                mean_predicted_events = pred_2024.mean()
                n_predicted_2024 = pred_2024.sum()
            else:
                mean_predicted_events = predicted_events.mean()
                n_predicted_2024 = predicted_events.sum()
        if observed_events is not None and hasattr(observed_events, 'index'):
            obs_2024 = observed_events.loc[observed_events.index.year == 2024]
            n_observed_2024 = obs_2024.sum()
        elif observed_events is not None:
            n_observed_2024 = observed_events.sum()

        df_summary = pd.DataFrame([
            {
                'type': 'tp',
                'N_fishermen': N,
                'wages': W,
                'mean_loss': stats['tp']['mean'],
                'p95': stats['tp']['p95'],
                'p99': stats['tp']['p99'],
                'max_loss': stats['tp']['max'],
                'run_name': desired_run,
                'std_loss': stats['tp']['std'],
                'observed_events': observed_events_count,
                'recall': recall,
                'threshold': threshold,
                'observed_events_2024': n_observed_2024,
                'mean_predicted_events_2024': mean_predicted_events,
                'n_predicted_events_2024': n_predicted_2024
            },
            {
                'type': 'fp',
                'N_fishermen': N,
                'wages': W,
                'mean_loss': stats['fp']['mean'],
                'p95': stats['fp']['p95'],
                'p99': stats['fp']['p99'],
                'max_loss': stats['fp']['max'],
                'run_name': desired_run,
                'std_loss': stats['fp']['std'],
                'observed_events': observed_events_count,
                'recall': recall,
                'threshold': threshold
            },
            {
                'type': 'fn',
                'N_fishermen': N,
                'wages': W,
                'mean_loss': stats['fn']['mean'],
                'p95': stats['fn']['p95'],
                'p99': stats['fn']['p99'],
                'max_loss': stats['fn']['max'],
                'run_name': desired_run,
                'std_loss': stats['fn']['std'],
                'observed_events': observed_events_count,
                'recall': recall,
                'threshold': threshold
            }
        ])
        excel_path = f"/Users/ageidv/suyana/peru_swan/aep_common_rule/aep_breakdown_{desired_run}_{feature}_thresh{threshold_multiplier}.xlsx"
        df_summary.to_excel(excel_path, index=False)
        print(f"Saved AEP breakdown to {excel_path}")
        
                # Save TP AEP curve data (probabilities and losses)
        if 'raw_tp_losses' in aep_result and len(aep_result['raw_tp_losses']) > 0:
            tp_losses = np.array(aep_result['raw_tp_losses'])
            
            # Create AEP curve data with more points for smooth curve
            sorted_losses = np.sort(tp_losses)
            n_simulations = len(sorted_losses)
            
            # Create smooth AEP curve using spline interpolation
            from scipy.interpolate import UnivariateSpline
            
            # Create empirical CDF from sorted losses
            n_simulations = len(sorted_losses)
            empirical_probs = np.arange(1, n_simulations + 1) / n_simulations
            exceedance_probs_empirical = 1 - empirical_probs
            
            # Create smooth interpolation points
            n_smooth = 1000
            exceedance_probs_smooth = np.linspace(0.001, 0.999, n_smooth)
            
            # Use spline interpolation for smooth curve
            try:
                # Fit spline to empirical data (inverse relationship)
                spline = UnivariateSpline(exceedance_probs_empirical, sorted_losses, s=0.1, k=3)
                interpolated_losses = spline(exceedance_probs_smooth)
                
                # Ensure monotonicity (losses should increase with decreasing probability)
                for i in range(1, len(interpolated_losses)):
                    if interpolated_losses[i] < interpolated_losses[i-1]:
                        interpolated_losses[i] = interpolated_losses[i-1]
                
                exceedance_probs = exceedance_probs_smooth
            except:
                # Fallback to percentile method if spline fails
                prob_high = np.linspace(1.0, 0.1, n_smooth//2)
                prob_low = np.logspace(-1, -3, n_smooth//2)
                exceedance_probs = np.concatenate([prob_high, prob_low])
                percentiles = (1 - exceedance_probs) * 100
                percentiles = np.clip(percentiles, 0, 100)
                interpolated_losses = np.percentile(sorted_losses, percentiles)
        
        aep_curve_data = pd.DataFrame({
            'run_name': desired_run,
            'exceedance_probability': exceedance_probs,
            'tp_loss': interpolated_losses,
            'return_period': 1 / exceedance_probs  # Return period in years
        })
        
        # Save to CSV
        aep_curve_path = f"/Users/ageidv/suyana/peru_swan/aep_common_rule/tp_aep_curve_{desired_run}_{feature}_thresh{threshold_multiplier}.csv"
        aep_curve_data.to_csv(aep_curve_path, index=False)
        print(f"Saved TP AEP curve data to {aep_curve_path}")
        print(f"  Number of data points: {len(aep_curve_data)}")
        print(f"  Loss range: ${aep_curve_data['tp_loss'].min():,.0f} - ${aep_curve_data['tp_loss'].max():,.0f}")
        print(f"  Return period range: {aep_curve_data['return_period'].min():.1f} - {aep_curve_data['return_period'].max():.1f} years")
    else:
        print("No TP losses available for AEP curve generation")
except Exception as e:
    print(f"AEP calculation failed: {e}")

# %%
# === Concatenate all AEP Excel summary files at the end of the script ===
import glob
import os
summary_dir = '/Users/ageidv/suyana/peru_swan/aep_common_rule'
pattern = os.path.join(summary_dir, 'aep_breakdown_run_*_thresh*.xlsx')
file_list = glob.glob(pattern)

all_dfs = []
for file in file_list:
    df = pd.read_excel(file)
    df['source_file'] = os.path.basename(file)
    all_dfs.append(df)

if all_dfs:
    df_all = pd.concat(all_dfs, ignore_index=True)
    
    # Calculate average yearly observed loss for each run and add to the dataframe
    print("Calculating average yearly observed losses for each run...")
    
    # Get unique runs
    unique_runs = df_all['run_name'].unique()
    
    # Dictionary to store average yearly observed losses by run
    run_observed_losses = {}
    
    for run in unique_runs:
        try:
            # Load the original data for this run
            swh_data_path = f'/Users/ageidv/suyana/peru_swan/wave_analysis_pipeline/data/processed/{run}/df_swan_waverys_merged.csv'
            if os.path.exists(swh_data_path):
                swh_data = pd.read_csv(swh_data_path)
                
                # Get N and W from the dataframe for this run
                run_data = df_all[df_all['run_name'] == run].iloc[0]
                N = run_data['N_fishermen']
                W = run_data['wages']
                min_days = 1  # Default value
                
                # Extract observed events
                if 'event_dummy_1' in swh_data.columns:
                    if 'date' in swh_data.columns:
                        # Ensure date column is properly converted to datetime
                        swh_data['date'] = pd.to_datetime(swh_data['date'])
                        observed_events = swh_data.set_index('date')['event_dummy_1']
                        print(f"{run}: Data has date column, set as index")
                    else:
                        observed_events = swh_data['event_dummy_1']
                        print(f"{run}: No date column found, using simple array")
                    
                    # Calculate average yearly observed loss
                    if hasattr(observed_events, 'index') and hasattr(observed_events.index, 'year'):
                        print(f"{run}: Has datetime index with years: {sorted(observed_events.index.year.unique())}")
                        # Group by year and calculate yearly losses
                        yearly_observed_losses = []
                        for year in observed_events.index.year.unique():
                            year_events = observed_events[observed_events.index.year == year]
                            year_loss = calculate_annual_loss_jit(year_events.values, N, W, min_days)
                            yearly_observed_losses.append(year_loss)
                            print(f"{run}: Year {year} loss = ${year_loss:,.0f}")
                        
                        if yearly_observed_losses:
                            avg_yearly_observed_loss = np.mean(yearly_observed_losses)
                            run_observed_losses[run] = avg_yearly_observed_loss
                            print(f"{run}: Average yearly observed loss = ${avg_yearly_observed_loss:,.0f}")
                        else:
                            run_observed_losses[run] = float('nan')
                            print(f"{run}: No yearly losses calculated")
                    else:
                        # If no datetime index, calculate total loss and divide by number of years
                        print(f"{run}: No datetime index with years, calculating total loss")
                        total_observed_loss = calculate_annual_loss_jit(observed_events.values, N, W, min_days)
                        
                        # Estimate number of years from data length (assuming daily data)
                        estimated_years = len(observed_events) / 365.25
                        avg_yearly_observed_loss = total_observed_loss / estimated_years
                        
                        run_observed_losses[run] = avg_yearly_observed_loss
                        print(f"{run}: Total observed loss = ${total_observed_loss:,.0f}, estimated {estimated_years:.1f} years, avg yearly = ${avg_yearly_observed_loss:,.0f}")
                else:
                    run_observed_losses[run] = float('nan')
                    print(f"{run}: No event_dummy_1 column found")
            else:
                run_observed_losses[run] = float('nan')
                print(f"{run}: Data file not found at {swh_data_path}")
        except Exception as e:
            run_observed_losses[run] = float('nan')
            print(f"{run}: Error calculating observed loss - {e}")
    
    # Add the average yearly observed loss to the dataframe
    df_all['avg_yearly_observed_loss'] = df_all['run_name'].map(run_observed_losses)
    
    # Create a summary by run with average yearly observed losses
    run_summary = df_all.groupby('run_name').agg({
        'avg_yearly_observed_loss': 'first',
        'N_fishermen': 'first',
        'wages': 'first',
        'observed_events': 'first',
        'threshold': 'first'
    }).reset_index()
    
    # Add some statistics about predicted vs observed
    run_summary['total_predicted_loss'] = df_all.groupby('run_name')['mean_loss'].sum().values
    run_summary['predicted_vs_observed_ratio'] = run_summary['total_predicted_loss'] / run_summary['avg_yearly_observed_loss']
    
    # Save run summary
    run_summary.to_excel(os.path.join(summary_dir, 'aep_run_summary_with_observed.xlsx'), index=False)
    run_summary.to_csv(os.path.join(summary_dir, 'aep_run_summary_with_observed.csv'), index=False)
    print(f"\nCreated run summary with observed losses: aep_run_summary_with_observed.xlsx")
    print("Run summary preview:")
    print(run_summary)
    
    df_all.to_excel(os.path.join(summary_dir, 'aep_breakdown_all_runs.xlsx'), index=False)
    df_all.to_csv(os.path.join(summary_dir, 'aep_breakdown_all_runs.csv'), index=False)
    print(f"Concatenated {len(file_list)} files into aep_breakdown_all_runs.xlsx and .csv")
else:
    print("No Excel summary files found to concatenate.")

# === Concatenate and plot all AEP curve files ===
print("\n=== Concatenating and plotting AEP curves ===")
import matplotlib.pyplot as plt
import glob

# Find all AEP curve files
aep_curve_pattern = os.path.join(summary_dir, 'tp_aep_curve_*.csv')
aep_curve_files = glob.glob(aep_curve_pattern)

if aep_curve_files:
    print(f"Found {len(aep_curve_files)} AEP curve files:")
    for file in aep_curve_files:
        print(f"  {os.path.basename(file)}")
    
    # Load and concatenate all AEP curve data
    all_aep_data = []
    for file in aep_curve_files:
        try:
            df = pd.read_csv(file)
            all_aep_data.append(df)
            print(f"  Loaded {len(df)} data points from {os.path.basename(file)}")
        except Exception as e:
            print(f"  Error loading {file}: {e}")
    
    if all_aep_data:
        # Concatenate all data
        combined_aep_data = pd.concat(all_aep_data, ignore_index=True)
        
        # Save combined data
        combined_aep_path = os.path.join(summary_dir, 'all_tp_aep_curves.csv')
        combined_aep_data.to_csv(combined_aep_path, index=False)
        print(f"\nSaved combined AEP data to: {combined_aep_path}")
        print(f"Total data points: {len(combined_aep_data)}")
        print(f"Runs included: {combined_aep_data['run_name'].unique()}")
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot each run with greens and blacks/greys
        unique_runs = sorted(combined_aep_data['run_name'].unique())
        
        # Create a mix of greens and greys/blacks
        n_runs = len(unique_runs)
        colors = []
        
        # First half: greens (light to dark)
        green_colors = plt.cm.Greens(np.linspace(0.3, 0.8, n_runs // 2))
        colors.extend(green_colors)
        
        # Second half: greys/blacks (light grey to black)
        grey_colors = plt.cm.Greys(np.linspace(0.3, 0.9, n_runs - len(green_colors)))
        colors.extend(grey_colors)
        
        # Create mapping for legend labels
        legend_mapping = {
            'run_g1': 'Area 1', 'run_g2': 'Area 2', 'run_g3': 'Area 3', 'run_g4': 'Area 4', 'run_g5': 'Area 5',
            'run_g6': 'Area 6', 'run_g7': 'Area 7', 'run_g8': 'Area 8', 'run_g9': 'Area 9', 'run_g10': 'Area 10'
        }
        
        for i, run_name in enumerate(unique_runs):
            run_data = combined_aep_data[combined_aep_data['run_name'] == run_name]
            legend_label = legend_mapping.get(run_name, run_name)  # Use Area X if available, otherwise original name
            plt.plot(run_data['tp_loss'], run_data['exceedance_probability'], 
                    label=legend_label, color=colors[i], linewidth=5, alpha=0.8)
        
        plt.xlabel('Loss ($)', fontsize=12)
        plt.ylabel('Exceedance Probability', fontsize=12)
        # plt.title('True Positive AEP Curves by Run', fontsize=14, fontweight='bold')  # Removed title
        plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=10)  # Bottom, single row (10 columns)
        plt.grid(True, alpha=0.3)
        # Linear scales instead of log
        # plt.xscale('log')
        # plt.yscale('log')
        
        # Format x-axis with dollar signs and thousands separators
        from matplotlib.ticker import FuncFormatter
        def currency_formatter(x, pos):
            if x >= 1e6:
                return f'${x/1e6:.0f}M'
            elif x >= 1e3:
                return f'${x/1e3:.0f}K'
            else:
                return f'${x:.0f}'
        
        plt.gca().xaxis.set_major_formatter(FuncFormatter(currency_formatter))
        
        # Use default y-axis orientation (1.0 at bottom, 0.0 at top)
        # plt.ylim(1.0, 0.0)  # This forces 1.0 at top, 0.0 at bottom
        
        # Add some formatting
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(summary_dir, 'all_tp_aep_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved AEP curves plot to: {plot_path}")
        
        # Print aggregate summary statistics
        print(f"\n=== Aggregate AEP Summary ===")
        print(f"Max aggregate loss: ${aggregate_aep_data['aggregate_tp_loss'].max():,.0f}")
        print(f"Mean aggregate loss: ${aggregate_aep_data['aggregate_tp_loss'].mean():,.0f}")
        print(f"Total data points: {len(aggregate_aep_data)}")
        print(f"Probability range: {aggregate_aep_data['exceedance_probability'].min():.3f} - {aggregate_aep_data['exceedance_probability'].max():.3f}")
        
        # === Fixed Aggregated Density Plots Section ===
print("\n=== Creating Corrected Aggregated Density Plots ===")

# First, let's properly calculate individual yearly losses for each run
def calculate_proper_yearly_losses(run_name, N, W, min_days=1):
    """Calculate proper yearly losses using daily event time series"""
    try:
        # Load the data file
        data_file = f'/Users/ageidv/suyana/peru_swan/wave_analysis_pipeline/data/processed/{run_name}/df_swan_waverys_merged.csv'
        if not os.path.exists(data_file):
            print(f"  {run_name}: Data file not found")
            return {}
        
        df = pd.read_csv(data_file)
        
        if 'date' not in df.columns or 'event_dummy_1' not in df.columns:
            print(f"  {run_name}: Missing required columns")
            return {}
        
        # Set up the data properly
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # Get unique years
        years = sorted(df.index.year.unique())
        yearly_losses_dict = {}
        
        print(f"  {run_name}: Processing {len(years)} years")
        
        for year in years:
            # Get the daily event series for this year
            year_data = df[df.index.year == year]['event_dummy_1'].values
            
            # Calculate annual loss using the proper function with daily time series
            year_loss = calculate_annual_loss_jit(year_data, N, W, min_days)
            yearly_losses_dict[year] = year_loss
            
            # Debug info
            total_events = year_data.sum()
            print(f"    {year}: {len(year_data)} days, {total_events} event days, ${year_loss:,.0f} loss")
        
        return yearly_losses_dict
        
    except Exception as e:
        print(f"  Error processing {run_name}: {e}")
        return {}

# Calculate proper yearly losses for all runs
all_yearly_losses_by_run = {}
for run_name in sorted(df_all['run_name'].unique()):
    run_data = df_all[df_all['run_name'] == run_name].iloc[0]
    N = run_data['N_fishermen']
    W = run_data['wages']
    
    yearly_losses_dict = calculate_proper_yearly_losses(run_name, N, W, min_days=1)
    if yearly_losses_dict:
        all_yearly_losses_by_run[run_name] = yearly_losses_dict
        print(f"{run_name}: {len(yearly_losses_dict)} yearly losses calculated")
    else:
        print(f"{run_name}: No yearly losses calculated")

# === Calculate Aggregated Observed Losses by Year ===
print(f"\n=== Calculating Aggregated Observed Losses ===")

if all_yearly_losses_by_run:
    # Find common years across all runs
    all_years_sets = [set(losses_dict.keys()) for losses_dict in all_yearly_losses_by_run.values()]
    common_years = set.intersection(*all_years_sets) if all_years_sets else set()
    print(f"Common years across all runs: {sorted(common_years)}")
    
    # Calculate aggregated observed losses for each common year
    aggregated_observed_losses = []
    for year in sorted(common_years):
        year_total = sum(losses_dict[year] for losses_dict in all_yearly_losses_by_run.values())
        aggregated_observed_losses.append(year_total)
        print(f"  {year}: ${year_total:,.0f} (sum of {len(all_yearly_losses_by_run)} runs)")
    
    aggregated_observed_losses = np.array(aggregated_observed_losses)
    print(f"Total aggregated observed yearly losses: {len(aggregated_observed_losses)}")
else:
    aggregated_observed_losses = np.array([])

# === Calculate Aggregated Simulated Losses ===
print(f"\n=== Calculating Aggregated Simulated Losses ===")

# We need to reconstruct simulated losses from the original AEP calculation
# Since we can't easily get the raw simulation data, let's approximate using the AEP curves
aggregated_simulated_losses = []

if len(aep_curve_files) > 0:
    print("Reconstructing simulated aggregate losses from AEP curves...")
    
    # Load all AEP curve data
    aep_data_by_run = {}
    for file in aep_curve_files:
        df_curve = pd.read_csv(file)
        run_name = df_curve['run_name'].iloc[0]
        aep_data_by_run[run_name] = df_curve
    
    # Find common probability points across all runs
    if aep_data_by_run:
        # Get the probability range that's common to all runs
        all_probs = [df['exceedance_probability'].values for df in aep_data_by_run.values()]
        
        # Use the first run's probabilities as reference and interpolate others
        reference_probs = sorted(aep_data_by_run[list(aep_data_by_run.keys())[0]]['exceedance_probability'].values)
        
        # For each probability, sum the losses across all runs
        aggregated_simulated_losses = []
        for prob in reference_probs:
            prob_total = 0
            for run_name, df_curve in aep_data_by_run.items():
                # Find the closest probability and get corresponding loss
                closest_idx = np.argmin(np.abs(df_curve['exceedance_probability'].values - prob))
                prob_total += df_curve['tp_loss'].iloc[closest_idx]
            aggregated_simulated_losses.append(prob_total)
        
        aggregated_simulated_losses = np.array(aggregated_simulated_losses)
        print(f"Aggregated simulated losses calculated: {len(aggregated_simulated_losses)} points")
    else:
        print("No AEP curve data available for simulation aggregation")
        aggregated_simulated_losses = np.array([])
else:
    print("No AEP curve files found")
    aggregated_simulated_losses = np.array([])

# === Create Aggregated Density Comparison Plot ===
print(f"\n=== Creating Aggregated Density Comparison Plot ===")

if len(aggregated_observed_losses) > 0 or len(aggregated_simulated_losses) > 0:
    plt.figure(figsize=(14, 10))
    
    # Plot observed aggregate losses
    if len(aggregated_observed_losses) > 0:
        plt.hist(aggregated_observed_losses, bins=min(15, len(aggregated_observed_losses)//2 + 1), 
                density=True, alpha=0.6, color='lightgreen', edgecolor='darkgreen', linewidth=2, 
                label=f'Observed Aggregate Losses ({len(aggregated_observed_losses)} years)')
        
        # Add KDE for observed
        if len(aggregated_observed_losses) > 3:
            from scipy.stats import gaussian_kde
            kde_obs = gaussian_kde(aggregated_observed_losses)
            x_range_obs = np.linspace(aggregated_observed_losses.min(), aggregated_observed_losses.max(), 200)
            plt.plot(x_range_obs, kde_obs(x_range_obs), color='darkgreen', linewidth=4, label='Observed KDE')
        
        # Statistics for observed
        mean_obs = np.mean(aggregated_observed_losses)
        median_obs = np.median(aggregated_observed_losses)
        p95_obs = np.percentile(aggregated_observed_losses, 95)
        p99_obs = np.percentile(aggregated_observed_losses, 99)
        
        plt.axvline(x=mean_obs, color='green', linestyle='--', linewidth=3, alpha=0.8,
                   label=f'Observed Mean: ${mean_obs:,.0f}')
        plt.axvline(x=p99_obs, color='darkgreen', linestyle=':', linewidth=3, alpha=0.8,
                   label=f'Observed P99: ${p99_obs:,.0f}')
    
    # Plot simulated aggregate losses
    if len(aggregated_simulated_losses) > 0:
        plt.hist(aggregated_simulated_losses, bins=min(30, len(aggregated_simulated_losses)//10), 
                density=True, alpha=0.4, color='lightblue', edgecolor='darkblue', linewidth=1,
                label=f'Simulated Aggregate Losses ({len(aggregated_simulated_losses)} simulations)')
        
        # Add KDE for simulated
        if len(aggregated_simulated_losses) > 3:
            kde_sim = gaussian_kde(aggregated_simulated_losses)
            x_range_sim = np.linspace(aggregated_simulated_losses.min(), aggregated_simulated_losses.max(), 200)
            plt.plot(x_range_sim, kde_sim(x_range_sim), color='darkblue', linewidth=4, label='Simulated KDE')
        
        # Statistics for simulated
        mean_sim = np.mean(aggregated_simulated_losses)
        p99_sim = np.percentile(aggregated_simulated_losses, 99)
        
        plt.axvline(x=mean_sim, color='blue', linestyle='--', linewidth=3, alpha=0.8,
                   label=f'Simulated Mean: ${mean_sim:,.0f}')
        plt.axvline(x=p99_sim, color='darkblue', linestyle=':', linewidth=3, alpha=0.8,
                   label=f'Simulated P99: ${p99_sim:,.0f}')
    
    plt.xlabel('Aggregate Annual Loss ($)', fontsize=14, fontweight='bold')
    plt.ylabel('Density', fontsize=14, fontweight='bold')
    plt.title('Distribution of Aggregate Annual Losses\n(Sum Across All Runs)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Format x-axis with currency
    def currency_formatter(x, pos):
        if x >= 1e6:
            return f'${x/1e6:.1f}M'
        elif x >= 1e3:
            return f'${x/1e3:.0f}K'
        else:
            return f'${x:.0f}'
    
    plt.gca().xaxis.set_major_formatter(FuncFormatter(currency_formatter))
    plt.tight_layout()
    
    # Save plot
    aggregate_comparison_path = os.path.join(summary_dir, 'aggregate_observed_vs_simulated_density.png')
    plt.savefig(aggregate_comparison_path, dpi=300, bbox_inches='tight')
    print(f"Saved aggregate comparison plot to: {aggregate_comparison_path}")
    plt.show()
    
    # Print comparison statistics
    print(f"\n=== Aggregate Losses Comparison Statistics ===")
    if len(aggregated_observed_losses) > 0:
        print(f"Observed Aggregate Losses:")
        print(f"  Years: {len(aggregated_observed_losses)}")
        print(f"  Mean: ${np.mean(aggregated_observed_losses):,.0f}")
        print(f"  Median: ${np.median(aggregated_observed_losses):,.0f}")
        print(f"  Std: ${np.std(aggregated_observed_losses):,.0f}")
        print(f"  Range: ${np.min(aggregated_observed_losses):,.0f} - ${np.max(aggregated_observed_losses):,.0f}")
        print(f"  P95: ${np.percentile(aggregated_observed_losses, 95):,.0f}")
        print(f"  P99: ${np.percentile(aggregated_observed_losses, 99):,.0f}")
    
    if len(aggregated_simulated_losses) > 0:
        print(f"\nSimulated Aggregate Losses:")
        print(f"  Simulations: {len(aggregated_simulated_losses)}")
        print(f"  Mean: ${np.mean(aggregated_simulated_losses):,.0f}")
        print(f"  Median: ${np.median(aggregated_simulated_losses):,.0f}")
        print(f"  Std: ${np.std(aggregated_simulated_losses):,.0f}")
        print(f"  Range: ${np.min(aggregated_simulated_losses):,.0f} - ${np.max(aggregated_simulated_losses):,.0f}")
        print(f"  P95: ${np.percentile(aggregated_simulated_losses, 95):,.0f}")
        print(f"  P99: ${np.percentile(aggregated_simulated_losses, 99):,.0f}")
    
    # Calculate ratio if both exist
    if len(aggregated_observed_losses) > 0 and len(aggregated_simulated_losses) > 0:
        obs_mean = np.mean(aggregated_observed_losses)
        sim_mean = np.mean(aggregated_simulated_losses)
        ratio = sim_mean / obs_mean if obs_mean > 0 else float('inf')
        print(f"\nSimulated/Observed Mean Ratio: {ratio:.2f}")

# === Create Aggregate AEP Curve ===
print(f"\n=== Creating Aggregate AEP Curve ===")

if len(aggregated_simulated_losses) > 0:
    # Sort simulated losses and create empirical CDF
    sorted_agg_losses = np.sort(aggregated_simulated_losses)
    n_simulations = len(sorted_agg_losses)
    
    # Calculate exceedance probabilities
    empirical_probs = np.arange(1, n_simulations + 1) / n_simulations
    exceedance_probs = 1 - empirical_probs
    
    # Create aggregate AEP DataFrame
    aggregate_aep_df = pd.DataFrame({
        'exceedance_probability': exceedance_probs,
        'aggregate_loss': sorted_agg_losses,
        'return_period': 1 / exceedance_probs
    })
    
    # Save aggregate AEP data
    aggregate_aep_path = os.path.join(summary_dir, 'aggregate_aep_curve_corrected.csv')
    aggregate_aep_df.to_csv(aggregate_aep_path, index=False)
    print(f"Saved corrected aggregate AEP curve to: {aggregate_aep_path}")
    
    # Plot aggregate AEP curve
    plt.figure(figsize=(12, 8))
    plt.plot(aggregate_aep_df['aggregate_loss'], aggregate_aep_df['exceedance_probability'], 
            color='darkgreen', linewidth=4, alpha=0.9, label='Aggregate AEP Curve')
    
    # Add percentile lines
    mean_loss = np.mean(sorted_agg_losses)
    p95_loss = np.percentile(sorted_agg_losses, 95)
    p99_loss = np.percentile(sorted_agg_losses, 99)
    
    plt.axvline(x=mean_loss, color='blue', linestyle='--', linewidth=3, alpha=0.7,
               label=f'Mean: ${mean_loss:,.0f}')
    plt.axvline(x=p95_loss, color='orange', linestyle='--', linewidth=3, alpha=0.7,
               label=f'P95: ${p95_loss:,.0f}')
    plt.axvline(x=p99_loss, color='red', linestyle='--', linewidth=3, alpha=0.7,
               label=f'P99: ${p99_loss:,.0f}')
    
    plt.xlabel('Aggregate Annual Loss ($)', fontsize=14, fontweight='bold')
    plt.ylabel('Exceedance Probability', fontsize=14, fontweight='bold')
    plt.title('Aggregate Annual Exceedance Probability Curve\n(Sum Across All Runs)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(currency_formatter))
    
    plt.tight_layout()
    
    # Save aggregate AEP plot
    aggregate_aep_plot_path = os.path.join(summary_dir, 'aggregate_aep_curve_corrected.png')
    plt.savefig(aggregate_aep_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved corrected aggregate AEP plot to: {aggregate_aep_plot_path}")
    plt.show()

# === Save Summary Data ===
if len(aggregated_observed_losses) > 0 and len(aggregated_simulated_losses) > 0:
    # Create summary DataFrame
    summary_data = {
        'metric': ['observed_mean', 'observed_p95', 'observed_p99', 'observed_max',
                  'simulated_mean', 'simulated_p95', 'simulated_p99', 'simulated_max'],
        'value': [
            np.mean(aggregated_observed_losses),
            np.percentile(aggregated_observed_losses, 95),
            np.percentile(aggregated_observed_losses, 99),
            np.max(aggregated_observed_losses),
            np.mean(aggregated_simulated_losses),
            np.percentile(aggregated_simulated_losses, 95),
            np.percentile(aggregated_simulated_losses, 99),
            np.max(aggregated_simulated_losses)
        ]
    }
    
    df_summary = pd.DataFrame(summary_data)
    summary_path = os.path.join(summary_dir, 'aggregate_losses_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved aggregate summary to: {summary_path}")

print("\n=== Aggregated Density Plots Complete ===")
#%%
# === Create professional scatterplot of recall vs fishermen ===
print(f"\n=== Creating Recall vs Fishermen Scatterplot ===")

# Load the concatenated Excel file
excel_file_path = os.path.join(summary_dir, 'aep_breakdown_all_runs.xlsx')
if os.path.exists(excel_file_path):
    df_all = pd.read_excel(excel_file_path)
    print(f"Loaded concatenated Excel file: {excel_file_path}")
    print(f"Total rows: {len(df_all)}")
    
    # Filter for TP rows only
    tp_data = df_all[df_all['type'] == 'tp'].copy()
    print(f"TP rows found: {len(tp_data)}")
    
    # Remove run_g1 and data points with N_fishermen close to 2000
    tp_data = tp_data[tp_data['run_name'] != 'run_g1'].copy()
    tp_data = tp_data[~((tp_data['N_fishermen'] >= 1800) & (tp_data['N_fishermen'] <= 2200))].copy()
    print(f"TP rows after filtering: {len(tp_data)}")
    
    if len(tp_data) > 0:
        # Create the scatterplot
        plt.figure(figsize=(12, 8))
        
        # Create color mapping for runs
        unique_runs = sorted(tp_data['run_name'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_runs)))
        
        # Create legend mapping
        legend_mapping = {
            'run_g1': 'Area 1', 'run_g2': 'Area 2', 'run_g3': 'Area 3', 'run_g4': 'Area 4', 'run_g5': 'Area 5',
            'run_g6': 'Area 6', 'run_g7': 'Area 7', 'run_g8': 'Area 8', 'run_g9': 'Area 9', 'run_g10': 'Area 10'
        }
        
        # Plot each run
        for i, run_name in enumerate(unique_runs):
            run_data = tp_data[tp_data['run_name'] == run_name]
            legend_label = legend_mapping.get(run_name, run_name)
            
            plt.scatter(run_data['N_fishermen'], run_data['recall'], 
                       c=[colors[i]], s=600, alpha=0.7, label=legend_label, edgecolors='white', linewidth=1)
        
        # Customize the plot
        plt.xlabel('Number of Fishermen', fontsize=14, fontweight='bold')
        plt.ylabel('Recall', fontsize=14, fontweight='bold')
        
        # Add grid
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Format x-axis with thousands separators
        from matplotlib.ticker import FuncFormatter
        def thousands_formatter(x, pos):
            return f'{int(x):,}'
        
        plt.gca().xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
        
        # Customize legend - single row at bottom
        plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=10, fontsize=12, frameon=True, 
                  fancybox=True, shadow=True)
        
        # Format axes
        plt.tight_layout()
        
        # Save the scatterplot
        scatterplot_path = os.path.join(summary_dir, 'recall_vs_fishermen_scatterplot.png')
        plt.savefig(scatterplot_path, dpi=300, bbox_inches='tight')
        print(f"Saved scatterplot to: {scatterplot_path}")
        
        # Show the plot
        plt.show()
        
        # Print summary statistics
        print(f"\n=== Scatterplot Summary ===")
        print(f"Recall range: {tp_data['recall'].min():.3f} - {tp_data['recall'].max():.3f}")
        print(f"Fishermen range: {tp_data['N_fishermen'].min():,.0f} - {tp_data['N_fishermen'].max():,.0f}")
        print(f"Mean recall: {tp_data['recall'].mean():.3f}")
        print(f"Mean fishermen: {tp_data['N_fishermen'].mean():,.0f}")
        
        # Print by area
        print(f"\n=== By Area ===")
        for run_name in unique_runs:
            run_data = tp_data[tp_data['run_name'] == run_name]
            legend_label = legend_mapping.get(run_name, run_name)
            print(f"{legend_label}: Recall = {run_data['recall'].iloc[0]:.3f}, Fishermen = {run_data['N_fishermen'].iloc[0]:,.0f}")
    else:
        print("No TP data found for scatterplot")
else:
    print(f"Concatenated Excel file not found: {excel_file_path}")

# %%

# %%
