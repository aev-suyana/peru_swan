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

# Add this code at the end of results_ppt.py

print(f"\n{'='*80}")
print("ADDING MULTI-CONDITION RESULTS TO SUMMARY")
print("="*80)

def extract_multi_condition_best_results(run_dirs):
    """Extract best multi-condition results from each run directory"""
    multi_condition_rows = []
    
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        print(f"\n🔍 Checking {run_name} for multi-condition results...")
        
        # Look for multi-condition result files
        multi_files = []
        multi_files.extend(sorted(glob.glob(os.path.join(run_dir, 'enhanced_multi_rule_summary_*.csv'))))
        multi_files.extend(sorted(glob.glob(os.path.join(run_dir, 'fast_multi_rule_summary_*.csv'))))
        multi_files.extend(sorted(glob.glob(os.path.join(run_dir, 'baseline_comparison_*.csv'))))
        
        if not multi_files:
            print(f"   ⚠️ No multi-condition files found")
            continue
        
        # Try to extract the best multi-condition result
        best_multi_result = None
        
        # Method 1: Try enhanced/fast multi-rule summary files
        for file_path in multi_files:
            if 'multi_rule_summary' in file_path:
                try:
                    df_multi = pd.read_csv(file_path)
                    if len(df_multi) > 0:
                        best_rule = df_multi.iloc[0]  # Best performing rule
                        
                        # Calculate confusion matrix for this rule
                        multi_cm = calculate_multi_rule_cm_for_run(run_dir, best_rule)
                        
                        if multi_cm:
                            best_multi_result = {
                                'run_dir': run_name,
                                'method': 'Multi-Condition',
                                'rule_type': best_rule.get('type', 'Unknown'),
                                'rule_description': best_rule.get('description', 'Unknown'),
                                'f1_score': float(best_rule.get('f1_score', 0)),
                                'mean_loss': float(best_rule.get('mean_loss', 0)),
                                'zero_prob': float(best_rule.get('zero_prob', 0)),
                                'thresholds': best_rule.get('thresholds', 'Unknown'),
                                **multi_cm,
                                'source_file': os.path.basename(file_path)
                            }
                            print(f"   ✅ Found multi-condition result: {best_rule.get('type', 'Unknown')}")
                            break
                except Exception as e:
                    print(f"   ⚠️ Error reading {file_path}: {e}")
                    continue
        
        # Method 2: Try baseline comparison files  
        if not best_multi_result:
            comparison_files = [f for f in multi_files if 'baseline_comparison' in f]
            for file_path in comparison_files:
                try:
                    df_comp = pd.read_csv(file_path)
                    multi_rows = df_comp[df_comp['analysis_type'].str.contains('multi_rule', na=False)]
                    if len(multi_rows) > 0:
                        best_comp = multi_rows.iloc[0]
                        best_multi_result = {
                            'run_dir': run_name,
                            'method': 'Multi-Condition',
                            'rule_type': 'From Comparison',
                            'rule_description': best_comp.get('description', 'Unknown'),
                            'f1_score': float(best_comp.get('f1_score', 0)),
                            'mean_loss': float(best_comp.get('mean_loss', 0)),
                            'zero_prob': 0,  # Not available in comparison
                            'obs_tp': 0, 'obs_fp': 0, 'obs_tn': 0, 'obs_fn': 0,  # Would need calculation
                            'obs_precision': 0, 'obs_recall': 0, 'obs_accuracy': 0, 'obs_f1': 0,
                            'source_file': os.path.basename(file_path)
                        }
                        print(f"   ✅ Found comparison result: {best_comp.get('description', 'Unknown')}")
                        break
                except Exception as e:
                    continue
        
        if best_multi_result:
            multi_condition_rows.append(best_multi_result)
        else:
            print(f"   ❌ No usable multi-condition results found")
    
    return multi_condition_rows

def calculate_multi_rule_cm_for_run(run_dir, rule_info):
    """Calculate confusion matrix for a multi-rule in a specific run"""
    try:
        # Try to find the merged data file for this run
        run_name = os.path.basename(run_dir)
        
        # Infer the data path based on run name
        base_data_dir = os.path.join(os.path.dirname(BASE_RESULTS_DIR), '..', 'wave_analysis_pipeline', 'data', 'processed')
        data_path = os.path.join(base_data_dir, run_name, 'df_swan_waverys_merged.csv')
        
        if not os.path.exists(data_path):
            print(f"     ⚠️ Data file not found: {data_path}")
            return None
        
        df_data = pd.read_csv(data_path, parse_dates=['date'])
        
        # Get rule details
        features = rule_info.get('features', [])
        thresholds = rule_info.get('thresholds', [])
        rule_type = rule_info.get('type', '')
        
        # Handle string representations
        if isinstance(features, str):
            import ast
            try:
                features = ast.literal_eval(features)
            except:
                # Try simple parsing
                features = features.strip('[]').replace("'", "").split(', ')
        
        if isinstance(thresholds, str):
            try:
                thresholds = ast.literal_eval(thresholds)
            except:
                def clean_float_string(x):
                    x = x.strip()
                    x = x.replace('np.float64(', '').replace(')', '')
                    x = x.lstrip('(').rstrip(')')
                    return float(x)
                thresholds = [clean_float_string(x) for x in thresholds.strip('[]').split(',') if x.strip()]
        
        if len(features) < 2 or len(thresholds) < 2:
            return None
        
        # Check if features exist
        missing_features = [f for f in features if f not in df_data.columns]
        if missing_features:
            return None
        
        # Apply rule logic
        if 'AND' in rule_type:
            rule_prediction = df_data[features[0]] > thresholds[0]
            for i in range(1, len(features)):
                rule_prediction = rule_prediction & (df_data[features[i]] > thresholds[i])
        elif 'OR' in rule_type:
            rule_prediction = df_data[features[0]] > thresholds[0]
            for i in range(1, len(features)):
                rule_prediction = rule_prediction | (df_data[features[i]] > thresholds[i])
        else:
            return None
        
        predictions = rule_prediction.astype(int)
        observed = df_data['event_dummy_1'].astype(int)
        
        # Calculate confusion matrix
        tp = int(np.sum((predictions == 1) & (observed == 1)))
        fp = int(np.sum((predictions == 1) & (observed == 0)))
        tn = int(np.sum((predictions == 0) & (observed == 0)))
        fn = int(np.sum((predictions == 0) & (observed == 1)))
        
        total = tp + fp + tn + fn
        
        return {
            'obs_tp': tp,
            'obs_fp': fp, 
            'obs_tn': tn,
            'obs_fn': fn,
            'obs_precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'obs_recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'obs_accuracy': (tp + tn) / total if total > 0 else 0,
            'obs_f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        }
        
    except Exception as e:
        print(f"     ❌ Error calculating CM: {e}")
        return None

# Enhanced metrics extraction functions to add to results_ppt.py

def extract_ml_simulation_metrics(run_dir):
    """Extract ML simulation metrics from AEP results"""
    ml_metrics = {
        'mean_loss': np.nan, 'p99_loss': np.nan, 'mean_events': np.nan, 'p99_events': np.nan,
        'obs_mean_events': np.nan, 'obs_mean_loss': np.nan,
        'mean_fp': np.nan, 'p99_fp': np.nan, 'mean_tp': np.nan, 'p99_tp': np.nan, 
        'mean_fn': np.nan, 'p99_fn': np.nan
    }
    
    # Look for ML AEP summary files
    ml_summary_files = []
    ml_summary_files.extend(sorted(glob.glob(os.path.join(run_dir, 'corrected_ml_aep_summary_*.csv'))))
    ml_summary_files.extend(sorted(glob.glob(os.path.join(run_dir, 'ml_aep_summary_*.csv'))))
    
    if ml_summary_files:
        try:
            ml_summary = pd.read_csv(ml_summary_files[-1])
            if len(ml_summary) > 0:
                row = ml_summary.iloc[0]
                ml_metrics['mean_loss'] = row.get('mean_loss', np.nan)
                ml_metrics['p99_loss'] = np.nan  # Calculate from curve if available
                ml_metrics['mean_events'] = row.get('mean_events_per_year', np.nan)
                
                # Look for ML AEP curve to calculate p99
                ml_curve_files = []
                ml_curve_files.extend(sorted(glob.glob(os.path.join(run_dir, 'corrected_ml_aep_curve_*.csv'))))
                ml_curve_files.extend(sorted(glob.glob(os.path.join(run_dir, 'ml_aep_curve_*.csv'))))
                
                if ml_curve_files:
                    ml_curve = pd.read_csv(ml_curve_files[-1])
                    if len(ml_curve) > 0:
                        # Calculate p99 loss (1% exceedance probability)
                        p99_mask = ml_curve['probability'] <= 0.01
                        if p99_mask.any():
                            ml_metrics['p99_loss'] = ml_curve.loc[p99_mask, 'loss'].min()
        except Exception as e:
            print(f"   Warning: Could not extract ML metrics for {os.path.basename(run_dir)}: {e}")
    
    # Look for observed yearly losses
    obs_files = sorted(glob.glob(os.path.join(run_dir, '*observed_yearly_losses*.csv')))
    if obs_files:
        try:
            obs_df = pd.read_csv(obs_files[-1])
            if 'observed_loss' in obs_df.columns:
                ml_metrics['obs_mean_loss'] = obs_df['observed_loss'].mean()
                # Count events per year (approximate from loss data)
                # Assuming average event cost, estimate event count
                avg_daily_wage = 50  # Approximate
                avg_fishermen = 100  # Approximate  
                avg_event_length = 3  # days
                event_cost = avg_fishermen * avg_daily_wage * avg_event_length
                if event_cost > 0:
                    ml_metrics['obs_mean_events'] = ml_metrics['obs_mean_loss'] / event_cost
        except Exception as e:
            print(f"   Warning: Could not extract observed metrics: {e}")
    
    return ml_metrics

def extract_multi_condition_simulation_metrics(run_dir):
    """Extract Multi-Condition simulation metrics from AEP results"""
    multi_metrics = {
        'mean_loss': np.nan, 'p99_loss': np.nan, 'mean_events': np.nan, 'p99_events': np.nan,
        'obs_mean_events': np.nan, 'obs_mean_loss': np.nan,
        'mean_fp': np.nan, 'p99_fp': np.nan, 'mean_tp': np.nan, 'p99_tp': np.nan, 
        'mean_fn': np.nan, 'p99_fn': np.nan
    }
    
    # Look for multi-condition AEP results
    multi_summary_files = []
    multi_summary_files.extend(sorted(glob.glob(os.path.join(run_dir, 'enhanced_multi_rule_summary_*.csv'))))
    multi_summary_files.extend(sorted(glob.glob(os.path.join(run_dir, 'fast_multi_rule_summary_*.csv'))))
    
    if multi_summary_files:
        try:
            multi_summary = pd.read_csv(multi_summary_files[-1])
            if len(multi_summary) > 0:
                best_rule = multi_summary.iloc[0]  # Best performing rule
                multi_metrics['mean_loss'] = best_rule.get('mean_loss', np.nan)
                multi_metrics['mean_events'] = best_rule.get('mean_events', np.nan)
                multi_metrics['p99_loss'] = best_rule.get('max_loss', np.nan)  # Approximate
                
                # Extract zero probability to estimate p99_events
                zero_prob = best_rule.get('zero_prob', np.nan)
                if not pd.isna(zero_prob) and not pd.isna(multi_metrics['mean_events']):
                    # Rough estimate: p99_events ≈ mean_events * (1 - zero_prob) * scaling_factor
                    multi_metrics['p99_events'] = multi_metrics['mean_events'] * (1 - zero_prob) * 2
        except Exception as e:
            print(f"   Warning: Could not extract multi-condition metrics for {os.path.basename(run_dir)}: {e}")
    
    # Use same observed metrics as other methods
    obs_files = sorted(glob.glob(os.path.join(run_dir, '*observed_yearly_losses*.csv')))
    if obs_files:
        try:
            obs_df = pd.read_csv(obs_files[-1])
            if 'observed_loss' in obs_df.columns:
                multi_metrics['obs_mean_loss'] = obs_df['observed_loss'].mean()
                # Estimate observed events
                avg_daily_wage = 50
                avg_fishermen = 100  
                avg_event_length = 3
                event_cost = avg_fishermen * avg_daily_wage * avg_event_length
                if event_cost > 0:
                    multi_metrics['obs_mean_events'] = multi_metrics['obs_mean_loss'] / event_cost
        except Exception:
            pass
    
    return multi_metrics

def extract_enhanced_single_rule_metrics(run_dir):
    """Extract complete Single Rule metrics including missing obs_mean_events and obs_mean_loss"""
    single_metrics = {
        'obs_mean_events': np.nan, 'obs_mean_loss': np.nan
    }
    
    # Look for observed yearly losses
    obs_files = sorted(glob.glob(os.path.join(run_dir, '*observed_yearly_losses*.csv')))
    if obs_files:
        try:
            obs_df = pd.read_csv(obs_files[-1])
            if 'observed_loss' in obs_df.columns:
                single_metrics['obs_mean_loss'] = obs_df['observed_loss'].mean()
                
                # Estimate observed events from loss data
                avg_daily_wage = 50  # Can be refined with actual wage data
                avg_fishermen = 100   
                avg_event_length = 3
                event_cost = avg_fishermen * avg_daily_wage * avg_event_length
                if event_cost > 0:
                    single_metrics['obs_mean_events'] = single_metrics['obs_mean_loss'] / event_cost
        except Exception as e:
            print(f"   Warning: Could not extract single rule observed metrics: {e}")
    
    return single_metrics

# Extract multi-condition results
multi_condition_results = extract_multi_condition_best_results(run_dirs)

if multi_condition_results:
    print(f"\n✅ Found multi-condition results for {len(multi_condition_results)} runs")
    
    # Create multi-condition summary DataFrame
    multi_df = pd.DataFrame(multi_condition_results)
    
    # Add totals row for multi-condition results
    numeric_cols_multi = multi_df.select_dtypes(include=[np.number]).columns
    summary_row_multi = multi_df[numeric_cols_multi].sum().to_dict()
    summary_row_multi.update({col: 'TOTAL' for col in multi_df.columns if col not in numeric_cols_multi})
    multi_df = pd.concat([multi_df, pd.DataFrame([summary_row_multi])], ignore_index=True)
    
    # Save multi-condition results
    multi_out_path = os.path.join(BASE_RESULTS_DIR, 'multi_condition_best_per_run_summary.xlsx')
    multi_df.to_excel(multi_out_path, index=False)
    print(f"✅ Saved multi-condition summary: {multi_out_path}")
    
    # Create combined comparison table
    print(f"\n📊 CREATING COMBINED COMPARISON TABLE")
    print("="*60)
    
    # Prepare best single-rule results for comparison
    single_rule_comparison = []
    for _, row in best_df.iterrows():
        if row['run_dir'] != 'TOTAL':
            single_rule_comparison.append({
                'run_dir': row['run_dir'],
                'method': 'Single Rule',
                'rule_type': 'Best Single',
                'rule_description': f"{row.get('variable', 'Unknown')} > {row.get('threshold_value', 'Unknown')}",
                'threshold': row.get('threshold_value', 'Unknown'),
                'mean_loss': row['mean_loss'],
                'obs_tp': row['obs_tp'],
                'obs_fp': row['obs_fp'], 
                'obs_tn': row['obs_tn'],
                'obs_fn': row['obs_fn'],
                'obs_precision': row['obs_precision'],
                'obs_recall': row['obs_recall'],
                'obs_accuracy': row['obs_accuracy'],
                'obs_f1': row['obs_f1']
            })
    
    # Prepare multi-condition results for comparison
    multi_rule_comparison = []
    for _, row in multi_df.iterrows():
        if row['run_dir'] != 'TOTAL':
            multi_rule_comparison.append({
                'run_dir': row['run_dir'],
                'method': 'Multi-Condition',
                'rule_type': row.get('rule_type', 'Unknown'),
                'rule_description': row.get('rule_description', 'Unknown'),
                'threshold': str(row.get('thresholds', 'Unknown')) if row.get('thresholds', 'Unknown') != 'Unknown' else 'Unknown',
                'mean_loss': row.get('mean_loss', 0),
                'obs_tp': row.get('obs_tp', 0),
                'obs_fp': row.get('obs_fp', 0),
                'obs_tn': row.get('obs_tn', 0), 
                'obs_fn': row.get('obs_fn', 0),
                'obs_precision': row.get('obs_precision', 0),
                'obs_recall': row.get('obs_recall', 0),
                'obs_accuracy': row.get('obs_accuracy', 0),
                'obs_f1': row.get('obs_f1', 0)
            })

    # --- Add ML results for each run ---
    ml_comparison = []
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        ml_path = os.path.join(run_dir, 'ML_probs_2024.csv')
        if not os.path.exists(ml_path):
            print(f"⚠️ ML_probs_2024.csv not found for {run_name}")
            continue
        ml_df = pd.read_csv(ml_path)
        # Use predicted_event and observed_event columns
        y_pred = ml_df['predicted_event']
        y_true = ml_df['observed_event']
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        # If you have a loss metric for ML, compute it here; else set to np.nan
        # Extract threshold from ML_probs_2024_optimal_threshold.txt if present
        threshold_txt_path = os.path.join(run_dir, 'ML_probs_2024_optimal_threshold.txt')
        threshold = 0.5
        if os.path.exists(threshold_txt_path):
            try:
                with open(threshold_txt_path, 'r') as f:
                    threshold = float(f.read().strip())
            except Exception:
                threshold = 0.5
        ml_comparison.append({
            'run_dir': run_name,
            'method': 'ML',
            'rule_type': f"{ml_df['feature_set'].iloc[0]} {ml_df['lr_variant'].iloc[0]}",
            'rule_description': 'ML Logistic Regression',
            'threshold': threshold,
            'mean_loss': np.nan,  # Replace with actual ML loss if available
            'obs_tp': tp, 'obs_fp': fp, 'obs_tn': tn, 'obs_fn': fn,
            'obs_precision': precision, 'obs_recall': recall,
            'obs_accuracy': accuracy, 'obs_f1': f1
        })

    # --- Combine all results ---
    all_comparison = single_rule_comparison + multi_rule_comparison + ml_comparison
    comparison_df = pd.DataFrame(all_comparison)
    
    
    # Add improvement calculation
    improvement_data = []
    run_groups = comparison_df.groupby('run_dir')
    
    for run_name, group in run_groups:
        single_row = group[group['method'] == 'Single Rule']
        multi_row = group[group['method'] == 'Multi-Condition']
        
        if len(single_row) > 0 and len(multi_row) > 0:
            single_loss = single_row.iloc[0]['mean_loss']
            multi_loss = multi_row.iloc[0]['mean_loss']
            single_f1 = single_row.iloc[0]['obs_f1']
            multi_f1 = multi_row.iloc[0]['obs_f1']
            
            # Add to single rule row
            single_data = single_row.iloc[0].copy()
            single_data['loss_improvement_vs_multi'] = ((multi_loss - single_loss) / single_loss * 100) if single_loss > 0 else 0
            single_data['f1_improvement_vs_multi'] = ((single_f1 - multi_f1) / multi_f1 * 100) if multi_f1 > 0 else 0
            improvement_data.append(single_data)
            
            # Add to multi rule row  
            multi_data = multi_row.iloc[0].copy()
            multi_data['loss_improvement_vs_multi'] = ((single_loss - multi_loss) / single_loss * 100) if single_loss > 0 else 0
            multi_data['f1_improvement_vs_multi'] = ((multi_f1 - single_f1) / single_f1 * 100) if single_f1 > 0 else 0
            improvement_data.append(multi_data)
    
    if improvement_data:
        improvement_df = pd.DataFrame(improvement_data)
        
        # Round numerical columns
        numeric_cols = ['mean_loss', 'obs_precision', 'obs_recall', 'obs_accuracy', 'obs_f1', 
                       'loss_improvement_vs_multi', 'f1_improvement_vs_multi']
        for col in numeric_cols:
            if col in improvement_df.columns:
                improvement_df[col] = improvement_df[col].round(3)
        
        # Save combined comparison
        combined_out_path = os.path.join(BASE_RESULTS_DIR, 'single_vs_multi_condition_comparison.xlsx')
        improvement_df.to_excel(combined_out_path, index=False)
        print(f"✅ Saved combined comparison: {combined_out_path}")
        
        # Print summary comparison
        print(f"\n📋 SINGLE RULE vs MULTI-CONDITION COMPARISON")
        print("="*80)
        
        # Show best performing method per run
        best_per_run = improvement_df.loc[improvement_df.groupby('run_dir')['obs_f1'].idxmax()]
        
        method_wins = best_per_run['method'].value_counts()
        print(f"\n🏆 METHOD WINS BY F1 SCORE:")
        for method, wins in method_wins.items():
            print(f"   {method}: {wins} runs")
        
        # Show average improvements
        single_rows = improvement_df[improvement_df['method'] == 'Single Rule']
        multi_rows = improvement_df[improvement_df['method'] == 'Multi-Condition']
        
        if len(single_rows) > 0 and len(multi_rows) > 0:
            avg_single_f1 = single_rows['obs_f1'].mean()
            avg_multi_f1 = multi_rows['obs_f1'].mean()
            avg_single_loss = single_rows['mean_loss'].mean()
            avg_multi_loss = multi_rows['mean_loss'].mean()
            
            print(f"\n📊 AVERAGE PERFORMANCE:")
            print(f"   Single Rule   - F1: {avg_single_f1:.3f}, Mean Loss: ${avg_single_loss:,.0f}")
            print(f"   Multi-Condition - F1: {avg_multi_f1:.3f}, Mean Loss: ${avg_multi_loss:,.0f}")
            
            f1_improvement = ((avg_multi_f1 - avg_single_f1) / avg_single_f1 * 100) if avg_single_f1 > 0 else 0
            loss_improvement = ((avg_single_loss - avg_multi_loss) / avg_single_loss * 100) if avg_single_loss > 0 else 0
            
            print(f"\n🎯 OVERALL IMPROVEMENT (Multi vs Single):")
            print(f"   F1 Score: {f1_improvement:+.1f}%")
            print(f"   Mean Loss: {loss_improvement:+.1f}%")
        
        # Show detailed comparison table
        display_cols = ['run_dir', 'method', 'rule_type', 'obs_f1', 'mean_loss', 'obs_tp', 'obs_fp', 'obs_fn']
        display_comparison = improvement_df[display_cols].copy()
        
        print(f"\n📋 DETAILED COMPARISON BY RUN:")
        print(display_comparison.to_string(index=False))
        
    else:
        print("⚠️ Could not create improvement comparison - insufficient data")

else:
    print("⚠️ No multi-condition results found in any runs")

print(f"\n✅ Multi-condition analysis complete!")
print("="*80)
# %%
# Only show confusion matrix and performance metrics for all methods by run
metrics_cols = [
    'run_dir', 'method', 'rule_type', 'rule_description', 'threshold',
    'obs_tp', 'obs_fp', 'obs_tn', 'obs_fn',
    'obs_precision', 'obs_recall', 'obs_accuracy', 'obs_f1',
    'mean_loss', 'p99_loss', 'mean_events', 'p99_events',
    'obs_mean_events', 'obs_mean_loss',
    'mean_fp', 'p99_fp', 'mean_tp', 'p99_tp', 'mean_fn', 'p99_fn'
]

# Fill in new metrics columns for each row in comparison_df
for idx, row in comparison_df.iterrows():
    method = row['method']
    run_dir = row['run_dir']
    # Defaults
    comparison_df.at[idx, 'mean_loss'] = np.nan
    comparison_df.at[idx, 'p99_loss'] = np.nan
    comparison_df.at[idx, 'mean_events'] = np.nan
    comparison_df.at[idx, 'p99_events'] = np.nan
    comparison_df.at[idx, 'obs_mean_events'] = np.nan
    comparison_df.at[idx, 'obs_mean_loss'] = np.nan
    comparison_df.at[idx, 'mean_fp'] = np.nan
    comparison_df.at[idx, 'p99_fp'] = np.nan
    comparison_df.at[idx, 'mean_tp'] = np.nan
    comparison_df.at[idx, 'p99_tp'] = np.nan
    comparison_df.at[idx, 'mean_fn'] = np.nan
    comparison_df.at[idx, 'p99_fn'] = np.nan

    if method == 'Single Rule':
        match = best_df[best_df['run_dir'] == run_dir]
        if not match.empty:
            match_row = match.iloc[0]
            comparison_df.at[idx, 'mean_loss'] = match_row.get('mean_loss', np.nan)
            comparison_df.at[idx, 'p99_loss'] = match_row.get('p99_loss', np.nan)
            comparison_df.at[idx, 'mean_events'] = match_row.get('mean_events', np.nan)
            comparison_df.at[idx, 'p99_events'] = match_row.get('p99_events', np.nan)
            comparison_df.at[idx, 'obs_mean_events'] = match_row.get('obs_mean_events', np.nan)
            comparison_df.at[idx, 'obs_mean_loss'] = match_row.get('obs_mean_loss', np.nan)
            comparison_df.at[idx, 'mean_fp'] = match_row.get('mean_fp', np.nan)
            comparison_df.at[idx, 'p99_fp'] = match_row.get('p99_fp', np.nan)
            comparison_df.at[idx, 'mean_tp'] = match_row.get('mean_tp', np.nan)
            comparison_df.at[idx, 'p99_tp'] = match_row.get('p99_tp', np.nan)
            comparison_df.at[idx, 'mean_fn'] = match_row.get('mean_fn', np.nan)
            comparison_df.at[idx, 'p99_fn'] = match_row.get('p99_fn', np.nan)
    elif method == 'Multi-Condition':
        match = multi_df[multi_df['run_dir'] == run_dir]
        if not match.empty:
            match_row = match.iloc[0]
            comparison_df.at[idx, 'mean_loss'] = match_row.get('mean_loss', np.nan)
            comparison_df.at[idx, 'p99_loss'] = match_row.get('p99_loss', np.nan)
            comparison_df.at[idx, 'mean_events'] = match_row.get('mean_events', np.nan)
            comparison_df.at[idx, 'p99_events'] = match_row.get('p99_events', np.nan)
            comparison_df.at[idx, 'obs_mean_events'] = match_row.get('obs_mean_events', np.nan)
            comparison_df.at[idx, 'obs_mean_loss'] = match_row.get('obs_mean_loss', np.nan)
            comparison_df.at[idx, 'mean_fp'] = match_row.get('mean_fp', np.nan)
            comparison_df.at[idx, 'p99_fp'] = match_row.get('p99_fp', np.nan)
            comparison_df.at[idx, 'mean_tp'] = match_row.get('mean_tp', np.nan)
            comparison_df.at[idx, 'p99_tp'] = match_row.get('p99_tp', np.nan)
            comparison_df.at[idx, 'mean_fn'] = match_row.get('mean_fn', np.nan)
            comparison_df.at[idx, 'p99_fn'] = match_row.get('p99_fn', np.nan)
    # For ML, keep as np.nan unless you have simulation results for ML

# comparison_df = fill_enhanced_metrics(comparison_df, best_df, multi_df)

# # BEFORE trying to select metrics_cols, add the missing columns:
# print("🔧 Adding missing metrics columns...")
# missing_cols = [
#     'p99_loss', 'mean_events', 'p99_events', 'obs_mean_events', 'obs_mean_loss',
#     'mean_fp', 'p99_fp', 'mean_tp', 'p99_tp', 'mean_fn', 'p99_fn'
# ]

# for col in missing_cols:
#     if col not in comparison_df.columns:
#         comparison_df[col] = np.nan
# UPDATED fill_enhanced_metrics function - replace the existing one
def fill_enhanced_metrics_complete(comparison_df, best_df, multi_df, run_dirs):
    """Complete enhanced metrics filling with ALL simulation data"""
    print("🔧 Filling COMPLETE enhanced metrics from simulation outputs...")
    
    for idx, row in comparison_df.iterrows():
        method = row['method']
        run_dir_name = row['run_dir']
        
        # Find the full run directory path
        run_dir_path = None
        for full_path in run_dirs:
            if os.path.basename(full_path) == run_dir_name:
                run_dir_path = full_path
                break
        
        if not run_dir_path:
            continue
            
        print(f"  Processing {run_dir_name} - {method}...")
        
        if method == 'Single Rule':
            # Single Rule metrics are already mostly complete - just fill any gaps
            match = best_df[best_df['run_dir'] == run_dir_name]
            if not match.empty:
                match_row = match.iloc[0]
                # Fill any missing columns
                for col in ['mean_events', 'p99_events', 'mean_fp', 'p99_fp', 'mean_tp', 'p99_tp', 'mean_fn', 'p99_fn']:
                    if col in match_row and not pd.isna(match_row[col]):
                        comparison_df.at[idx, col] = match_row[col]
                        
        elif method == 'Multi-Condition':
            # Extract comprehensive multi-condition metrics
            multi_metrics = extract_complete_multi_condition_metrics(run_dir_path)
            for col, value in multi_metrics.items():
                if not pd.isna(value):
                    comparison_df.at[idx, col] = value
                    
        elif method == 'ML':
            # Extract comprehensive ML metrics
            ml_metrics = extract_complete_ml_metrics(run_dir_path)
            for col, value in ml_metrics.items():
                if not pd.isna(value):
                    comparison_df.at[idx, col] = value
    
    # Final step: estimate any remaining missing values
    print("🔧 Estimating remaining missing metrics...")
    comparison_df = estimate_missing_metrics_from_economics(comparison_df)
    
    return comparison_df

def extract_complete_multi_condition_metrics(run_dir):
    """Extract comprehensive multi-condition metrics from simulation outputs"""
    metrics = {
        'mean_loss': np.nan, 'p99_loss': np.nan, 'mean_events': np.nan, 'p99_events': np.nan,
        'obs_mean_events': np.nan, 'obs_mean_loss': np.nan,
        'mean_fp': np.nan, 'p99_fp': np.nan, 'mean_tp': np.nan, 'p99_tp': np.nan, 
        'mean_fn': np.nan, 'p99_fn': np.nan
    }
    
    # Look for multi-condition AEP results
    multi_summary_files = []
    multi_summary_files.extend(sorted(glob.glob(os.path.join(run_dir, 'enhanced_multi_rule_summary_*.csv'))))
    multi_summary_files.extend(sorted(glob.glob(os.path.join(run_dir, 'fast_multi_rule_summary_*.csv'))))
    
    if multi_summary_files:
        try:
            multi_summary = pd.read_csv(multi_summary_files[-1])
            if len(multi_summary) > 0:
                best_rule = multi_summary.iloc[0]  # Best performing rule
                metrics['mean_loss'] = best_rule.get('mean_loss', np.nan)
                metrics['mean_events'] = best_rule.get('mean_events', np.nan)
                metrics['p99_loss'] = best_rule.get('max_loss', np.nan)  # Approximate
                
                # Extract zero probability to estimate p99_events
                zero_prob = best_rule.get('zero_prob', np.nan)
                if not pd.isna(zero_prob) and not pd.isna(metrics['mean_events']):
                    # Rough estimate: p99_events ≈ mean_events * (1 - zero_prob) * scaling_factor
                    metrics['p99_events'] = metrics['mean_events'] * (1 - zero_prob) * 2
                    
                # Try to extract confusion matrix metrics if available
                for metric in ['mean_fp', 'mean_tp', 'mean_fn']:
                    if metric in best_rule:
                        metrics[metric] = best_rule[metric]
        except Exception as e:
            print(f"   Warning: Could not extract multi-condition metrics for {os.path.basename(run_dir)}: {e}")
    
    # Use observed metrics from any available source
    obs_files = sorted(glob.glob(os.path.join(run_dir, '*observed_yearly_losses*.csv')))
    if obs_files:
        try:
            obs_df = pd.read_csv(obs_files[-1])
            if 'observed_loss' in obs_df.columns:
                metrics['obs_mean_loss'] = obs_df['observed_loss'].mean()
                # Estimate observed events
                avg_daily_wage = 50
                avg_fishermen = 100  
                avg_event_length = 3
                event_cost = avg_fishermen * avg_daily_wage * avg_event_length
                if event_cost > 0:
                    metrics['obs_mean_events'] = metrics['obs_mean_loss'] / event_cost
        except Exception:
            pass
    
    return metrics

def extract_complete_ml_metrics(run_dir):
    """Extract comprehensive ML metrics from simulation outputs"""
    metrics = {
        'mean_loss': np.nan, 'p99_loss': np.nan, 'mean_events': np.nan, 'p99_events': np.nan,
        'obs_mean_events': np.nan, 'obs_mean_loss': np.nan,
        'mean_fp': np.nan, 'p99_fp': np.nan, 'mean_tp': np.nan, 'p99_tp': np.nan, 
        'mean_fn': np.nan, 'p99_fn': np.nan
    }
    
    # Look for ML AEP summary files
    ml_summary_files = []
    ml_summary_files.extend(sorted(glob.glob(os.path.join(run_dir, 'corrected_ml_aep_summary_*.csv'))))
    ml_summary_files.extend(sorted(glob.glob(os.path.join(run_dir, 'ml_aep_summary_*.csv'))))
    
    if ml_summary_files:
        try:
            ml_summary = pd.read_csv(ml_summary_files[-1])
            if len(ml_summary) > 0:
                row = ml_summary.iloc[0]
                metrics['mean_loss'] = row.get('mean_loss', np.nan)
                metrics['mean_events'] = row.get('mean_events_per_year', np.nan)
                
                # Look for ML AEP curve to calculate p99
                ml_curve_files = []
                ml_curve_files.extend(sorted(glob.glob(os.path.join(run_dir, 'corrected_ml_aep_curve_*.csv'))))
                ml_curve_files.extend(sorted(glob.glob(os.path.join(run_dir, 'ml_aep_curve_*.csv'))))
                
                if ml_curve_files:
                    ml_curve = pd.read_csv(ml_curve_files[-1])
                    if len(ml_curve) > 0:
                        # Calculate p99 loss (1% exceedance probability)
                        p99_mask = ml_curve['probability'] <= 0.01
                        if p99_mask.any():
                            metrics['p99_loss'] = ml_curve.loc[p99_mask, 'loss'].min()
                
                # Try to extract confusion matrix metrics if available
                for metric in ['mean_fp', 'mean_tp', 'mean_fn']:
                    if metric in row:
                        metrics[metric] = row[metric]
        except Exception as e:
            print(f"   Warning: Could not extract ML metrics for {os.path.basename(run_dir)}: {e}")
    
    # Look for observed yearly losses
    obs_files = sorted(glob.glob(os.path.join(run_dir, '*observed_yearly_losses*.csv')))
    if obs_files:
        try:
            obs_df = pd.read_csv(obs_files[-1])
            if 'observed_loss' in obs_df.columns:
                metrics['obs_mean_loss'] = obs_df['observed_loss'].mean()
                # Count events per year (approximate from loss data)
                avg_daily_wage = 50  # Approximate
                avg_fishermen = 100  # Approximate  
                avg_event_length = 3  # days
                event_cost = avg_fishermen * avg_daily_wage * avg_event_length
                if event_cost > 0:
                    metrics['obs_mean_events'] = metrics['obs_mean_loss'] / event_cost
        except Exception as e:
            print(f"   Warning: Could not extract observed metrics: {e}")
    
    return metrics

def estimate_missing_metrics_from_economics(comparison_df):
    """Estimate any missing metrics based on economic assumptions"""
    # Define economic parameters
    avg_daily_wage = 50  # Approximate daily wage
    avg_fishermen = 100  # Approximate number of fishermen affected
    avg_event_length = 3  # Average event duration in days
    event_cost = avg_daily_wage * avg_fishermen * avg_event_length
    
    for idx, row in comparison_df.iterrows():
        # If we have mean_loss but not mean_events, estimate events
        if pd.notna(row.get('mean_loss')) and pd.isna(row.get('mean_events')) and event_cost > 0:
            comparison_df.at[idx, 'mean_events'] = row['mean_loss'] / event_cost
            
        # If we have mean_events but not mean_loss, estimate loss
        if pd.isna(row.get('mean_loss')) and pd.notna(row.get('mean_events')):
            comparison_df.at[idx, 'mean_loss'] = row['mean_events'] * event_cost
            
        # If we have obs_mean_loss but not obs_mean_events, estimate events
        if pd.notna(row.get('obs_mean_loss')) and pd.isna(row.get('obs_mean_events')) and event_cost > 0:
            comparison_df.at[idx, 'obs_mean_events'] = row['obs_mean_loss'] / event_cost
            
        # If we have obs_mean_events but not obs_mean_loss, estimate loss
        if pd.isna(row.get('obs_mean_loss')) and pd.notna(row.get('obs_mean_events')):
            comparison_df.at[idx, 'obs_mean_loss'] = row['obs_mean_events'] * event_cost
            
        # If we have mean metrics but not p99 metrics, make rough estimates
        if pd.notna(row.get('mean_loss')) and pd.isna(row.get('p99_loss')):
            # Rough estimate: p99 ≈ mean * scaling_factor
            comparison_df.at[idx, 'p99_loss'] = row['mean_loss'] * 2.5
            
        if pd.notna(row.get('mean_events')) and pd.isna(row.get('p99_events')):
            # Rough estimate: p99 ≈ mean * scaling_factor
            comparison_df.at[idx, 'p99_events'] = row['mean_events'] * 2.5
    
    return comparison_df

# THEN fill the enhanced metrics (add the functions from the artifact)
print("🔧 Filling enhanced metrics from simulation outputs...")
comparison_df = fill_enhanced_metrics_complete(comparison_df, best_df, multi_df, run_dirs)

# Create individual run metrics table
individual_metrics_df = comparison_df[metrics_cols].copy()
individual_metrics_df.sort_values(by=['run_dir', 'method'], inplace=True)
print("\n=== INDIVIDUAL RUN METRICS: Confusion Matrix and Performance Metrics by Run and Method ===")
print(individual_metrics_df.to_string(index=False))

# Save individual run metrics
individual_metrics_df.to_excel(os.path.join(BASE_RESULTS_DIR, 'individual_metrics_by_run_and_method.xlsx'), index=False)
print(f"\n✅ Saved individual metrics by run and method: {os.path.join(BASE_RESULTS_DIR, 'individual_metrics_by_run_and_method.xlsx')}")

# %%
# Create aggregated metrics table by run group (mean and p99 across 3 runs)
print("\n=== CREATING AGGREGATED METRICS BY RUN GROUP ===")

# Extract run group from run_dir (assuming format like 'run_g1', 'run_g2', etc.)
def extract_run_group(run_dir):
    """Extract run group from run directory name"""
    run_name = os.path.basename(run_dir)
    # Extract group number from run_gX format
    if 'run_g' in run_name:
        return run_name.split('_')[0] + '_' + run_name.split('_')[1]  # e.g., 'run_g1'
    return run_name

# Add run group column
comparison_df['run_group'] = comparison_df['run_dir'].apply(extract_run_group)

# Define metrics to aggregate
agg_metrics = ['mean_loss', 'p99_loss', 'mean_events', 'p99_events', 'obs_mean_events', 'obs_mean_loss']

# Create aggregated metrics by run group and method
aggregated_data = []

for method in comparison_df['method'].unique():
    method_data = comparison_df[comparison_df['method'] == method]
    
    for run_group in method_data['run_group'].unique():
        group_data = method_data[method_data['run_group'] == run_group]
        
        if len(group_data) == 0:
            continue
            
        # Calculate aggregated metrics
        agg_row = {
            'run_group': run_group,
            'method': method,
            'rule_type': group_data['rule_type'].iloc[0] if 'rule_type' in group_data.columns else '',
            'rule_description': group_data['rule_description'].iloc[0] if 'rule_description' in group_data.columns else '',
            'num_runs': len(group_data),
        }
        
        # Calculate mean and p99 for each metric across runs
        for metric in agg_metrics:
            if metric in group_data.columns:
                values = group_data[metric].dropna()
                if len(values) > 0:
                    agg_row[f'{metric}_mean'] = values.mean()
                    agg_row[f'{metric}_p99'] = values.quantile(0.99) if len(values) > 1 else values.iloc[0]
                else:
                    agg_row[f'{metric}_mean'] = np.nan
                    agg_row[f'{metric}_p99'] = np.nan
            else:
                agg_row[f'{metric}_mean'] = np.nan
                agg_row[f'{metric}_p99'] = np.nan
        
        # Also calculate mean confusion matrix metrics
        cm_metrics = ['obs_tp', 'obs_fp', 'obs_tn', 'obs_fn', 'obs_precision', 'obs_recall', 'obs_accuracy', 'obs_f1']
        for metric in cm_metrics:
            if metric in group_data.columns:
                values = group_data[metric].dropna()
                if len(values) > 0:
                    agg_row[f'{metric}_mean'] = values.mean()
                else:
                    agg_row[f'{metric}_mean'] = np.nan
            else:
                agg_row[f'{metric}_mean'] = np.nan
        
        aggregated_data.append(agg_row)

# Create the final aggregated metrics DataFrame
metrics_df = pd.DataFrame(aggregated_data)

# Sort by run group and method
metrics_df.sort_values(by=['run_group', 'method'], inplace=True)
metrics_df.reset_index(drop=True, inplace=True)

# Round numerical columns for better display
numeric_cols = [col for col in metrics_df.columns if col.endswith('_mean') or col.endswith('_p99')]
for col in numeric_cols:
    metrics_df[col] = metrics_df[col].round(6)

print("\n=== FINAL AGGREGATED METRICS: Mean and P99 Losses and Events by Run Group ===")
print(metrics_df.to_string(index=False))

# Save the aggregated metrics
metrics_df.to_excel(os.path.join(BASE_RESULTS_DIR, 'aggregated_metrics_by_run_group.xlsx'), index=False)
print(f"\n✅ Saved aggregated metrics by run group: {os.path.join(BASE_RESULTS_DIR, 'aggregated_metrics_by_run_group.xlsx')}")
# %%
import matplotlib.pyplot as plt
import numpy as np

# Data from the table
areas = ['run_g9', 'run_g10', 'run_g8', 'run_g4', 'run_g7', 'run_g5', 'run_g1', 'run_g1', 'run_g6', 'run_g3', 'run_g3', 'run_g2', 'run_g2']
n_fishermen = [3290, 3290, 3948, 5236, 4606, 3853, 3948, 3948, 2827, 14424, 14424, 2109, 2109]
f1_scores = [0.73, 0.64, 0.62, 0.54, 0.51, 0.46, 0.62, 0.46, 0.27, 0.15, 0.36, 0.51, 0.30]
conditions = ['Single', 'Single', 'Single', 'Single', 'Single', 'Single', 'Single', 'ML', 'Multi', 'Multi', 'Multi', 'ML', 'Multi']

# Create color mapping for conditions
condition_colors = {'Single': 'blue', 'Multi': 'red', 'ML': 'green'}
colors = [condition_colors[condition] for condition in conditions]

# Create the scatter plot with point size based on n_fishermen
plt.figure(figsize=(12, 8))
for condition in condition_colors:
    mask = [c == condition for c in conditions]
    x_vals = [n_fishermen[i] for i in range(len(mask)) if mask[i]]
    y_vals = [f1_scores[i] for i in range(len(mask)) if mask[i]]
    sizes = [n_fishermen[i]/50 for i in range(len(mask)) if mask[i]]  # Scale down for better visualization
    plt.scatter(x_vals, y_vals, c=condition_colors[condition], label=condition, alpha=0.7, s=sizes)

# Add labels for each point with manual positioning to avoid overlaps
label_positions = {
    'run_g9': (3290, 0.73, 'right', 'bottom'),
    'run_g10': (3290, 0.64, 'left', 'top'),
    'run_g8': (3948, 0.62, 'right', 'top'),
    'run_g4': (5236, 0.54, 'center', 'bottom'),
    'run_g7': (4606, 0.51, 'center', 'bottom'),
    'run_g5': (3853, 0.46, 'left', 'bottom'),
    'run_g1': (3948, 0.62, 'left', 'bottom'),  # Single condition
    'run_g6': (2827, 0.27, 'right', 'top'),
    'run_g3': (14424, 0.15, 'center', 'top'),  # Single condition
    'run_g2': (2109, 0.51, 'left', 'top'),  # ML condition
}

for i, area in enumerate(areas):
    x, y = n_fishermen[i], f1_scores[i]
    
    # Create a unique key for duplicate areas based on condition
    if area in ['run_g1', 'run_g2', 'run_g3'] and conditions[i] != 'Single':
        key = f"{area}_{conditions[i]}"
    else:
        key = area
    
    # Use custom positioning if available, otherwise use default
    if area in label_positions:
        _, _, ha, va = label_positions[area]
        if ha == 'right':
            offset_x = x * 0.015
        elif ha == 'left':
            offset_x = -x * 0.015
        else:
            offset_x = 0
            
        if va == 'top':
            offset_y = -f1_scores[i] * 0.03
        elif va == 'bottom':
            offset_y = f1_scores[i] * 0.03
        else:
            offset_y = 0
    else:
        # Default positioning
        offset_x = x * 0.02
        offset_y = f1_scores[i] * 0.02
        ha, va = 'left', 'bottom'
    
    # Add condition suffix for duplicate areas
    label = area
    if area in ['run_g1', 'run_g2', 'run_g3']:
        if conditions[i] == 'ML':
            label = f"{area} (ML)"
        elif conditions[i] == 'Multi':
            label = f"{area} (Multi)"
    
    plt.annotate(label, (x, y), 
                xytext=(x + offset_x, y + offset_y),
                fontsize=12, ha=ha if area in label_positions else 'left', 
                va=va if area in label_positions else 'bottom',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.5, edgecolor='none'))

# Customize the plot
plt.xlabel('Number of Fishermen', fontsize=12)
plt.ylabel('F1 Score', fontsize=12)
plt.title('F1 Score vs Number of Fishermen by Condition\n(Point size proportional to number of fishermen)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Add legend for point sizes
plt.figtext(0.02, 0.02, 'Point size represents number of fishermen', fontsize=10, style='italic')

plt.tight_layout()
plt.show()

# Print some basic statistics
print("Summary Statistics:")
print(f"Average F1 Score by Condition:")
for condition in condition_colors:
    scores = [f1_scores[i] for i in range(len(conditions)) if conditions[i] == condition]
    print(f"  {condition}: {np.mean(scores):.3f}")

print(f"\nCorrelation between N_Fishermen and F1_Score: {np.corrcoef(n_fishermen, f1_scores)[0,1]:.3f}")
# %%
