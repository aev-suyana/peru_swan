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
    
    # Combine all results
    all_comparison = single_rule_comparison + multi_rule_comparison
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
