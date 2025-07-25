def prepare_data(df, target_column=None):"""
SIMPLIFIED RULE EVALUATION - FOCUSED ON SWH_MAX_SWAN FEATURES
============================================================

Simplified version that focuses on swh_max_swan and anom_swh_max_swan related features
with only single and double condition rules, comprehensive output, and no feature selection.

Author: Wave Analysis Team
Date: 2025
"""

import os
import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

from gp_config import config, get_input_files, get_output_files

# ============================================================================
# gp_configURATION
# ============================================================================

class SimplifiedRuleConfig:
    """Centralized gp_configuration for simplified rule evaluation"""
    N_FOLDS = 6
    USE_TIME_SERIES_CV = True
    N_JOBS = 10  # Parallel processing cores
    RANDOM_STATE = 42
    
    # Target column gp_configuration
    DEFAULT_TARGET_COLUMN = 'event_dummy_5'
    
    # Rule evaluation parameters
    MIN_FOLDS_SUCCESS = 4  # Minimum successful folds for rule to be considered valid
    
    # Threshold grid for optimization
    THRESHOLD_PERCENTILES = [50, 60, 70, 75, 80, 85, 90, 95, 99]

# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data(df):
    """Prepare data for rule evaluation"""
    print("📊 Preparing data for rule evaluation...")
    
    # Ensure proper data types
    df = df.copy()
    
    # Handle date column/index
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    elif df.index.name == 'date' or isinstance(df.index, pd.DatetimeIndex):
        # Date is the index, reset it to a column
        df = df.reset_index()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'index' in df.columns:
            df = df.rename(columns={'index': 'date'})
            df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    else:
        # No date column found, create a simple index-based date
        print("⚠️ No date column found, creating sequential dates")
        df['date'] = pd.date_range(start='2018-01-01', periods=len(df), freq='D')
    
    # Validate target column
    if 'event_dummy_1' not in df.columns:
        raise ValueError("Target column 'event_dummy_1' not found!")
    
    event_count = df['event_dummy_1'].sum()
    event_rate = df['event_dummy_1'].mean()
    
    print(f"✅ Data prepared: {len(df)} samples, {event_count} events ({event_rate*100:.1f}%)")
    
    return df

def get_target_features(df):
    """Get features that are related to SWAN and contain MAX"""
    print("🔍 Finding features that are SWAN-related AND contain MAX...")
    
    # Look for features that contain both "swh" AND "max" AND have "_swan" suffix
    # This ensures we only get SWAN features (not WAVERYS features)
    target_features = []
    
    for col in df.columns:
        # Check if column is numeric
        if col in df.columns and df[col].dtype in ['float64', 'int64']:
            # AND condition: must contain both "swh" AND "max" AND "_swan"
            contains_swh = 'swh' in col.lower()
            contains_max = 'max' in col.lower()
            contains_swan = '_swan' in col.lower()
            
            if contains_swh and contains_max and contains_swan:
                target_features.append(col)
    
    # Sort features for better readability
    target_features.sort()
    
    print(f"✅ Found {len(target_features)} features (SWAN + MAX):")
    for i, feature in enumerate(target_features, 1):
        print(f"  {i:2d}. {feature}")
    
    if len(target_features) == 0:
        raise ValueError("No features found that contain both 'swh', 'max', and '_swan'! Check column names.")
    
    # Filter out unwanted feature types
    excluded_patterns = ['_lag_', '_deseasonalized', '_detrended', '_rolling_mean', '_persistence_']
    
    filtered_features = []
    excluded_features = []
    
    for feature in target_features:
        should_exclude = any(pattern in feature.lower() for pattern in excluded_patterns)
        if should_exclude:
            excluded_features.append(feature)
        else:
            filtered_features.append(feature)
    
    # Show breakdown of feature types
    basic_features = [f for f in filtered_features if not any(suffix in f for suffix in ['_ma_', '_trend_', '_change_'])]
    enhanced_features = [f for f in filtered_features if any(suffix in f for suffix in ['_ma_', '_trend_', '_change_'])]
    
    print(f"\n📊 Feature breakdown:")
    print(f"  Basic features: {len(basic_features)}")
    print(f"  Enhanced features: {len(enhanced_features)}")
    print(f"  Excluded features: {len(excluded_features)}")
    
    if len(enhanced_features) > 0:
        print(f"\n📋 Sample enhanced features (after filtering):")
        for feature in sorted(enhanced_features)[:10]:
            print(f"    - {feature}")
        if len(enhanced_features) > 10:
            print(f"    ... and {len(enhanced_features) - 10} more")
    
    if len(excluded_features) > 0:
        print(f"\n🚫 Sample excluded features:")
        for feature in sorted(excluded_features)[:5]:
            print(f"    - {feature}")
        if len(excluded_features) > 5:
            print(f"    ... and {len(excluded_features) - 5} more")
    
    print(f"\n✅ Using {len(filtered_features)} features for rule evaluation")
    
    return filtered_features

# ============================================================================
# RULE GENERATION
# ============================================================================

def generate_simple_rules(features):
    """Generate single and double condition rules only"""
    print(f"🔧 Generating rules from {len(features)} features...")
    
    rules = []
    
    # 1. Single feature rules
    for feature in features:
        rules.append({
            'type': 'single',
            'feature': feature,
            'name': f'{feature} > threshold',
            'complexity': 1
        })
    
    print(f"✅ Added {len(features)} single-feature rules")
    
    # 2. Double feature rules (AND conditions)
    and_count = 0
    for i, feature1 in enumerate(features):
        for j, feature2 in enumerate(features[i+1:], i+1):
            rules.append({
                'type': 'and',
                'feature1': feature1,
                'feature2': feature2,
                'name': f'({feature1} > t1) AND ({feature2} > t2)',
                'complexity': 2
            })
            and_count += 1
    
    print(f"✅ Added {and_count} double-feature AND rules")
    
    # 3. Double feature rules (OR conditions)
    or_count = 0
    for i, feature1 in enumerate(features):
        for j, feature2 in enumerate(features[i+1:], i+1):
            rules.append({
                'type': 'or',
                'feature1': feature1,
                'feature2': feature2,
                'name': f'({feature1} > t1) OR ({feature2} > t2)',
                'complexity': 2
            })
            or_count += 1
    
    print(f"✅ Added {or_count} double-feature OR rules")
    
    total_rules = len(rules)
    print(f"🎯 Total rules generated: {total_rules}")
    print(f"   Single rules: {len(features)}")
    print(f"   Double rules: {and_count + or_count}")
    
    return rules

# ============================================================================
# RULE APPLICATION
# ============================================================================

def apply_single_condition(df, feature, threshold):
    """Apply single feature condition with robust error handling"""
    if feature not in df.columns:
        return pd.Series(False, index=df.index)
    
    if threshold is None or pd.isna(threshold):
        return pd.Series(False, index=df.index)
    
    feature_values = df[feature].fillna(0)
    return feature_values > threshold

def apply_and_condition(df, feature1, threshold1, feature2, threshold2):
    """Apply AND condition with robust error handling"""
    if any(f not in df.columns for f in [feature1, feature2]):
        return pd.Series(False, index=df.index)
    
    if any(pd.isna(t) or t is None for t in [threshold1, threshold2]):
        return pd.Series(False, index=df.index)
    
    cond1 = df[feature1].fillna(0) > threshold1
    cond2 = df[feature2].fillna(0) > threshold2
    return cond1 & cond2

def apply_or_condition(df, feature1, threshold1, feature2, threshold2):
    """Apply OR condition with robust error handling"""
    if any(f not in df.columns for f in [feature1, feature2]):
        return pd.Series(False, index=df.index)
    
    if any(pd.isna(t) or t is None for t in [threshold1, threshold2]):
        return pd.Series(False, index=df.index)
    
    cond1 = df[feature1].fillna(0) > threshold1
    cond2 = df[feature2].fillna(0) > threshold2
    return cond1 | cond2

def apply_rule(df, rule, thresholds):
    """Apply rule based on type with comprehensive error handling"""
    try:
        if rule['type'] == 'single':
            return apply_single_condition(df, rule['feature'], thresholds.get(rule['feature']))
        
        elif rule['type'] == 'and':
            return apply_and_condition(
                df, 
                rule['feature1'], thresholds.get(rule['feature1']),
                rule['feature2'], thresholds.get(rule['feature2'])
            )
        
        elif rule['type'] == 'or':
            return apply_or_condition(
                df, 
                rule['feature1'], thresholds.get(rule['feature1']),
                rule['feature2'], thresholds.get(rule['feature2'])
            )
        
        else:
            print(f"⚠️ Unknown rule type: {rule['type']}")
            return pd.Series(False, index=df.index)
    
    except Exception as e:
        print(f"❌ Error applying rule {rule['name']}: {e}")
        return pd.Series(False, index=df.index)

# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================

def find_optimal_thresholds(df_train, df_test, rule):
    """Find optimal thresholds for a rule using grid search"""
    
    def get_feature_thresholds(feature):
        """Get candidate thresholds for a feature"""
        if feature not in df_train.columns:
            return [0]
        
        feature_values = df_train[feature].dropna()
        if len(feature_values) == 0:
            return [0]
        
        # Use percentile-based thresholds
        thresholds = []
        for p in SimplifiedRuleConfig.THRESHOLD_PERCENTILES:
            threshold = np.percentile(feature_values, p)
            thresholds.append(threshold)
        
        return sorted(list(set(thresholds)))  # Remove duplicates and sort
    
    # Get features for this rule
    if rule['type'] == 'single':
        features = [rule['feature']]
    else:
        features = [rule['feature1'], rule['feature2']]
    
    # Get candidate thresholds for each feature
    feature_thresholds = {}
    for feature in features:
        feature_thresholds[feature] = get_feature_thresholds(feature)
    
    # Grid search for best combination
    best_f1 = -1
    best_thresholds = {}
    y_true = df_test['event_dummy_1'].values
    
    # Handle single vs double feature rules
    if rule['type'] == 'single':
        feature = features[0]
        for threshold in feature_thresholds[feature]:
            test_thresholds = {feature: threshold}
            y_pred = apply_rule(df_test, rule, test_thresholds).astype(int).values
            
            if np.sum(y_pred) == 0:  # No positive predictions
                continue
            
            try:
                f1 = f1_score(y_true, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresholds = test_thresholds.copy()
            except:
                continue
    
    else:  # Double feature rules
        feature1, feature2 = features[0], features[1]
        
        # Limit combinations to prevent explosion
        max_combinations = 50
        f1_thresholds = feature_thresholds[feature1][:7]  # Top 7 thresholds
        f2_thresholds = feature_thresholds[feature2][:7]  # Top 7 thresholds
        
        combination_count = 0
        for t1 in f1_thresholds:
            for t2 in f2_thresholds:
                if combination_count >= max_combinations:
                    break
                
                test_thresholds = {feature1: t1, feature2: t2}
                y_pred = apply_rule(df_test, rule, test_thresholds).astype(int).values
                
                if np.sum(y_pred) == 0:  # No positive predictions
                    continue
                
                try:
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresholds = test_thresholds.copy()
                except:
                    continue
                
                combination_count += 1
            
            if combination_count >= max_combinations:
                break
    
    return best_thresholds, best_f1

# ============================================================================
# CROSS-VALIDATION EVALUATION
# ============================================================================

def evaluate_rule_cv(rule, df, cv_splits):
    """Evaluate a single rule using cross-validation"""
    fold_results = []
    fold_thresholds = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        try:
            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx]
            
            # Find optimal thresholds
            best_thresholds, best_f1 = find_optimal_thresholds(df_train, df_test, rule)
            
            if not best_thresholds or best_f1 <= 0:
                continue
            
            # Apply rule with best thresholds
            y_pred = apply_rule(df_test, rule, best_thresholds).astype(int)
            y_true = df_test['event_dummy_1']
            
            # Calculate metrics
            if np.sum(y_pred) == 0:
                continue
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Store results
            result = {
                'fold': fold_idx,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'thresholds': best_thresholds.copy()
            }
            
            fold_results.append(result)
            fold_thresholds.append(best_thresholds)
            
        except Exception as e:
            print(f"⚠️ Fold {fold_idx} failed for rule {rule['name']}: {e}")
            continue
    
    return fold_results, fold_thresholds

def _evaluate_rule_wrapper(args):
    """Wrapper function for parallel rule evaluation"""
    rule_idx, rule, df, cv_splits = args
    try:
        fold_results, fold_thresholds = evaluate_rule_cv(rule, df, cv_splits)
        return rule_idx, rule, fold_results, fold_thresholds, None
    except Exception as e:
        return rule_idx, rule, [], [], str(e)

def run_parallel_rule_evaluation(rules, df, cv_splits):
    """Run rule evaluation in parallel"""
    print(f"🚀 Evaluating {len(rules)} rules using {SimplifiedRuleConfig.N_JOBS} cores...")
    
    # Prepare arguments for parallel processing
    args_list = [(i, rule, df, cv_splits) for i, rule in enumerate(rules)]
    
    all_results = []
    all_fold_thresholds = {}
    failed_rules = []
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=SimplifiedRuleConfig.N_JOBS) as executor:
        # Submit all jobs
        future_to_args = {
            executor.submit(_evaluate_rule_wrapper, args): args for args in args_list
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_args):
            rule_idx, rule, fold_results, fold_thresholds, error = future.result()
            completed += 1
            
            if error:
                failed_rules.append((rule_idx, rule, error))
                print(f"❌ Rule {rule_idx + 1}/{len(rules)} failed: {rule['name']}")
                continue
            
            if len(fold_results) < SimplifiedRuleConfig.MIN_FOLDS_SUCCESS:
                print(f"⚠️ Rule {rule_idx + 1}/{len(rules)} insufficient folds: {rule['name']}")
                continue
            
            # Calculate aggregated metrics
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            aggregated = {f'{metric}_mean': np.mean([r[metric] for r in fold_results]) for metric in metrics}
            aggregated.update({f'{metric}_std': np.std([r[metric] for r in fold_results]) for metric in metrics})
            
            # Add rule information
            result = {
                'rule_idx': rule_idx,
                'rule_name': rule['name'],
                'rule_type': rule['type'],
                'complexity': rule['complexity'],
                'n_folds_successful': len(fold_results),
                **aggregated
            }
            
            # Add feature information
            if rule['type'] == 'single':
                result['feature'] = rule['feature']
                result['feature1'] = rule['feature']
                result['feature2'] = None
            else:
                result['feature1'] = rule['feature1']
                result['feature2'] = rule['feature2']
                result['feature'] = f"{rule['feature1']} + {rule['feature2']}"
            
            # Add threshold information (one column per fold)
            if fold_thresholds:
                # Create threshold columns for each fold
                for fold_idx, fold_thresh in enumerate(fold_thresholds):
                    if fold_thresh:  # If fold has thresholds
                        # Create compact threshold string for this fold
                        fold_threshold_str = []
                        for feature, threshold in fold_thresh.items():
                            # Short feature name (remove _swan suffix for readability)
                            short_feature = feature.replace('_swan', '')
                            fold_threshold_str.append(f"{short_feature}:{round(threshold, 4)}")
                        
                        result[f'thresholds_fold_{fold_idx}'] = " | ".join(fold_threshold_str)
                    else:
                        result[f'thresholds_fold_{fold_idx}'] = ""
                
                # Also add mean thresholds across all folds for convenience
                all_thresholds = {}
                for fold_thresh in fold_thresholds:
                    for feature, threshold in fold_thresh.items():
                        if feature not in all_thresholds:
                            all_thresholds[feature] = []
                        all_thresholds[feature].append(threshold)
                
                # Create mean threshold string
                mean_threshold_str = []
                for feature, thresholds in all_thresholds.items():
                    mean_threshold = round(np.mean(thresholds), 4)
                    short_feature = feature.replace('_swan', '')
                    mean_threshold_str.append(f"{short_feature}:{mean_threshold}")
                
                result['thresholds_mean'] = " | ".join(mean_threshold_str)
            
            all_results.append(result)
            all_fold_thresholds[rule_idx] = fold_thresholds
            
            # Progress update
            if completed % 50 == 0:
                print(f"📊 Progress: {completed}/{len(rules)} rules evaluated")
    
    print(f"✅ Evaluation complete: {len(all_results)} successful rules")
    
    if failed_rules:
        print(f"⚠️ {len(failed_rules)} rules failed evaluation")
    
    return all_results, all_fold_thresholds

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def create_cv_splits(df):
    """Create cross-validation splits"""
    print(f"📅 Creating {SimplifiedRuleConfig.N_FOLDS}-fold cross-validation splits...")
    
    if SimplifiedRuleConfig.USE_TIME_SERIES_CV:
        tscv = TimeSeriesSplit(n_splits=SimplifiedRuleConfig.N_FOLDS)
        cv_splits = list(tscv.split(df))
        print(f"✅ Using TimeSeriesSplit")
    else:
        skf = StratifiedKFold(
            n_splits=SimplifiedRuleConfig.N_FOLDS, 
            shuffle=True, 
            random_state=SimplifiedRuleConfig.RANDOM_STATE
        )
        cv_splits = list(skf.split(df, df['event_dummy_1']))
        print(f"✅ Using StratifiedKFold")
    
    return cv_splits

def save_results(results_df, fold_thresholds, output_dir):
    """Save all results to files"""
    print("💾 Saving results...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save main results
    results_path = os.path.join(output_dir, 'simplified_rule_cv_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"✅ Main results saved: {results_path}")
    
    # Save fold thresholds
    if fold_thresholds:
        # Convert to DataFrame format
        threshold_rows = []
        for rule_idx, thresholds_list in fold_thresholds.items():
            rule_info = results_df[results_df['rule_idx'] == rule_idx].iloc[0]
            
            for fold_idx, threshold_dict in enumerate(thresholds_list):
                row = {
                    'rule_idx': rule_idx,
                    'rule_name': rule_info['rule_name'],
                    'rule_type': rule_info['rule_type'],
                    'fold': fold_idx
                }
                row.update(threshold_dict)
                threshold_rows.append(row)
        
        if threshold_rows:
            threshold_df = pd.DataFrame(threshold_rows)
            threshold_path = os.path.join(output_dir, 'simplified_fold_thresholds.csv')
            threshold_df.to_csv(threshold_path, index=False)
            print(f"✅ Fold thresholds saved: {threshold_path}")
    
    return results_path

def run_simplified_rule_evaluation(prediction_data):
    """Main pipeline for simplified rule evaluation"""
    print("🚀 SIMPLIFIED RULE EVALUATION PIPELINE")
    print("="*80)
    print(f"Run: {config.RUN_PATH}")
    print(f"Reference port: {config.reference_port}")
    print("="*80)
    
    # Step 1: Prepare data
    df = prepare_data(prediction_data)
    
    # Step 2: Get target features (no feature selection)
    target_features = get_target_features(df)
    
    # Step 3: Generate simple rules
    rules = generate_simple_rules(target_features)
    
    # Step 4: Create CV splits
    cv_splits = create_cv_splits(df)
    
    # Step 5: Evaluate rules
    start_time = datetime.now()
    results, fold_thresholds = run_parallel_rule_evaluation(rules, df, cv_splits)
    end_time = datetime.now()
    
    print(f"⏱️ Evaluation completed in {end_time - start_time}")
    
    # Step 6: Create results DataFrame
    if not results:
        print("❌ No successful rule evaluations!")
        return pd.DataFrame(), {}
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f1_mean', ascending=False).reset_index(drop=True)
    
    # Step 7: Save results
    results_path = save_results(results_df, fold_thresholds, config.results_output_dir)
    
    # Step 8: Print summary
    print(f"\n🎉 SIMPLIFIED RULE EVALUATION COMPLETE!")
    print("="*80)
    print(f"📊 Total rules evaluated: {len(results_df)}")
    print(f"🏆 Best F1 score: {results_df['f1_mean'].max():.3f}")
    print(f"📈 Rules with F1 > 0.5: {len(results_df[results_df['f1_mean'] > 0.5])}")
    print(f"📁 Results saved to: {results_path}")
    
    # Show top 10 rules
    print(f"\n🏆 TOP 10 RULES:")
    top_rules = results_df.head(10)[['rule_name', 'rule_type', 'f1_mean', 'precision_mean', 'recall_mean']]
    print(top_rules.to_string(index=False))
    
    return results_df, fold_thresholds

# ============================================================================
# DEBUG AND TESTING
# ============================================================================

def debug_feature_selection(df):
    """Debug function to test feature selection patterns"""
    print("🔍 DEBUGGING FEATURE SELECTION")
    print("="*60)
    
    # Test the updated get_target_features function
    target_features = get_target_features(df)
    
    print(f"\n📊 FEATURE SELECTION SUMMARY:")
    print(f"Total columns in dataset: {len(df.columns)}")
    print(f"SWAN-related features found: {len(target_features)}")
    
    # Show some examples of what was captured
    print(f"\n📋 SAMPLE CAPTURED FEATURES:")
    for i, feature in enumerate(target_features[:20], 1):
        print(f"  {i:2d}. {feature}")
    
    if len(target_features) > 20:
        print(f"  ... and {len(target_features) - 20} more features")
    
    # Check for specific important features (SWAN + MAX)
    important_features = [
        'swh_max_swan',
        'anom_swh_max_swan',
        'swh_max_trend_3_swan',
        'anom_swh_max_trend_3_swan',
        'swh_max_ma_7_swan',
        'anom_swh_max_ma_7_swan',
        'swh_max_deseasonalized_swan',
        'anom_swh_max_deseasonalized_swan',
        'swh_max_detrended_swan',
        'anom_swh_max_detrended_swan'
    ]
    
    print(f"\n🎯 CHECKING IMPORTANT FEATURES:")
    for feature in important_features:
        if feature in target_features:
            print(f"  ✅ {feature}")
        else:
            print(f"  ❌ {feature} - NOT FOUND")
    
    return target_features

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n🔧 SIMPLIFIED_RULE_EVALUATION.PY")
    print(f"Run: {config.RUN_PATH}")
    print(f"Reference port: {config.reference_port}")

    # Load merged features file
    merged_path = os.path.join(config.run_output_dir, 'df_swan_waverys_merged.csv')
    print(f"🔎 Looking for merged features at: {merged_path}")
    
    if not os.path.exists(merged_path):
        print(f"❌ Input file not found: {merged_path}")
        return
    
    df = pd.read_csv(merged_path)
    print(f"✅ Loaded merged features: {merged_path} ({df.shape})")

    # Debug feature selection first
    print("\n" + "="*80)
    debug_feature_selection(df)
    print("="*80)

    # Run simplified rule evaluation
    results_df, fold_thresholds = run_simplified_rule_evaluation(df)
    
    print("\n🎉 Simplified rule evaluation completed!")

if __name__ == "__main__":
    main()