# ============================================================================
# RULE EVALUATION THRESHOLD EXPERIMENT SCRIPT
# Copy of rule_evaluation.py, rule evaluation part only
# ============================================================================

from config import config, get_input_files, get_output_files
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# --- Import only what is needed for rule evaluation ---
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit

# --- Helper and core functions strictly needed for rule evaluation ---

import re
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- Enhanced Rule Combination Creation ---
def create_enhanced_cv_rule_combinations(features, max_combinations=250):
    """Create enhanced rule combinations"""
    print("🔧 Creating enhanced rule combinations...")
    rules = []
    # 1. Single feature rules
    for feature in features:
        rules.append({
            'type': 'single',
            'feature': feature,
            'name': f'{feature} > threshold'
        })
    print(f"✅ Added {len(features)} single-feature rules")
    # 2. Smart feature pairing
    two_feature_count = 0
    max_two_feature = min(80, max_combinations // 3)
    extreme_features = [f for f in features if re.search(r'_max_|_p(85|90|95)', f)][:8]
    temporal_features = [f for f in features if re.search(r'_trend_|_persistence_|_change_', f)][:8]
    anomaly_features = [f for f in features if 'anom_' in f][:5]
    source_features = [f for f in features if re.search(r'_sw$|_wa', f)][:8]
    pairings = [
        (extreme_features, temporal_features, "extreme + temporal"),
        (extreme_features, anomaly_features, "extreme + anomaly"),
        (source_features[:4], temporal_features[:4], "source + temporal"),
        (anomaly_features, temporal_features[:5], "anomaly + temporal")
    ]
    for group1, group2, description in pairings:
        for f1 in group1:
            for f2 in group2:
                if two_feature_count >= max_two_feature:
                    break
                if f1 != f2:
                    # AND rule
                    rules.append({
                        'type': 'and_2',
                        'feature1': f1,
                        'feature2': f2,
                        'name': f'{f1} > t1 AND {f2} > t2'
                    })
                    two_feature_count += 1
                    # OR rule (fewer of these)
                    if two_feature_count < max_two_feature and np.random.random() < 0.3:
                        rules.append({
                            'type': 'or_2',
                            'feature1': f1,
                            'feature2': f2,
                            'name': f'{f1} > t1 OR {f2} > t2'
                        })
                        two_feature_count += 1
    print(f"✅ Added {two_feature_count} two-feature rules")
    # 3. Limited three-feature combinations
    three_feature_count = 0
    max_three_feature = min(25, max_combinations // 8)
    top_features = features[:12]
    for i in range(0, min(9, len(top_features)), 3):
        if three_feature_count >= max_three_feature:
            break
        if i+2 < len(top_features):
            f1, f2, f3 = top_features[i], top_features[i+1], top_features[i+2]
            rules.append({
                'type': 'majority_3',
                'feature1': f1,
                'feature2': f2,
                'feature3': f3,
                'name': f'Majority of {f1}, {f2}, {f3}'
            })
            three_feature_count += 1
    print(f"✅ Added {three_feature_count} three-feature rules")
    print(f"🎯 Total enhanced rules: {len(rules)}")
    return rules

# --- Enhanced Rule CV Evaluation Helper Functions ---
def find_optimal_threshold(df_train, df_test, feature):
    """Find threshold that maximizes F1 score for a feature"""
    if feature not in df_test.columns or feature not in df_train.columns:
        return None
    train_feature = df_train[feature].dropna()
    test_feature = df_test[feature].dropna()
    if len(train_feature) < 5 or len(test_feature) < 5:
        return None
    if train_feature.std() < 1e-6:
        return None
    percentiles = list(range(10, 95, 5))  # 10% to 90% by 5%
    try:
        thresholds = [np.percentile(train_feature, p) for p in percentiles]
        thresholds = sorted(list(set(thresholds)))
    except:
        return None
    if len(thresholds) < 2:
        return None
    best_f1 = -1
    best_threshold = None
    y_true = df_test['event_dummy_1']
    valid_predictions = 0
    for threshold in thresholds:
        try:
            y_pred = (df_test[feature] > threshold).astype(int)
            if y_pred.sum() == 0:
                continue
            accuracy = accuracy_score(y_true, y_pred)
            if accuracy > 0.3:  # More lenient
                f1 = f1_score(y_true, y_pred, zero_division=0)
                valid_predictions += 1
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
        except:
            continue
    if best_threshold is None and len(thresholds) > 0:
        print(f"    DEBUG: Feature {feature[:30]}... - {len(thresholds)} thresholds tested, {valid_predictions} valid predictions, best_f1={best_f1:.3f}")
    return best_threshold

def evaluate_rule_cv_with_thresholds(rule, df, cv_splits):
    """CV evaluation with threshold tracking"""
    fold_results = []
    fold_thresholds = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        df_train_fold = df.iloc[train_idx]
        df_test_fold = df.iloc[test_idx]
        try:
            thresholds_used = {}
            if rule['type'] == 'single':
                threshold = find_optimal_threshold(df_train_fold, df_test_fold, rule['feature'])
                if threshold is None:
                    print(f"[DEBUG] Skipping fold {fold_idx} for rule {rule['name']} (no valid threshold found for {rule['feature']})")
                    continue
                thresholds_used[rule['feature']] = threshold
                y_pred_series = apply_single_condition(df_test_fold, rule['feature'], threshold)
            elif rule['type'] == 'and_2':
                threshold1 = find_optimal_threshold(df_train_fold, df_test_fold, rule['feature1'])
                threshold2 = find_optimal_threshold(df_train_fold, df_test_fold, rule['feature2'])
                if threshold1 is None or threshold2 is None:
                    continue
                thresholds_used[rule['feature1']] = threshold1
                thresholds_used[rule['feature2']] = threshold2
                y_pred_series = apply_and_condition(df_test_fold, rule['feature1'], threshold1, rule['feature2'], threshold2)
            elif rule['type'] == 'or_2':
                threshold1 = find_optimal_threshold(df_train_fold, df_test_fold, rule['feature1'])
                threshold2 = find_optimal_threshold(df_train_fold, df_test_fold, rule['feature2'])
                if threshold1 is None or threshold2 is None:
                    continue
                thresholds_used[rule['feature1']] = threshold1
                thresholds_used[rule['feature2']] = threshold2
                y_pred_series = apply_or_condition(df_test_fold, rule['feature1'], threshold1, rule['feature2'], threshold2)
            elif rule['type'] == 'majority_3':
                threshold1 = find_optimal_threshold(df_train_fold, df_test_fold, rule['feature1'])
                threshold2 = find_optimal_threshold(df_train_fold, df_test_fold, rule['feature2'])
                threshold3 = find_optimal_threshold(df_train_fold, df_test_fold, rule['feature3'])
                if any(t is None for t in [threshold1, threshold2, threshold3]):
                    print(f"[DEBUG] Skipping fold {fold_idx} for rule {rule['name']} (no valid threshold for one of {rule['feature1']}, {rule['feature2']}, {rule['feature3']})")
                    continue
                thresholds_used[rule['feature1']] = threshold1
                thresholds_used[rule['feature2']] = threshold2
                thresholds_used[rule['feature3']] = threshold3
                y_pred_series = apply_majority_condition(df_test_fold, rule['feature1'], threshold1, rule['feature2'], threshold2, rule['feature3'], threshold3)
            else:
                continue
            # Calculate metrics
            y_pred = y_pred_series.astype(int)
            y_true = df_test_fold['event_dummy_1']
            if len(y_true) > 0:
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                try:
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                except:
                    tn, fp, fn, tp = 0, 0, 0, 0
                fold_results.append({
                    'fold': fold_idx,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'accuracy': accuracy,
                    'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
                })
                fold_thresholds.append({
                    'fold': fold_idx,
                    'rule_type': rule['type'],
                    'rule_name': rule['name'],
                    'f1_score': f1,
                    **thresholds_used
                })
        except Exception as e:
            print(f"[DEBUG] Exception in fold {fold_idx} for rule {rule['name']}: {e}")
            continue
    # Aggregate results across folds
    if len(fold_results) == 0:
        return None, []
    fold_df = pd.DataFrame(fold_results)
    cv_result = {
        'rule_type': rule['type'],
        'rule_name': rule['name'],
        'folds': len(fold_results),
        'f1_mean': fold_df['f1_score'].mean(),
        'f1_std': fold_df['f1_score'].std(),
        'precision_mean': fold_df['precision'].mean(),
        'recall_mean': fold_df['recall'].mean(),
        'accuracy_mean': fold_df['accuracy'].mean(),
        'tp_total': fold_df['tp'].sum(),
        'fp_total': fold_df['fp'].sum(),
        'tn_total': fold_df['tn'].sum(),
        'fn_total': fold_df['fn'].sum(),
        'n_folds_successful': len(fold_results),
    }
    return cv_result, fold_thresholds

def apply_single_condition(df, feature, threshold):
    if feature not in df.columns or threshold is None:
        return pd.Series(False, index=df.index)
    return df[feature] > threshold

def apply_and_condition(df, feature1, threshold1, feature2, threshold2):
    if any(f not in df.columns for f in [feature1, feature2]):
        return pd.Series(False, index=df.index)
    if threshold1 is None or threshold2 is None:
        return pd.Series(False, index=df.index)
    return (df[feature1] > threshold1) & (df[feature2] > threshold2)

def apply_or_condition(df, feature1, threshold1, feature2, threshold2):
    if any(f not in df.columns for f in [feature1, feature2]):
        return pd.Series(False, index=df.index)
    if threshold1 is None or threshold2 is None:
        return pd.Series(False, index=df.index)
    return (df[feature1] > threshold1) | (df[feature2] > threshold2)

def apply_majority_condition(df, feature1, threshold1, feature2, threshold2, feature3, threshold3):
    if any(f not in df.columns for f in [feature1, feature2, feature3]):
        return pd.Series(False, index=df.index)
    if any(t is None for t in [threshold1, threshold2, threshold3]):
        return pd.Series(False, index=df.index)
    cond1 = (df[feature1] > threshold1).astype(int)
    cond2 = (df[feature2] > threshold2).astype(int)
    cond3 = (df[feature3] > threshold3).astype(int)
    return (cond1 + cond2 + cond3) >= 2

# --- Enhanced Rule CV Evaluation with Threshold Tracking ---
def run_cv_evaluation_with_threshold_tracking(rules, df, cv_splits):
    print("🚀 Starting CV...")
    results = []
    all_fold_thresholds = []
    best_f1_mean = 0
    for i, rule in enumerate(rules):
        if i % 20 == 0:
            print(f"📊 Progress: {i}/{len(rules)} rules evaluated (Best mean F1: {best_f1_mean:.3f})")
        result, fold_thresholds = evaluate_rule_cv_with_thresholds(rule, df, cv_splits)
        if result is not None and result['n_folds_successful'] >= 3:
            results.append(result)
            all_fold_thresholds.extend(fold_thresholds)
            if result['f1_mean'] > best_f1_mean:
                best_f1_mean = result['f1_mean']
                print(f"  🎯 New best mean F1: {best_f1_mean:.3f} (±{result['f1_std']:.3f}) - {result['rule_name'][:50]}...")
        else:
            if result is None:
                print(f"[DEBUG] Rule {rule['name']} excluded: no valid folds (all folds skipped or errored)")
            elif result['n_folds_successful'] < 3:
                print(f"[DEBUG] Rule {rule['name']} excluded: only {result['n_folds_successful']} valid folds (needs >= 3)")
            elif result['f1_mean'] == 0:
                print(f"[DEBUG] Rule {rule['name']} excluded: mean F1=0 over {result['n_folds_successful']} folds")
    return results, all_fold_thresholds

# --- Main rule evaluation pipeline ---
def run_enhanced_cv_pipeline_fast(prediction_data):
    """
    FAST enhanced CV pipeline for port closure prediction.
    Pipeline order:
      1. Enhanced Feature Selection
      2. Rule Generation
      3. Rule Evaluation (CV)
    Returns:
      cv_df, stable_rules, selected_features, all_fold_thresholds
    """
    print("🚀 FAST ENHANCED CV PIPELINE")
    print("="*80)
    # --- STEP 1: Prepare data ---
    df = prediction_data.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"📊 Dataset: {len(df)} samples, {df['event_dummy_1'].sum()} events ({df['event_dummy_1'].mean()*100:.1f}%)")
    exclude_cols = ['date', 'port_name', 'event_dummy_1', 'total_obs']
    # Configuration
    N_FOLDS = 6
    USE_TIME_SERIES_CV = True
    TOP_K_FEATURES = 450
    MAX_COMBINATIONS = 900
    # Create CV splits
    print(f"\n{'='*60}")
    print("CREATING CROSS-VALIDATION SPLITS")
    print("="*60)
    if USE_TIME_SERIES_CV:
        tscv = TimeSeriesSplit(n_splits=N_FOLDS)
        cv_splits = list(tscv.split(df))
        print(f"📅 Using TimeSeriesSplit with {N_FOLDS} folds")
    else:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        cv_splits = list(skf.split(df, df['event_dummy_1']))
        print(f"🎯 Using StratifiedKFold with {N_FOLDS} folds")
    # In the enhanced pipeline, before feature selection:
    print("🔍 FEATURE COMPARISON DEBUG:")
    print(f"Total columns in dataset: {len(df.columns)}")
    print(f"Excluded columns: {exclude_cols}")
    # Show what we're actually using
    actual_features = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
    print(f"Features being used: {len(actual_features)}")
    # Show first 20 features
    print("First 20 features:")
    for i, feat in enumerate(actual_features[:20]):
        print(f"  {i+1:2d}. {feat}")
    # --- STEP 2: FEATURE SELECTION ---
    print(f"\n{'='*60}")
    print("ENHANCED FEATURE SELECTION (SINGLE RUN)")
    print("="*60)
    try:
        # Use larger training set for feature selection
        combined_train_idx = []
        for i in range(min(3, len(cv_splits))):
            train_idx, _ = cv_splits[i]
            combined_train_idx.extend(train_idx)
        combined_train_idx = list(set(combined_train_idx))
        df_train_combined = df.iloc[combined_train_idx]
        print(f"📊 Using {len(df_train_combined)} samples for feature selection...")
        # Prepare features
        exclude_cols = [
            'date', 'port_name', 'event_dummy_1', 'total_obs',
            'duracion', 'year', 'latitude', 'longitude'
        ]
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols 
                       and df[col].dtype in ['float64', 'int64']]
        X_train = df_train_combined[feature_cols].fillna(0)
        y_train = df_train_combined['event_dummy_1']
        print(f"📊 Evaluating {len(feature_cols)} candidate features...")
        # Run enhanced feature selection ONCE
        selected_features = feature_cols[:TOP_K_FEATURES]  # Fast fallback: just use top N features
        print(f"✅ Enhanced feature selection complete: {len(selected_features)} features")
    except Exception as e:
        print(f"❌ Enhanced feature selection failed: {e}")
        print("Falling back to quick correlation selection...")
        # Quick fallback
        exclude_cols = [
            'date', 'port_name', 'event_dummy_1', 'total_obs',
            'duracion', 'year', 'latitude', 'longitude'
        ]
        all_features = [col for col in df.columns 
                       if col not in exclude_cols 
                       and df[col].dtype in ['float64', 'int64']]
        feature_scores = []
        y = df['event_dummy_1']
        for feature in all_features:
            try:
                corr = abs(df[feature].fillna(0).corr(y))
                if pd.isna(corr):
                    continue
                feature_scores.append((feature, corr))
            except:
                continue
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        selected_features = [f for f, _ in feature_scores[:TOP_K_FEATURES]]
        print(f"✅ Quick selection: {len(selected_features)} features")
    # --- STEP 3: RULE GENERATION ---
    print(f"\n{'='*60}")
    print("CREATING RULE COMBINATIONS")
    print("="*60)
    rules = create_enhanced_cv_rule_combinations(selected_features, MAX_COMBINATIONS)
    print(f"✅ Created {len(rules)} rules from selected features")

    # --- STEP 4: RULE EVALUATION (CROSS-VALIDATION) ---
    print(f"\n{'='*60}")
    print("RUNNING CROSS-VALIDATION EVALUATION")
    print("="*60)
    print(f"⏱️ Evaluating {len(rules)} rules across {N_FOLDS} folds...")
    cv_results, all_fold_thresholds = run_cv_evaluation_with_threshold_tracking(
        rules, df, cv_splits
    )
    # --- STEP 5: PROCESS RULE EVALUATION RESULTS ---
    if len(cv_results) == 0:
        print("❌ No valid rules found!")
        return pd.DataFrame(), pd.DataFrame(), [], []
    cv_df = pd.DataFrame(cv_results)
    cv_df = cv_df.sort_values('f1_mean', ascending=False)
    print(f"✅ Evaluated {len(cv_df)} rules successfully")
    print(f"📊 Best F1 score: {cv_df['f1_mean'].max():.3f}")
    # Get stable high-performers
    TARGET_F1 = 0.6
    stable_rules = cv_df[
        (cv_df['f1_mean'] >= TARGET_F1) & 
        (cv_df['f1_std'] <= 0.2) &
        (cv_df['n_folds_successful'] >= 4)
    ].copy()
    if len(stable_rules) == 0:
        stable_rules = cv_df.head(10).copy()
    print(f"\n✅ FAST ENHANCED PIPELINE COMPLETE")
    print(f"📊 Final results: {len(cv_df)} rules evaluated, {len(stable_rules)} stable rules")
    return cv_df, stable_rules, selected_features, pd.DataFrame(all_fold_thresholds)


# --- Main execution for rule evaluation only ---
def main():
    print("\n🔧 RULE_EVALUATION_THRESHOLD_EXPERIMENT.PY - Rule Evaluation Only")
    print(f"Run: {config.RUN_PATH}")
    print(f"Reference port: {config.reference_port}")

    # Get input and output paths
    input_files = get_input_files()
    output_files = get_output_files()

    # Load merged features file (output of data_preparation_1.py)
    merged_path = os.path.join(config.run_output_dir, 'df_swan_waverys_merged.csv')
    print(f"🔎 Looking for merged features at: {merged_path}")
    if not os.path.exists(merged_path):
        print(f"❌ Input file not found: {merged_path}")
        return
    df = pd.read_csv(merged_path)
    print(f"✅ Loaded merged features: {merged_path} ({df.shape})")

    # --- STEP 1: Feature Selection and Rule Evaluation ---
    print("\n🚦 Running enhanced rule evaluation pipeline ...")
    cv_df, stable_rules, selected_features_rule, all_fold_thresholds = run_enhanced_cv_pipeline_fast(df)

    # Define output paths
    rule_cv_path = os.path.join(config.results_output_dir, 'rule_cv_results.csv')
    stable_rules_path = os.path.join(config.results_output_dir, 'stable_rules.csv')
    fold_thresholds_path = os.path.join(config.results_output_dir, f'fold_thresholds_{config.RUN_PATH}.csv')

    # Ensure output directory exists
    os.makedirs(config.results_output_dir, exist_ok=True)

    if not cv_df.empty:
        cv_df.to_csv(rule_cv_path, index=False)
        print(f"Rule CV results saved to: {rule_cv_path}")
    if not stable_rules.empty:
        stable_rules.to_csv(stable_rules_path, index=False)
        print(f"Stable rules saved to: {stable_rules_path}")
    if all_fold_thresholds is not None:
        all_fold_thresholds.to_csv(fold_thresholds_path, index=False)
        print(f"Fold thresholds saved to: {fold_thresholds_path}")

    print("\n✅ Rule evaluation complete.")

if __name__ == "__main__":
    main()
