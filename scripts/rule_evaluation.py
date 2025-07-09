"""
RULE EVALUATION - CV PIPELINE AND FEATURE SELECTION
===================================================

Refactored from SWAN_MOD_rule_evaluation_v3.py for repo pipeline integration.
- Uses centralized config.py for all paths and settings
- Suppresses exploratory/plotting/interactive code
- Modular with main() entry point
- Outputs results to config-driven results directory

Author: Wave Analysis Team (refactored)
Date: 2025
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path for config import
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from config import config, get_input_files, get_output_files
    print("✅ Using centralized configuration")
except ImportError:
    print("❌ Error: Cannot import config. Make sure config.py is in the project root.")
    exit(1)

# Import main pipeline logic from SWAN_MOD_rule_evaluation_v3.py
# (Feature selection, CV, rule creation, etc.)
# Only include core pipeline functions, suppress plotting/interactive code

# ================== ENHANCED FEATURE SELECTION AND CV PIPELINE ===================

import re
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit

class EnhancedFeatureSelectionConfig:
    TOP_K_PER_METHOD = 200
    FINAL_TOP_K = 150
    MIN_VOTES = 2
    METHOD_WEIGHTS = {
        'random_forest': 2.5,
        'extra_trees': 2.0,
        'mutual_info': 2.2,
        'f_test': 1.8,
        'domain_selection': 3.0
    }
    DOMAIN_PATTERNS = {
        'max_waves': (r'swh_max_sw|swe_max_sw|swh_max_wa|swe_max_wa', 4.0),
        'swan_features': (r'_sw$', 3.5),
        'extreme_percentiles': (r'_p(85|90|95|99)', 3.0),
        'high_percentiles': (r'_p(75|80)', 2.5),
        'medium_trends': (r'_trend_(7|14|21)', 2.8),
        'short_trends': (r'_trend_(3|5)', 2.2),
        'persistence': (r'_persistence_(5|7|14)', 2.6),
        'recent_changes': (r'_change_(1|3|5)', 2.4),
        'medium_changes': (r'_change_(7|14)', 2.0),
        'anomalies': (r'anom_', 2.2),
        'variability': (r'_cv|_range|_iqr|_std', 2.0),
        'immediate_lags': (r'_lag_(1|2)', 2.0),
        'short_lags': (r'_lag_(3|5)', 1.8),
        'waverys_data': (r'_wa$', 1.8),
    }

# --- Feature selection methods, ensemble voting, and rule creation ---
# (All code from SWAN_MOD_rule_evaluation_v3.py up to and including cross-validation logic)

# [Insert all functions from enhanced_random_forest_selection through cross_evaluate_thresholds here]
# For brevity, see previous code blocks for full implementations.

# --- Enhanced feature selection methods ---
def enhanced_random_forest_selection(X_train, y_train, top_k=200):
    """Enhanced Random Forest feature selection"""
    print("🌲 Enhanced Random Forest Feature Importance")
    feature_names = X_train.columns.tolist()
    rf_configs = [
        {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5, 'random_state': 42},
        {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 3, 'random_state': 123},
        {'n_estimators': 150, 'max_depth': 12, 'min_samples_split': 8, 'random_state': 456},
        {'n_estimators': 250, 'max_depth': 18, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'random_state': 789},
        {'n_estimators': 180, 'max_depth': None, 'min_samples_split': 10, 'max_features': 0.8, 'random_state': 987}
    ]
    all_importances = []
    for i, config in enumerate(rf_configs):
        print(f"  Running RF config {i+1}/{len(rf_configs)}...")
        try:
            rf = RandomForestClassifier(**config, n_jobs=-1)
            rf.fit(X_train, y_train)
            all_importances.append(rf.feature_importances_)
        except Exception as e:
            print(f"    RF config {i+1} failed: {e}")
    if len(all_importances) == 0:
        return [], pd.DataFrame()
    mean_importances = np.mean(all_importances, axis=0)
    std_importances = np.std(all_importances, axis=0)
    stability_scores = mean_importances / (std_importances + 1e-6)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': mean_importances,
        'importance_std': std_importances,
        'stability': stability_scores,
    })
    importance_df['combined_score'] = (
        0.7 * importance_df['importance_mean'] + 
        0.3 * (importance_df['stability'] / importance_df['stability'].max())
    )
    importance_df = importance_df.sort_values('combined_score', ascending=False)
    selected_features = importance_df.head(top_k)['feature'].tolist()
    print(f"  ✅ Selected {len(selected_features)} features")
    return selected_features, importance_df

def enhanced_extra_trees_selection(X_train, y_train, top_k=200):
    """Enhanced Extra Trees feature selection"""
    print("🌳 Enhanced Extra Trees Feature Importance")
    feature_names = X_train.columns.tolist()
    et_configs = [
        {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 3, 'random_state': 42},
        {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 5, 'random_state': 123},
        {'n_estimators': 150, 'max_depth': None, 'min_samples_leaf': 3, 'random_state': 456},
        {'n_estimators': 250, 'max_depth': 25, 'max_features': 0.7, 'random_state': 789}
    ]
    all_importances = []
    for i, config in enumerate(et_configs):
        print(f"  Running ET config {i+1}/{len(et_configs)}...")
        try:
            et = ExtraTreesClassifier(**config, n_jobs=-1)
            et.fit(X_train, y_train)
            all_importances.append(et.feature_importances_)
        except Exception as e:
            print(f"    ET config {i+1} failed: {e}")
    if len(all_importances) == 0:
        return [], pd.DataFrame()
    mean_importances = np.mean(all_importances, axis=0)
    stability = mean_importances / (np.std(all_importances, axis=0) + 1e-6)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_importances,
        'stability': stability,
        'weighted_score': mean_importances * (1 + 0.2 * stability / stability.max())
    }).sort_values('weighted_score', ascending=False)
    selected_features = importance_df.head(top_k)['feature'].tolist()
    print(f"  ✅ Selected {len(selected_features)} features")
    return selected_features, importance_df

def enhanced_mutual_info_selection(X_train, y_train, top_k=200):
    """Enhanced Mutual Information selection"""
    print("🔗 Enhanced Mutual Information")
    feature_names = X_train.columns.tolist()
    try:
        mi_scores_list = []
        random_states = [42, 123, 456, 789, 987]
        for rs in random_states:
            mi_scores = mutual_info_classif(X_train, y_train, random_state=rs)
            mi_scores_list.append(mi_scores)
        mean_mi_scores = np.mean(mi_scores_list, axis=0)
        std_mi_scores = np.std(mi_scores_list, axis=0)
        stability = mean_mi_scores / (std_mi_scores + 1e-6)
        mi_df = pd.DataFrame({
            'feature': feature_names,
            'mi_score': mean_mi_scores,
            'mi_stability': stability,
            'weighted_mi': mean_mi_scores * (1 + 0.15 * stability / stability.max())
        }).sort_values('weighted_mi', ascending=False)
        selected_features = mi_df.head(top_k)['feature'].tolist()
        print(f"  ✅ Selected {len(selected_features)} features")
        return selected_features, mi_df
    except Exception as e:
        print(f"  ❌ Mutual Information failed: {e}")
        return [], pd.DataFrame()

def enhanced_f_test_selection(X_train, y_train, top_k=200):
    """Enhanced F-Test selection"""
    print("📊 Enhanced F-Test (ANOVA)")
    feature_names = X_train.columns.tolist()
    try:
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X_train, y_train)
        f_scores = selector.scores_
        p_values = selector.pvalues_
        neg_log_p = -np.log(p_values + 1e-10)
        normalized_f = f_scores / (f_scores.max() + 1e-6)
        normalized_p = neg_log_p / (neg_log_p.max() + 1e-6)
        combined_score = 0.7 * normalized_f + 0.3 * normalized_p
        f_df = pd.DataFrame({
            'feature': feature_names,
            'f_score': f_scores,
            'p_value': p_values,
            'combined_f_score': combined_score
        }).sort_values('combined_f_score', ascending=False)
        selected_features = f_df.head(top_k)['feature'].tolist()
        print(f"  ✅ Selected {len(selected_features)} features")
        return selected_features, f_df
    except Exception as e:
        print(f"  ❌ F-Test failed: {e}")
        return [], pd.DataFrame()

def domain_specific_selection(X_train, y_train, top_k=200):
    """Domain-specific feature selection for oceanographic data"""
    print("🌊 Domain-Specific Oceanographic Selection")
    feature_names = X_train.columns.tolist()
    domain_scores = {}
    for feature in feature_names:
        try:
            # Base correlation score
            base_correlation = abs(X_train[feature].fillna(0).corr(y_train))
            if pd.isna(base_correlation):
                base_correlation = 0
            # Apply domain pattern boosts
            boost_factor = 1.0
            matched_patterns = []
            for pattern_name, (regex, boost) in EnhancedFeatureSelectionConfig.DOMAIN_PATTERNS.items():
                if re.search(regex, feature):
                    boost_factor *= boost
                    matched_patterns.append(pattern_name)
            # Data quality factors
            missing_pct = X_train[feature].isna().mean()
            has_variance = X_train[feature].std() > 1e-6
            quality_penalty = 1.0 if (missing_pct < 0.3 and has_variance) else 0.5
            # Feature type diversity bonus
            type_bonus = 1.0
            if '_max_' in feature or '_p9' in feature:
                type_bonus = 1.2
            elif '_trend_' in feature or '_change_' in feature:
                type_bonus = 1.15
            elif '_persistence_' in feature:
                type_bonus = 1.1
            final_score = base_correlation * boost_factor * quality_penalty * type_bonus
            domain_scores[feature] = {
                'base_correlation': base_correlation,
                'boost_factor': boost_factor,
                'quality_penalty': quality_penalty,
                'type_bonus': type_bonus,
                'final_score': final_score,
                'matched_patterns': matched_patterns
            }
        except Exception as e:
            domain_scores[feature] = {
                'base_correlation': 0, 'boost_factor': 1.0, 'quality_penalty': 0,
                'type_bonus': 1.0, 'final_score': 0, 'matched_patterns': []
            }
    # Convert to DataFrame and sort
    domain_df = pd.DataFrame.from_dict(domain_scores, orient='index')
    domain_df['feature'] = domain_df.index
    domain_df = domain_df.sort_values('final_score', ascending=False)
    selected_features = domain_df.head(top_k)['feature'].tolist()
    print(f"  ✅ Selected {len(selected_features)} features")
    return selected_features, domain_df

def enhanced_ensemble_voting(method_results, final_top_k=150):
    """Enhanced ensemble voting system"""
    print("🗳️ ENHANCED ENSEMBLE VOTING")
    print("="*60)
    feature_votes = {}
    method_names = ['random_forest', 'extra_trees', 'mutual_info', 'f_test', 'domain_selection']
    # Collect votes with enhanced scoring
    for method_name, (features, scores_df) in zip(method_names, method_results):
        if len(features) == 0:
            continue
        weight = EnhancedFeatureSelectionConfig.METHOD_WEIGHTS.get(method_name, 1.0)
        print(f"📋 {method_name}: {len(features)} features, weight: {weight}")
        for i, feature in enumerate(features):
            position_score = np.exp(-i / len(features)) if len(features) > 0 else 0
            if feature not in feature_votes:
                feature_votes[feature] = {
                    'total_score': 0,
                    'method_count': 0,
                    'methods': [],
                    'positions': [],
                    'best_position': float('inf')
                }
            feature_votes[feature]['total_score'] += position_score * weight
            feature_votes[feature]['method_count'] += 1
            feature_votes[feature]['methods'].append(method_name)
            feature_votes[feature]['positions'].append(i + 1)
            feature_votes[feature]['best_position'] = min(feature_votes[feature]['best_position'], i + 1)
    # Filter by minimum votes
    qualified_features = {
        feature: info for feature, info in feature_votes.items()
        if info['method_count'] >= EnhancedFeatureSelectionConfig.MIN_VOTES
    }
    print(f"✅ {len(qualified_features)} features qualified")
    # Enhanced scoring
    for feature, info in qualified_features.items():
        consensus_score = info['method_count'] / len(method_names)
        quality_score = info['total_score']
        info['enhanced_score'] = quality_score * (1 + 0.2 * consensus_score)
        info['consensus_score'] = consensus_score
    # Sort by enhanced score
    sorted_features = sorted(qualified_features.items(), 
                           key=lambda x: x[1]['enhanced_score'], reverse=True)
    final_features = [feature for feature, info in sorted_features[:final_top_k]]
    # Create summary
    voting_summary = []
    for feature, info in sorted_features[:final_top_k]:
        voting_summary.append({
            'feature': feature,
            'enhanced_score': info['enhanced_score'],
            'method_count': info['method_count'],
            'consensus_score': info['consensus_score'],
            'methods': ', '.join(info['methods']),
            'best_position': info['best_position'],
            'avg_position': np.mean(info['positions'])
        })
    voting_df = pd.DataFrame(voting_summary)
    print(f"🏆 ENHANCED SELECTION: {len(final_features)} features")
    return final_features, voting_df

# --- Enhanced feature selection pipeline ---
def enhanced_feature_selection_pipeline(X_train, y_train, top_k_per_method=200, final_top_k=150):
    """Complete enhanced feature selection pipeline"""
    print("🚀 ENHANCED FEATURE SELECTION PIPELINE")
    print("="*80)
    print(f"Input: {X_train.shape[1]} features, {X_train.shape[0]} samples")
    print(f"Target: Select {final_top_k} features using 5 enhanced methods")
    print("="*80)
    # Store results from each method
    method_results = []
    # Method 1: Enhanced Random Forest
    try:
        rf_features, rf_scores = enhanced_random_forest_selection(X_train, y_train, top_k_per_method)
        method_results.append((rf_features, rf_scores))
    except Exception as e:
        print(f"❌ Enhanced Random Forest failed: {e}")
        method_results.append(([], pd.DataFrame()))
    # Method 2: Enhanced Extra Trees
    try:
        et_features, et_scores = enhanced_extra_trees_selection(X_train, y_train, top_k_per_method)
        method_results.append((et_features, et_scores))
    except Exception as e:
        print(f"❌ Enhanced Extra Trees failed: {e}")
        method_results.append(([], pd.DataFrame()))
    # Method 3: Enhanced Mutual Information
    try:
        mi_features, mi_scores = enhanced_mutual_info_selection(X_train, y_train, top_k_per_method)
        method_results.append((mi_features, mi_scores))
    except Exception as e:
        print(f"❌ Enhanced Mutual Information failed: {e}")
        method_results.append(([], pd.DataFrame()))
    # Method 4: Enhanced F-Test
    try:
        f_features, f_scores = enhanced_f_test_selection(X_train, y_train, top_k_per_method)
        method_results.append((f_features, f_scores))
    except Exception as e:
        print(f"❌ Enhanced F-Test failed: {e}")
        method_results.append(([], pd.DataFrame()))
    # Method 5: Domain-Specific Selection
    try:
        domain_features, domain_scores = domain_specific_selection(X_train, y_train, top_k_per_method)
        method_results.append((domain_features, domain_scores))
    except Exception as e:
        print(f"❌ Domain-Specific Selection failed: {e}")
        method_results.append(([], pd.DataFrame()))
    # Check if any methods succeeded
    successful_methods = sum(1 for features, _ in method_results if len(features) > 0)
    if successful_methods == 0:
        print("❌ All enhanced methods failed!")
        return [], pd.DataFrame(), {}
    print(f"\n✅ {successful_methods}/5 enhanced methods completed")
    # Enhanced ensemble voting
    final_features, voting_summary = enhanced_ensemble_voting(method_results, final_top_k)
    # Create report
    enhanced_report = {
        'total_input_features': X_train.shape[1],
        'successful_methods': successful_methods,
        'final_features_selected': len(final_features),
    }
    print(f"\n🎯 ENHANCED PIPELINE COMPLETE")
    print(f"  📊 Final features selected: {len(final_features)}")
    print(f"  📈 Success rate: {successful_methods}/5 methods")
    return final_features, voting_summary, enhanced_report

# --- Rule combination creation ---
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
    # Categorize features for smart pairing
    extreme_features = [f for f in features if re.search(r'_max_|_p(85|90|95)', f)][:8]
    temporal_features = [f for f in features if re.search(r'_trend_|_persistence_|_change_', f)][:8]
    anomaly_features = [f for f in features if 'anom_' in f][:5]
    source_features = [f for f in features if re.search(r'_sw$|_wa', f)][:8]    
    # Strategic pairings
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

# --- Rule evaluation with threshold tracking ---
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
    percentiles = list(range(10, 95, 5))
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
            if accuracy > 0.3:
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
            continue
    # Aggregate results across folds
    if len(fold_results) == 0:
        return None, []
    fold_df = pd.DataFrame(fold_results)
    cv_result = {
        'rule_name': rule['name'],
        'rule_type': rule['type'],
        'n_folds_successful': len(fold_results),
        'f1_mean': fold_df['f1_score'].mean(),
        'precision_mean': fold_df['precision'].mean(),
        'recall_mean': fold_df['recall'].mean(),
        'accuracy_mean': fold_df['accuracy'].mean(),
        'f1_std': fold_df['f1_score'].std(),
        'precision_std': fold_df['precision'].std(),
        'recall_std': fold_df['recall'].std(),
        'accuracy_std': fold_df['accuracy'].std(),
        'f1_min': fold_df['f1_score'].min(),
        'f1_max': fold_df['f1_score'].max(),
        'f1_stability': fold_df['f1_score'].mean() / (fold_df['f1_score'].std() + 1e-6),
        'tp_total': fold_df['tp'].sum(),
        'fp_total': fold_df['fp'].sum(),
        'tn_total': fold_df['tn'].sum(),
        'fn_total': fold_df['fn'].sum(),
    }
    return cv_result, fold_thresholds

# --- Main pipeline function ---
def run_enhanced_cv_pipeline_fast(prediction_data):
    # Exclude non-feature columns and keep only numeric columns
    exclude_cols = ['date', 'event_dummy_1', 'reference_port']  # add any other known non-features here
    candidate_features = [col for col in prediction_data.columns if col not in exclude_cols]
    # Keep only numeric columns
    numeric_features = prediction_data[candidate_features].select_dtypes(include=[np.number]).columns.tolist()
    X = prediction_data[numeric_features].copy()
    y = prediction_data['event_dummy_1'].copy()

    # Drop rows with NaNs in X (and y accordingly)
    non_nan_idx = X.dropna().index
    X = X.loc[non_nan_idx].reset_index(drop=True)
    y = y.loc[non_nan_idx].reset_index(drop=True)

    # Feature selection
    selected_features, voting_summary, _ = enhanced_feature_selection_pipeline(
        X, y,
        top_k_per_method=EnhancedFeatureSelectionConfig.TOP_K_PER_METHOD,
        final_top_k=EnhancedFeatureSelectionConfig.FINAL_TOP_K
    )
    # Cross-validation
    N_FOLDS = 6
    cv_splits = list(TimeSeriesSplit(n_splits=N_FOLDS).split(prediction_data))
    rules = create_enhanced_cv_rule_combinations(selected_features)
    cv_results = []
    for rule in rules:
        cv_result, _ = evaluate_rule_cv_with_thresholds(rule, prediction_data, cv_splits)
        if cv_result is not None:
            cv_results.append(cv_result)
    cv_results_df = pd.DataFrame(cv_results)
    return cv_results_df, voting_summary, selected_features

# --- MAIN EXECUTION ---
def main():
    print("\n🔧 RULE_EVALUATION.PY - CV Pipeline")
    print(f"Run: {config.RUN_PATH}")
    print(f"Reference port: {config.reference_port}")

    # Get input and output paths
    input_files = get_input_files()
    output_files = get_output_files()

    # Load merged features file (output of data_preparation_1.py)
    merged_path = os.path.join(config.run_output_dir, 'df_swan_waverys_merged.csv')
    if not os.path.exists(merged_path):
        print(f"❌ Input file not found: {merged_path}")
        return
    df = pd.read_csv(merged_path)
    print(f"✅ Loaded merged features: {merged_path} ({df.shape})")

    # Run enhanced CV pipeline
    print("\n🚦 Running enhanced CV pipeline...")
    cv_results_df, voting_summary, selected_features = run_enhanced_cv_pipeline_fast(df)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(config.results_output_dir, f'cv_results_{timestamp}.csv')
    voting_summary_path = os.path.join(config.results_output_dir, f'voting_summary_{timestamp}.csv')
    selected_features_path = os.path.join(config.results_output_dir, f'selected_features_{timestamp}.txt')

    # Ensure output directory exists
    output_dir = os.path.dirname(results_path)
    os.makedirs(output_dir, exist_ok=True)

    if cv_results_df is not None and not cv_results_df.empty:
        cv_results_df.to_csv(results_path, index=False)
        print(f"✅ Saved CV results: {results_path}")
    else:
        print("⚠️ No CV results to save.")
    if voting_summary is not None and not voting_summary.empty:
        voting_summary.to_csv(voting_summary_path, index=False)
        print(f"✅ Saved feature voting summary: {voting_summary_path}")
    else:
        print("⚠️ No voting summary to save.")
    if selected_features:
        with open(selected_features_path, 'w') as f:
            for feat in selected_features:
                f.write(f"{feat}\n")
        print(f"✅ Saved selected features: {selected_features_path}")
    else:
        print("⚠️ No selected features to save.")

    print("\n🎉 Rule evaluation completed!")

if __name__ == "__main__":
    main()
