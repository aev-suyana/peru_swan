# """
# CORRECTED ML AEP CALCULATOR - Using Independent Daily Probability Draws
# =====================================================================

# This is the CORRECT approach for ML probability predictions:
# - Each day's probability represents P(port closes | features)
# - Draw random numbers independently for each day
# - No block bootstrap needed (temporal correlation already in ML features)

# Author: Wave Analysis Team  
# Date: 2025
# """

# import os
# import pandas as pd
# import numpy as np
# from datetime import datetime
# import matplotlib.pyplot as plt
# import glob
# from config import config
# from numba import jit

# # --- PARAMETERS ---
# ENABLE_PLOTTING = False  # Set True to enable validation plots

# # --- ECONOMIC PARAMETERS (same as before) ---
# wage_path = os.path.join(config.RAW_DATA_DIR, 'wages_caleta.xlsx')
# try:
#     df_wages = pd.read_excel(wage_path)
#     df_wages.rename(columns={'port_name':'port_name_wages'}, inplace=True)
#     df_wages.rename(columns={'caleta':'port_name'}, inplace=True)
#     df_wages['port_name'] = df_wages['port_name'].str.upper().str.replace(' ', '_')
#     df_wages = df_wages[df_wages['port_name'] != 'PUERTO_SUPE']
#     df_wages['daily_wages'] = df_wages['w_p50']/30
#     df_wages['n_ports'] = df_wages['port_name_wages'].apply(lambda x: len(x.split(',')))
#     df_wages['n_fishermen'] = round(df_wages['fishermen_province']/df_wages['n_ports'])
    
#     port_data = df_wages[df_wages['port_name'] == config.reference_port]
#     if not port_data.empty:
#         N_PARAM = port_data['n_fishermen'].iloc[0]
#         W_PARAM = port_data['daily_wages'].iloc[0]
#     else:
#         N_PARAM = df_wages['n_fishermen'].sum()
#         W_PARAM = (df_wages['daily_wages'] * df_wages['n_fishermen']).sum() / df_wages['n_fishermen'].sum()
    
#     print(f"[ML-AEP] N_PARAM (fishermen): {N_PARAM}")
#     print(f"[ML-AEP] W_PARAM (wage): {W_PARAM}")
# except Exception as e:
#     print(f"⚠️ Could not load wages data: {e}")
#     N_PARAM = 14424.0
#     W_PARAM = 11.27

# # =====================================================================================
# # CORRECTED ML AEP CALCULATION (Independent Daily Draws)
# # =====================================================================================

# @jit(nopython=True, cache=True)
# def calculate_annual_loss_jit(daily_predictions, N, W, min_days):
#     """JIT-compiled annual loss calculation"""
#     if len(daily_predictions) == 0:
#         return 0.0
    
#     total_loss = 0.0
#     i = 0
#     while i < len(daily_predictions):
#         if daily_predictions[i] == 1:
#             # Start of event
#             event_length = 1
#             j = i + 1
#             while j < len(daily_predictions) and daily_predictions[j] == 1:
#                 event_length += 1
#                 j += 1
            
#             if event_length >= min_days:
#                 total_loss += event_length * N * W
            
#             i = j
#         else:
#             i += 1
    
#     return total_loss

# def calculate_ml_aep_correct(daily_probabilities, N, W, min_days=1, n_simulations=4000):
#     """
#     CORRECT ML AEP calculation using independent daily probability draws
    
#     This is the theoretically correct approach for calibrated ML probabilities:
#     - Each probability represents P(port closes | features on this day)
#     - Draw random numbers independently for each day
#     - No temporal correlation needed (already captured in ML features)
#     """
#     print(f"🎲 CORRECTED ML AEP ANALYSIS (Independent Daily Draws)")
#     print("=" * 60)
#     print(f"  Daily probabilities: {len(daily_probabilities)} days")
#     print(f"  Probability range: {daily_probabilities.min():.4f} to {daily_probabilities.max():.4f}")
#     print(f"  Mean probability: {daily_probabilities.mean():.4f}")
#     print(f"  Simulations: {n_simulations}")
#     print(f"  Economic params: {N} fishermen × ${W}/day")
    
#     annual_losses = []
#     event_counts = []
    
#     # Run simulations with independent daily draws
#     for sim in range(n_simulations):
#         np.random.seed(sim)  # For reproducibility
        
#         # Draw random numbers for each day
#         random_draws = np.random.random(len(daily_probabilities))
        
#         # Compare with probabilities to get events
#         daily_events = (random_draws < daily_probabilities).astype(int)
        
#         # Calculate annual loss
#         annual_loss = calculate_annual_loss_jit(daily_events, N, W, min_days)
#         annual_losses.append(annual_loss)
        
#         # Count events for statistics
#         event_counts.append(daily_events.sum())
        
#         # Progress indicator
#         if sim % 1000 == 0:
#             print(f"  Completed {sim}/{n_simulations} simulations...")
    
#     annual_losses = np.array(annual_losses)
#     event_counts = np.array(event_counts)
    
#     print(f"  ✅ Completed {len(annual_losses)} simulations")
    
#     # Calculate AEP curve
#     losses_sorted = np.sort(annual_losses)[::-1]
#     exceedance_prob = np.arange(1, len(losses_sorted)+1) / (len(losses_sorted)+1)
    
#     aep_curve = pd.DataFrame({
#         'loss': losses_sorted,
#         'probability': exceedance_prob
#     })
    
#     # Summary statistics
#     summary = {
#         'mean_loss': float(np.mean(annual_losses)),
#         'std_loss': float(np.std(annual_losses)),
#         'max_loss': float(np.max(annual_losses)),
#         'min_loss': float(np.min(annual_losses)),
#         'zero_prob': float(np.mean(annual_losses == 0)),
#         'method': 'independent_daily_probability_draws',
#         'n_simulations': len(annual_losses),
#         'mean_events_per_year': float(np.mean(event_counts)),
#         'std_events_per_year': float(np.std(event_counts)),
#         'min_days': min_days,
#         'n_fishermen': N,
#         'daily_wage': W
#     }
    
#     print(f"\n📊 Corrected ML AEP Results:")
#     print(f"  Mean annual loss: ${summary['mean_loss']:,.0f}")
#     print(f"  Std annual loss: ${summary['std_loss']:,.0f}")
#     print(f"  Max annual loss: ${summary['max_loss']:,.0f}")
#     print(f"  Zero loss probability: {summary['zero_prob']:.1%}")
#     print(f"  Mean events per year: {summary['mean_events_per_year']:.1f}")
#     print(f"  Std events per year: {summary['std_events_per_year']:.1f}")
    
#     return {
#         'summary': summary,
#         'aep_curve': aep_curve,
#         'annual_losses': annual_losses.tolist(),
#         'event_counts': event_counts.tolist()
#     }

# def calculate_confusion_matrix_analysis(daily_probabilities, observed_events, N, W, min_days=1, n_simulations=1000):
#     """
#     Calculate confusion matrix analysis for ML predictions
#     """
#     print(f"\n🔍 Confusion Matrix Analysis...")
    
#     if observed_events is None or len(observed_events) != len(daily_probabilities):
#         print("  ⚠️ No observed events available - skipping confusion matrix")
#         return None
    
#     observed_events = np.array(observed_events)
    
#     fp_costs = []
#     fn_costs = []
#     tp_costs = []
    
#     for sim in range(n_simulations):
#         np.random.seed(sim)
        
#         # Generate predicted events
#         random_draws = np.random.random(len(daily_probabilities))
#         predicted_events = (random_draws < daily_probabilities).astype(int)
        
#         # Calculate TP, FP, FN event counts
#         tp_events = 0
#         fp_events = 0
#         fn_events = 0
        
#         # Find predicted events
#         i = 0
#         while i < len(predicted_events):
#             if predicted_events[i] == 1:
#                 # Found predicted event
#                 event_start = i
#                 while i < len(predicted_events) and predicted_events[i] == 1:
#                     i += 1
#                 event_end = i
                
#                 # Check if this overlaps with observed events
#                 overlaps = False
#                 for j in range(event_start, event_end):
#                     if j < len(observed_events) and observed_events[j] == 1:
#                         overlaps = True
#                         break
                
#                 if overlaps:
#                     tp_events += 1
#                 else:
#                     fp_events += 1
#             else:
#                 i += 1
        
#         # Find observed events not covered by predictions (FN)
#         i = 0
#         while i < len(observed_events):
#             if observed_events[i] == 1:
#                 # Found observed event
#                 event_start = i
#                 while i < len(observed_events) and observed_events[i] == 1:
#                     i += 1
#                 event_end = i
                
#                 # Check if covered by predictions
#                 covered = False
#                 for j in range(event_start, event_end):
#                     if j < len(predicted_events) and predicted_events[j] == 1:
#                         covered = True
#                         break
                
#                 if not covered:
#                     fn_events += 1
#             else:
#                 i += 1
        
#         # Calculate costs (simplified - assume average event length)
#         fp_cost = fp_events * N * W * 3  # Average FP event length
#         fn_cost = fn_events * N * W * 4  # Average missed event length
#         tp_cost = tp_events * N * W * 3  # Average correct event length
        
#         fp_costs.append(fp_cost)
#         fn_costs.append(fn_cost)
#         tp_costs.append(tp_cost)
    
#     return {
#         'fp_mean_cost': float(np.mean(fp_costs)),
#         'fn_mean_cost': float(np.mean(fn_costs)),
#         'tp_mean_cost': float(np.mean(tp_costs)),
#         'total_mean_cost': float(np.mean(np.array(fp_costs) + np.array(fn_costs) + np.array(tp_costs)))
#     }

# def find_ml_files():
#     """Find latest ML probability files"""
#     results_root = os.path.join(os.getcwd(), 'results', 'cv_results')
#     ml_probs_candidates = sorted(
#         glob.glob(os.path.join(results_root, '**', 'ML_probs_2024.csv'), recursive=True),
#         key=os.path.getmtime, reverse=True
#     )
    
#     if not ml_probs_candidates:
#         return None, None
    
#     ml_probs_path = ml_probs_candidates[0]
#     ml_probs_dir = os.path.dirname(ml_probs_path)
#     ml_threshold_path = os.path.join(ml_probs_dir, 'ML_probs_2024_optimal_threshold.txt')
    
#     return ml_probs_path, ml_threshold_path

# # =====================================================================================
# # MAIN CORRECTED ML AEP CALCULATOR
# # =====================================================================================

# def main():
#     """Main corrected ML AEP calculator"""
#     print("\n🎲 CORRECTED ML AEP CALCULATOR - Independent Daily Draws")
#     print("=" * 70)
#     print("This uses the THEORETICALLY CORRECT approach for ML probabilities:")
#     print("• Each day's probability = P(port closes | features)")
#     print("• Independent random draws (no block bootstrap)")
#     print("• Temporal correlation already captured in ML features")
#     print("=" * 70)
    
#     print(f"Run: {config.RUN_PATH}")
#     print(f"Reference port: {config.reference_port}")
    
#     # Find ML files
#     ml_probs_path, ml_threshold_path = find_ml_files()
    
#     if not ml_probs_path:
#         print("❌ No ML_probs_2024.csv found in results/cv_results.")
#         print("   Please run rule_evaluation.py first to generate ML probabilities.")
#         return
    
#     print(f"✅ Using ML probabilities: {ml_probs_path}")
    
#     # Load ML probabilities
#     ml_probs = pd.read_csv(ml_probs_path, parse_dates=['date'])
#     print(f"✅ Loaded ML probabilities: {ml_probs.shape}")
    
#     # Load original data for observed events
#     merged_path = os.path.join(config.run_output_dir, 'df_swan_waverys_merged.csv')
#     if not os.path.exists(merged_path):
#         print(f"❌ Input file not found: {merged_path}")
#         return
    
#     df = pd.read_csv(merged_path, parse_dates=['date'])
    
#     # Merge data
#     df_ml = df.set_index('date').join(
#         ml_probs.set_index('date')[['calibrated_probability']], 
#         how='inner'
#     )
#     df_ml = df_ml.dropna(subset=['calibrated_probability'])
    
#     print(f"✅ Merged data: {df_ml.shape}")
#     print(f"   Date range: {df_ml.index.min()} to {df_ml.index.max()}")
    
#     # Extract daily probabilities
#     daily_probabilities = df_ml['calibrated_probability'].values
    
#     # Setup observed events
#     observed_events = None
#     if 'event_dummy_1' in df_ml.columns:
#         observed_events = df_ml['event_dummy_1'].values
#         print(f"✅ Found observed events: {observed_events.sum()} out of {len(observed_events)} days")
    
#     # Run CORRECTED ML AEP analysis
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     results_dir = config.results_output_dir
    
#     print(f"\n🎲 Running CORRECTED ML AEP simulation...")
    
#     ml_results = calculate_ml_aep_correct(
#         daily_probabilities=daily_probabilities,
#         N=N_PARAM,
#         W=W_PARAM,
#         min_days=config.MIN_DAYS,
#         n_simulations=config.N_SIMULATIONS
#     )
    
#     # Add confusion matrix analysis
#     cm_results = calculate_confusion_matrix_analysis(
#         daily_probabilities, observed_events, N_PARAM, W_PARAM, 
#         config.MIN_DAYS, n_simulations=1000
#     )
    
#     if cm_results:
#         print(f"\n🔍 Confusion Matrix Results:")
#         print(f"  FP Cost: ${cm_results['fp_mean_cost']:,.0f}")
#         print(f"  FN Cost: ${cm_results['fn_mean_cost']:,.0f}")
#         print(f"  TP Cost: ${cm_results['tp_mean_cost']:,.0f}")
#         print(f"  Total Cost: ${cm_results['total_mean_cost']:,.0f}")
        
#         ml_results['confusion_matrix'] = cm_results
    
#     # Calculate observed yearly losses
#     if observed_events is not None:
#         obs_yearly_losses = {}
#         years = df_ml.index.year
#         for year in np.unique(years):
#             mask = (years == year)
#             obs_loss = calculate_annual_loss_jit(
#                 observed_events[mask], N_PARAM, W_PARAM, config.MIN_DAYS
#             )
#             obs_yearly_losses[int(year)] = float(obs_loss)
        
#         ml_results['obs_yearly_losses'] = obs_yearly_losses
#         print(f"\n📅 Observed Yearly Losses:")
#         for year, loss in obs_yearly_losses.items():
#             print(f"  {year}: ${loss:,.0f}")
    
#     # Save results
#     os.makedirs(results_dir, exist_ok=True)
    
#     # Save summary and curve
#     corrected_summary_path = os.path.join(results_dir, f'corrected_ml_aep_summary_{timestamp}.csv')
#     corrected_curve_path = os.path.join(results_dir, f'corrected_ml_aep_curve_{timestamp}.csv')
    
#     pd.DataFrame([ml_results['summary']]).to_csv(corrected_summary_path, index=False)
#     ml_results['aep_curve'].to_csv(corrected_curve_path, index=False)
    
#     print(f"\n✅ Saved corrected ML AEP summary: {corrected_summary_path}")
#     print(f"✅ Saved corrected ML AEP curve: {corrected_curve_path}")
    
#     # Save observed yearly losses
#     if 'obs_yearly_losses' in ml_results:
#         obs_losses_df = pd.DataFrame(
#             list(ml_results['obs_yearly_losses'].items()), 
#             columns=['year', 'observed_loss']
#         )
#         obs_losses_path = os.path.join(results_dir, f'corrected_ml_observed_yearly_losses_{timestamp}.csv')
#         obs_losses_df.to_csv(obs_losses_path, index=False)
#         print(f"✅ Saved corrected ML observed yearly losses: {obs_losses_path}")
    
#     # Optional: Plot comparison
#     if ENABLE_PLOTTING:
#         plt.figure(figsize=(12, 6))
#         plt.plot(ml_results['aep_curve']['loss'], 
#                 ml_results['aep_curve']['probability'], 
#                 marker='o', color='green', linewidth=2, markersize=3,
#                 label='Corrected ML (Independent Draws)')
        
#         plt.xlabel('Annual Loss ($)')
#         plt.ylabel('Exceedance Probability')
#         plt.title('Corrected ML AEP Curve (Independent Daily Probability Draws)')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
        
#         plot_path = os.path.join(results_dir, f'corrected_ml_aep_plot_{timestamp}.png')
#         plt.savefig(plot_path)
#         print(f"✅ Saved corrected ML AEP plot: {plot_path}")
#         plt.show()
    
#     # Final summary
#     print(f"\n===== CORRECTED ML AEP SUMMARY =====")
#     print(f"Method: Independent daily probability draws")
#     print(f"Mean annual loss: ${ml_results['summary']['mean_loss']:,.0f}")
#     print(f"Zero loss probability: {ml_results['summary']['zero_prob']:.1%}")
#     print(f"Mean events per year: {ml_results['summary']['mean_events_per_year']:.1f}")
#     print(f"Simulations: {ml_results['summary']['n_simulations']}")
    
#     print(f"\n🎉 Corrected ML AEP calculation completed!")
#     print("This approach correctly treats ML probabilities as daily event probabilities.")

# if __name__ == "__main__":
#     main()

"""
CORRECTED ML AEP CALCULATOR - Using Independent Daily Probability Draws
=====================================================================

This is the CORRECT approach for ML probability predictions:
- Each day's probability represents P(port closes | features)
- Draw random numbers independently for each day
- No block bootstrap needed (temporal correlation already in ML features)

Author: Wave Analysis Team  
Date: 2025
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import glob
from config import config
from numba import jit

# --- PARAMETERS ---
ENABLE_PLOTTING = False  # Set True to enable validation plots

# --- ECONOMIC PARAMETERS (same as before) ---
wage_path = os.path.join(config.RAW_DATA_DIR, 'wages_caleta.xlsx')
try:
    df_wages = pd.read_excel(wage_path)
    df_wages.rename(columns={'port_name':'port_name_wages'}, inplace=True)
    df_wages.rename(columns={'caleta':'port_name'}, inplace=True)
    df_wages['port_name'] = df_wages['port_name'].str.upper().str.replace(' ', '_')
    df_wages = df_wages[df_wages['port_name'] != 'PUERTO_SUPE']
    df_wages['daily_wages'] = df_wages['w_p50']/30
    df_wages['n_ports'] = df_wages['port_name_wages'].apply(lambda x: len(x.split(',')))
    df_wages['n_fishermen'] = round(df_wages['fishermen_province']/df_wages['n_ports'])
    
    port_data = df_wages[df_wages['port_name'] == config.reference_port]
    if not port_data.empty:
        N_PARAM = port_data['n_fishermen'].iloc[0]
        W_PARAM = port_data['daily_wages'].iloc[0]
    else:
        N_PARAM = df_wages['n_fishermen'].sum()
        W_PARAM = (df_wages['daily_wages'] * df_wages['n_fishermen']).sum() / df_wages['n_fishermen'].sum()
    
    print(f"[ML-AEP] N_PARAM (fishermen): {N_PARAM}")
    print(f"[ML-AEP] W_PARAM (wage): {W_PARAM}")
except Exception as e:
    print(f"⚠️ Could not load wages data: {e}")
    N_PARAM = 14424.0
    W_PARAM = 11.27

# =====================================================================================
# CORRECTED ML AEP CALCULATION (Independent Daily Draws)
# =====================================================================================

@jit(nopython=True, cache=True)
def calculate_annual_loss_jit(daily_predictions, N, W, min_days):
    """JIT-compiled annual loss calculation"""
    if len(daily_predictions) == 0:
        return 0.0
    
    total_loss = 0.0
    i = 0
    while i < len(daily_predictions):
        if daily_predictions[i] == 1:
            # Start of event
            event_length = 1
            j = i + 1
            while j < len(daily_predictions) and daily_predictions[j] == 1:
                event_length += 1
                j += 1
            
            if event_length >= min_days:
                total_loss += event_length * N * W
            
            i = j
        else:
            i += 1
    
    return total_loss

@jit(nopython=True, cache=True)
def count_events_jit(daily_predictions):
    """JIT-compiled event counting"""
    if len(daily_predictions) == 0:
        return 0
    
    event_count = 0
    i = 0
    while i < len(daily_predictions):
        if daily_predictions[i] == 1:
            # Found start of event
            event_count += 1
            # Skip to end of this event
            while i < len(daily_predictions) and daily_predictions[i] == 1:
                i += 1
        else:
            i += 1
    
    return event_count

def calculate_ml_aep_correct(daily_probabilities, N, W, min_days=1, n_simulations=4000):
    """
    CORRECTED ML AEP calculation using independent daily probability draws
    
    This is the theoretically correct approach for calibrated ML probabilities:
    - Each probability represents P(port closes | features on this day)
    - Draw random numbers independently for each day
    - No temporal correlation needed (already captured in ML features)
    """
    print(f"🎲 CORRECTED ML AEP ANALYSIS (Independent Daily Draws)")
    print("=" * 60)
    print(f"  Daily probabilities: {len(daily_probabilities)} days")
    print(f"  Probability range: {daily_probabilities.min():.4f} to {daily_probabilities.max():.4f}")
    print(f"  Mean probability: {daily_probabilities.mean():.4f}")
    print(f"  Simulations: {n_simulations}")
    print(f"  Economic params: {N} fishermen × ${W}/day")
    
    annual_losses = []
    event_counts = []
    daily_event_totals = []  # Total days with events
    
    # Run simulations with independent daily draws
    for sim in range(n_simulations):
        np.random.seed(sim)  # For reproducibility
        
        # Draw random numbers for each day
        random_draws = np.random.random(len(daily_probabilities))
        
        # Compare with probabilities to get events
        daily_events = (random_draws < daily_probabilities).astype(int)
        
        # Calculate annual loss
        annual_loss = calculate_annual_loss_jit(daily_events, N, W, min_days)
        annual_losses.append(annual_loss)
        
        # Count distinct events (event sequences)
        num_events = count_events_jit(daily_events)
        event_counts.append(num_events)
        
        # Count total days with events
        total_event_days = daily_events.sum()
        daily_event_totals.append(total_event_days)
        
        # Progress indicator
        if sim % 1000 == 0:
            print(f"  Completed {sim}/{n_simulations} simulations...")
    
    annual_losses = np.array(annual_losses)
    event_counts = np.array(event_counts)
    daily_event_totals = np.array(daily_event_totals)
    
    print(f"  ✅ Completed {len(annual_losses)} simulations")
    
    # Calculate AEP curve
    losses_sorted = np.sort(annual_losses)[::-1]
    exceedance_prob = np.arange(1, len(losses_sorted)+1) / (len(losses_sorted)+1)
    
    aep_curve = pd.DataFrame({
        'loss': losses_sorted,
        'probability': exceedance_prob
    })
    
    # Enhanced summary statistics matching other scripts
    summary = {
    # Standard loss metrics
    'mean_loss': float(np.mean(annual_losses)),
    'std_loss': float(np.std(annual_losses)),
    'max_loss': float(np.max(annual_losses)),
    'min_loss': float(np.min(annual_losses)),
    'median_loss': float(np.median(annual_losses)),
    'p99_loss': float(np.percentile(annual_losses, 99)),
    'zero_prob': float(np.mean(annual_losses == 0)),
    
    # ✅ FIXED EVENT METRICS - USE STANDARD COLUMN NAMES
    'mean_events': float(np.mean(event_counts)),  # ✅ Was 'mean_events_per_year'
    'std_events': float(np.std(event_counts)),
    'max_events': float(np.max(event_counts)),
    'median_events': float(np.median(event_counts)),
    'p99_events': float(np.percentile(event_counts, 99)),  # ✅ WAS MISSING!
        
    # Total event days metrics  
    'mean_event_days': float(np.mean(daily_event_totals)),
    'std_event_days': float(np.std(daily_event_totals)),
    'p99_event_days': float(np.percentile(daily_event_totals, 99)),
        
    # Method metadata
    'method': 'independent_daily_probability_draws',
    'n_simulations': len(annual_losses),
    'min_days': min_days,
    'n_fishermen': N,
    'daily_wage': W,
        
    # Additional ML-specific metrics
    'mean_daily_probability': float(daily_probabilities.mean()),
    'std_daily_probability': float(daily_probabilities.std()),
    'max_daily_probability': float(daily_probabilities.max()),
    'min_daily_probability': float(daily_probabilities.min())
    }
    
    print(f"\n📊 Enhanced ML AEP Results:")
    print(f"  Mean annual loss: ${summary['mean_loss']:,.0f}")
    print(f"  P99 annual loss: ${summary['p99_loss']:,.0f}")
    print(f"  Max annual loss: ${summary['max_loss']:,.0f}")
    print(f"  Zero loss probability: {summary['zero_prob']:.1%}")
    print(f"  Mean events per year: {summary['mean_events']:.1f}")
    print(f"  P99 events per year: {summary['p99_events']:.1f}")
    print(f"  Mean event days per year: {summary['mean_event_days']:.1f}")
    print(f"  P99 event days per year: {summary['p99_event_days']:.1f}")
    
    return {
        'summary': summary,
        'aep_curve': aep_curve,
        'annual_losses': annual_losses.tolist(),
        'event_counts': event_counts.tolist(),
        'daily_event_totals': daily_event_totals.tolist()
    }

def calculate_confusion_matrix_analysis(daily_probabilities, observed_events, optimal_threshold, N, W, min_days=1, n_simulations=1000):
    """
    Calculate confusion matrix analysis for ML predictions using optimal threshold
    """
    print(f"\n🔍 Enhanced Confusion Matrix Analysis...")
    print(f"  Using optimal threshold: {optimal_threshold:.4f}")
    
    if observed_events is None or len(observed_events) != len(daily_probabilities):
        print("  ⚠️ No observed events available - skipping confusion matrix")
        return None
    
    observed_events = np.array(observed_events)
    
    # Calculate observed confusion matrix using threshold
    predicted_binary = (daily_probabilities > optimal_threshold).astype(int)
    
    obs_tp = int(np.sum((predicted_binary == 1) & (observed_events == 1)))
    obs_fp = int(np.sum((predicted_binary == 1) & (observed_events == 0)))
    obs_tn = int(np.sum((predicted_binary == 0) & (observed_events == 0)))
    obs_fn = int(np.sum((predicted_binary == 0) & (observed_events == 1)))
    
    obs_precision = obs_tp / (obs_tp + obs_fp) if (obs_tp + obs_fp) > 0 else 0
    obs_recall = obs_tp / (obs_tp + obs_fn) if (obs_tp + obs_fn) > 0 else 0
    obs_accuracy = (obs_tp + obs_tn) / len(observed_events) if len(observed_events) > 0 else 0
    obs_f1 = 2 * obs_tp / (2 * obs_tp + obs_fp + obs_fn) if (2 * obs_tp + obs_fp + obs_fn) > 0 else 0
    
    print(f"  Observed confusion matrix: TP={obs_tp}, FP={obs_fp}, TN={obs_tn}, FN={obs_fn}")
    print(f"  Observed metrics: Precision={obs_precision:.3f}, Recall={obs_recall:.3f}, F1={obs_f1:.3f}")
    
    # Simulate confusion matrix costs
    fp_costs = []
    fn_costs = []
    tp_costs = []
    tp_counts = []
    fp_counts = []
    fn_counts = []
    
    for sim in range(n_simulations):
        np.random.seed(sim)
        
        # Generate predicted events using threshold
        predicted_events = (daily_probabilities > optimal_threshold).astype(int)
        
        # Calculate TP, FP, FN counts for this simulation
        tp_events = 0
        fp_events = 0
        fn_events = 0
        
        # Count event-based TP, FP, FN (simplified approach)
        # For each predicted event sequence, check if it overlaps with observed
        i = 0
        while i < len(predicted_events):
            if predicted_events[i] == 1:
                # Found predicted event
                event_start = i
                while i < len(predicted_events) and predicted_events[i] == 1:
                    i += 1
                event_end = i
                event_length = event_end - event_start
                
                # Check if this overlaps with observed events
                overlaps = False
                for j in range(event_start, event_end):
                    if j < len(observed_events) and observed_events[j] == 1:
                        overlaps = True
                        break
                
                if overlaps:
                    tp_events += 1
                    tp_costs.append(event_length * N * W)
                else:
                    fp_events += 1
                    fp_costs.append(event_length * N * W)
            else:
                i += 1
        
        # Count missed observed events (FN)
        i = 0
        while i < len(observed_events):
            if observed_events[i] == 1:
                # Found observed event
                event_start = i
                while i < len(observed_events) and observed_events[i] == 1:
                    i += 1
                event_end = i
                event_length = event_end - event_start
                
                # Check if covered by predictions
                covered = False
                for j in range(event_start, event_end):
                    if j < len(predicted_events) and predicted_events[j] == 1:
                        covered = True
                        break
                
                if not covered:
                    fn_events += 1
                    fn_costs.append(event_length * N * W)
            else:
                i += 1
        
        tp_counts.append(tp_events)
        fp_counts.append(fp_events)
        fn_counts.append(fn_events)
        
        # Pad cost arrays if needed
        if not tp_costs:
            tp_costs.append(0)
        if not fp_costs:
            fp_costs.append(0)
        if not fn_costs:
            fn_costs.append(0)
    
    # Convert to arrays and calculate statistics
    tp_costs = np.array(tp_costs) if tp_costs else np.array([0])
    fp_costs = np.array(fp_costs) if fp_costs else np.array([0])
    fn_costs = np.array(fn_costs) if fn_costs else np.array([0])
    tp_counts = np.array(tp_counts)
    fp_counts = np.array(fp_counts)
    fn_counts = np.array(fn_counts)
    
    confusion_matrix_summary = {
        # Observed confusion matrix
        'obs_tp': obs_tp,
        'obs_fp': obs_fp,
        'obs_tn': obs_tn,
        'obs_fn': obs_fn,
        'obs_precision': obs_precision,
        'obs_recall': obs_recall,
        'obs_accuracy': obs_accuracy,
        'obs_f1': obs_f1,
        
        # Simulated costs
        'mean_tp_cost': float(np.mean(tp_costs)),
        'mean_fp_cost': float(np.mean(fp_costs)),
        'mean_fn_cost': float(np.mean(fn_costs)),
        'p99_tp_cost': float(np.percentile(tp_costs, 99)),
        'p99_fp_cost': float(np.percentile(fp_costs, 99)),
        'p99_fn_cost': float(np.percentile(fn_costs, 99)),
        
        # Simulated event counts
        'mean_tp': float(np.mean(tp_counts)),
        'mean_fp': float(np.mean(fp_counts)),
        'mean_fn': float(np.mean(fn_counts)),
        'p99_tp': float(np.percentile(tp_counts, 99)),
        'p99_fp': float(np.percentile(fp_counts, 99)),
        'p99_fn': float(np.percentile(fn_counts, 99)),
        
        # Total costs
        'total_mean_cost': float(np.mean(tp_costs) + np.mean(fp_costs) + np.mean(fn_costs)),
        'optimal_threshold': optimal_threshold
    }
    
    print(f"  Simulated costs - TP: ${confusion_matrix_summary['mean_tp_cost']:,.0f}, "
          f"FP: ${confusion_matrix_summary['mean_fp_cost']:,.0f}, "
          f"FN: ${confusion_matrix_summary['mean_fn_cost']:,.0f}")
    
    return confusion_matrix_summary

def find_ml_files():
    """Find latest ML probability files"""
    results_root = os.path.join(os.getcwd(), 'results', 'cv_results')
    ml_probs_candidates = sorted(
        glob.glob(os.path.join(results_root, '**', 'ML_probs_2024.csv'), recursive=True),
        key=os.path.getmtime, reverse=True
    )
    
    if not ml_probs_candidates:
        return None, None
    
    ml_probs_path = ml_probs_candidates[0]
    ml_probs_dir = os.path.dirname(ml_probs_path)
    ml_threshold_path = os.path.join(ml_probs_dir, 'ML_probs_2024_optimal_threshold.txt')
    
    return ml_probs_path, ml_threshold_path

# =====================================================================================
# MAIN CORRECTED ML AEP CALCULATOR
# =====================================================================================

def main():
    """Main corrected ML AEP calculator"""
    print("\n🎲 ENHANCED ML AEP CALCULATOR - Independent Daily Draws")
    print("=" * 70)
    print("This uses the THEORETICALLY CORRECT approach for ML probabilities:")
    print("• Each day's probability = P(port closes | features)")
    print("• Independent random draws (no block bootstrap)")
    print("• Temporal correlation already captured in ML features")
    print("• Enhanced output matching other simulation scripts")
    print("=" * 70)
    
    print(f"Run: {config.RUN_PATH}")
    print(f"Reference port: {config.reference_port}")
    
    # Find ML files
    ml_probs_path, ml_threshold_path = find_ml_files()
    
    if not ml_probs_path:
        print("❌ No ML_probs_2024.csv found in results/cv_results.")
        print("   Please run rule_evaluation.py first to generate ML probabilities.")
        return
    
    print(f"✅ Using ML probabilities: {ml_probs_path}")
    
    # Load optimal threshold
    optimal_threshold = 0.5  # Default
    if ml_threshold_path and os.path.exists(ml_threshold_path):
        try:
            with open(ml_threshold_path, 'r') as f:
                optimal_threshold = float(f.read().strip())
            print(f"✅ Using optimal threshold: {optimal_threshold:.4f}")
        except Exception as e:
            print(f"⚠️ Could not load optimal threshold: {e}, using default 0.5")
    else:
        print(f"⚠️ Optimal threshold file not found, using default 0.5")
    
    # Load ML probabilities
    ml_probs = pd.read_csv(ml_probs_path, parse_dates=['date'])
    print(f"✅ Loaded ML probabilities: {ml_probs.shape}")
    
    # Load original data for observed events
    merged_path = os.path.join(config.run_output_dir, 'df_swan_waverys_merged.csv')
    if not os.path.exists(merged_path):
        print(f"❌ Input file not found: {merged_path}")
        return
    
    df = pd.read_csv(merged_path, parse_dates=['date'])
    
    # Merge data
    df_ml = df.set_index('date').join(
        ml_probs.set_index('date')[['calibrated_probability']], 
        how='inner'
    )
    df_ml = df_ml.dropna(subset=['calibrated_probability'])
    
    print(f"✅ Merged data: {df_ml.shape}")
    print(f"   Date range: {df_ml.index.min()} to {df_ml.index.max()}")
    
    # Extract daily probabilities
    daily_probabilities = df_ml['calibrated_probability'].values
    
    # Setup observed events
    observed_events = None
    if 'event_dummy_1' in df_ml.columns:
        observed_events = df_ml['event_dummy_1'].values
        print(f"✅ Found observed events: {observed_events.sum()} out of {len(observed_events)} days")
    
    # Run ENHANCED ML AEP analysis
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = config.results_output_dir
    
    print(f"\n🎲 Running ENHANCED ML AEP simulation...")
    
    ml_results = calculate_ml_aep_correct(
        daily_probabilities=daily_probabilities,
        N=N_PARAM,
        W=W_PARAM,
        min_days=config.MIN_DAYS,
        n_simulations=config.N_SIMULATIONS
    )
    
    # Add enhanced confusion matrix analysis
    cm_results = calculate_confusion_matrix_analysis(
        daily_probabilities, observed_events, optimal_threshold, 
        N_PARAM, W_PARAM, config.MIN_DAYS, n_simulations=1000
    )
    
    if cm_results:
        # Add confusion matrix results to main results
        ml_results['confusion_matrix_results'] = cm_results
        
        # Add CM metrics to summary for consistency with other scripts
        ml_results['summary'].update({
        'obs_tp': cm_results['obs_tp'],
        'obs_fp': cm_results['obs_fp'],
        'obs_tn': cm_results['obs_tn'],
        'obs_fn': cm_results['obs_fn'],
        'obs_precision': cm_results['obs_precision'],
        'obs_recall': cm_results['obs_recall'],
        'obs_accuracy': cm_results['obs_accuracy'],
        'obs_f1': cm_results['obs_f1'],
        'mean_tp': cm_results['mean_tp'],
        'mean_fp': cm_results['mean_fp'],
        'mean_fn': cm_results['mean_fn'],
        'p99_tp': cm_results['p99_tp'],
        'p99_fp': cm_results['p99_fp'],
        'p99_fn': cm_results['p99_fn'],
        'mean_tp_cost': cm_results['mean_tp_cost'],
        'mean_fp_cost': cm_results['mean_fp_cost'],
        'mean_fn_cost': cm_results['mean_fn_cost'],
        'p99_tp_cost': cm_results['p99_tp_cost'],
        'p99_fp_cost': cm_results['p99_fp_cost'],
        'p99_fn_cost': cm_results['p99_fn_cost'],
        'optimal_threshold': optimal_threshold,
        
        # ✅ ENSURE EVENT METRICS ARE PRESERVED
        'mean_events': ml_results['summary']['mean_events'],
        'p99_events': ml_results['summary']['p99_events']
    })
    
    # Calculate observed yearly losses
    if observed_events is not None:
        obs_yearly_losses = {}
        years = df_ml.index.year
        for year in np.unique(years):
            mask = (years == year)
            obs_loss = calculate_annual_loss_jit(
                observed_events[mask], N_PARAM, W_PARAM, config.MIN_DAYS
            )
            obs_yearly_losses[int(year)] = float(obs_loss)
        
        ml_results['obs_yearly_losses'] = obs_yearly_losses
        
        # Add observed yearly statistics to summary
        yearly_losses = list(obs_yearly_losses.values())
        ml_results['summary'].update({
            'obs_mean_yearly_loss': float(np.mean(yearly_losses)),
            'obs_std_yearly_loss': float(np.std(yearly_losses)),
            'obs_max_yearly_loss': float(np.max(yearly_losses))
        })
        
        print(f"\n📅 Observed Yearly Losses:")
        for year, loss in obs_yearly_losses.items():
            print(f"  {year}: ${loss:,.0f}")
        print(f"  Average: ${np.mean(yearly_losses):,.0f}")
    
    # Save results
    os.makedirs(results_dir, exist_ok=True)
    
    # Save enhanced summary and curve
    enhanced_summary_path = os.path.join(results_dir, f'enhanced_ml_aep_summary_{timestamp}.csv')
    enhanced_curve_path = os.path.join(results_dir, f'enhanced_ml_aep_curve_{timestamp}.csv')
    
    pd.DataFrame([ml_results['summary']]).to_csv(enhanced_summary_path, index=False)
    ml_results['aep_curve'].to_csv(enhanced_curve_path, index=False)
    
    print(f"\n✅ Saved enhanced ML AEP summary: {enhanced_summary_path}")
    print(f"✅ Saved enhanced ML AEP curve: {enhanced_curve_path}")
    
    # Save observed yearly losses
    if 'obs_yearly_losses' in ml_results:
        obs_losses_df = pd.DataFrame(
            list(ml_results['obs_yearly_losses'].items()), 
            columns=['year', 'observed_loss']
        )
        obs_losses_path = os.path.join(results_dir, f'enhanced_ml_observed_yearly_losses_{timestamp}.csv')
        obs_losses_df.to_csv(obs_losses_path, index=False)
        print(f"✅ Saved enhanced ML observed yearly losses: {obs_losses_path}")
    
    # Save confusion matrix detailed results if available
    if cm_results:
        cm_summary_path = os.path.join(results_dir, f'enhanced_ml_confusion_matrix_{timestamp}.csv')
        pd.DataFrame([cm_results]).to_csv(cm_summary_path, index=False)
        print(f"✅ Saved enhanced ML confusion matrix: {cm_summary_path}")
    
    # Optional: Plot comparison
    if ENABLE_PLOTTING:
        plt.figure(figsize=(12, 6))
        plt.plot(ml_results['aep_curve']['loss'], 
                ml_results['aep_curve']['probability'], 
                marker='o', color='green', linewidth=2, markersize=3,
                label='Enhanced ML (Independent Draws)')
        
        plt.xlabel('Annual Loss ($)')
        plt.ylabel('Exceedance Probability')
        plt.title('Enhanced ML AEP Curve (Independent Daily Probability Draws)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(results_dir, f'enhanced_ml_aep_plot_{timestamp}.png')
        plt.savefig(plot_path)
        print(f"✅ Saved enhanced ML AEP plot: {plot_path}")
        plt.show()
    
    # Final enhanced summary
    print(f"\n===== ENHANCED ML AEP SUMMARY =====")
    print(f"Method: Independent daily probability draws")
    print(f"Mean annual loss: ${ml_results['summary']['mean_loss']:,.0f}")
    print(f"P99 annual loss: ${ml_results['summary']['p99_loss']:,.0f}")
    print(f"Zero loss probability: {ml_results['summary']['zero_prob']:.1%}")
    print(f"Mean events per year: {ml_results['summary']['mean_events']:.1f}")
    print(f"P99 events per year: {ml_results['summary']['p99_events']:.1f}")
    print(f"Simulations: {ml_results['summary']['n_simulations']}")
    
    if cm_results:
        print(f"Observed F1 Score: {cm_results['obs_f1']:.3f}")
        print(f"Optimal Threshold: {optimal_threshold:.4f}")
    
    print(f"\n🎉 Enhanced ML AEP calculation completed!")
    print("This approach correctly treats ML probabilities as daily event probabilities")
    print("and provides comprehensive metrics matching other simulation scripts.")

if __name__ == "__main__":
    main()