"""
AEP CALCULATION - FINAL PIPELINE STAGE
======================================

This script loads the best single-feature rule and threshold from CV results
and runs the unified AEP analysis and plotting using the production pipeline structure.

Author: Wave Analysis Team
Date: 2025
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from config import config, get_input_files, get_output_files

# Import numba, tqdm, etc. as in the legacy script
from tqdm.auto import tqdm
from numba import jit, prange
from concurrent.futures import ThreadPoolExecutor
import threading
import gc
import itertools
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# --- PARAMETERS ---
ENABLE_PLOTTING = False  # Set True to enable validation plots
# --- Economic parameters: dynamically load from wages_caleta.xlsx (full cleaning logic) ---
wage_path = os.path.join(config.RAW_DATA_DIR, 'wages_caleta.xlsx')
try:
    df_wages = pd.read_excel(wage_path)
    # Clean and normalize port names as in original script
    df_wages.rename(columns={'port_name':'port_name_wages'}, inplace=True)
    df_wages.rename(columns={'caleta':'port_name'}, inplace=True)
    df_wages['port_name'] = df_wages['port_name'].str.upper()
    df_wages['port_name'] = df_wages['port_name'].str.replace(' ', '_')
    df_wages = df_wages[df_wages['port_name'] != 'PUERTO_SUPE']
    df_wages['daily_wages'] = df_wages['w_p50']/30
    df_wages['port_name'] = df_wages['port_name'].str.replace('CALETA_EL_CHACO','CALETA_EL_CHACHO')
    df_wages['port_name'] = df_wages['port_name'].str.replace('CALETA_CULEBRAS','COLETA_CULEBRAS')
    df_wages['port_name'] = df_wages['port_name'].str.replace('CALETA_LOBITOS_(TALARA)','CALETA_LOBITOS')
    df_wages['port_name'] = df_wages['port_name'].str.replace('CALETA_SAN_ANDRÉS','CALETA_SAN_ANDRES')
    df_wages['port_name'] = df_wages['port_name'].str.replace('CALLAO','DPA_CALLAO')
    df_wages['port_name'] = df_wages['port_name'].str.replace('CHORRILLOS','DPA_CHORRILLOS')
    df_wages['port_name'] = df_wages['port_name'].str.replace('ENSENADA_DE_SECHURA','ENSENADA_SECHURA')
    df_wages['port_name'] = df_wages['port_name'].str.replace('PUERTO_MATARANI_(MUELLE_OCEAN_FISH)','MUELLE_OCEAN_FISH')
    df_wages['port_name'] = df_wages['port_name'].str.replace('PUNTA_PICATA', 'TERMINAL_PESQUERO_PUNTA_PICATA')
    df_wages['port_name'] = df_wages['port_name'].str.replace('CALETA_EL_CHACO','CALETA_EL_CHACHO')
    df_wages['n_ports'] = df_wages['port_name_wages'].apply(lambda x: len(x.split(',')))
    df_wages['n_fishermen'] = round(df_wages['fishermen_province']/df_wages['n_ports'])
    df_wages_join = df_wages[['port_name', 'daily_wages', 'n_fishermen']].reset_index(drop=True)
    df_wages_join['daily_losses'] = df_wages_join['n_fishermen'] * df_wages_join['daily_wages']

    avg_wages = df_wages_join['daily_wages'].median()
    avg_fishermen = df_wages_join['n_fishermen'].median()
    avg_losses = df_wages_join['daily_losses'].median()
    # Add row for PUERTO_CHIMBOTE with mean values
    chimbote_row = pd.DataFrame({
        'port_name': ['PUERTO_CHIMBOTE'],
        'daily_wages': [avg_wages],
        'n_fishermen': [avg_fishermen],
        'daily_losses': [avg_losses]
    })
    coishco_row = pd.DataFrame({
        'port_name': ['CALETA_COISHCO'],
        'daily_wages': [avg_wages],
        'n_fishermen': [avg_fishermen],
        'daily_losses': [avg_losses]
    })

    eldorado_row = pd.DataFrame({
        'port_name': ['CALETA_EL_DORADO'],
        'daily_wages': [avg_wages],
        'n_fishermen': [avg_fishermen],
        'daily_losses': [avg_losses]
    })

    elnuro_row = pd.DataFrame({
        'port_name': ['CALETA_EL_ÑURO'],
        'daily_wages': [avg_wages],
        'n_fishermen': [avg_fishermen],
        'daily_losses': [avg_losses]
    })

    santa_row = pd.DataFrame({
        'port_name': ['CALETA_SANTA'],
        'daily_wages': [avg_wages],
        'n_fishermen': [avg_fishermen],
        'daily_losses': [avg_losses]
    })

    ilo_row = pd.DataFrame({
        'port_name': ['MUELLE_FISCAL_ILO'],
        'daily_wages': [avg_wages],
        'n_fishermen': [avg_fishermen],
        'daily_losses': [avg_losses]
    })

    callao_row = pd.DataFrame({
        'port_name': ['DPA_CALLAO'],
        'daily_wages': [avg_wages],
        'n_fishermen': [avg_fishermen],
        'daily_losses': [avg_losses]
    })

    chorrillos_row = pd.DataFrame({
        'port_name': ['DPA_CHORRILLOS'],
        'daily_wages': [avg_wages],
        'n_fishermen': [avg_fishermen],
        'daily_losses': [avg_losses]
    })

    huarmey_row = pd.DataFrame({
        'port_name': ['PUERTO_HUARMEY'],
        'daily_wages': [avg_wages],
        'n_fishermen': [avg_fishermen],
        'daily_losses': [avg_losses]
    })

    samanco_row = pd.DataFrame({
        'port_name': ['PUERTO_SAMANCO'],
        'daily_wages': [avg_wages],
        'n_fishermen': [avg_fishermen],
        'daily_losses': [avg_losses]
    })

    supe_row = pd.DataFrame({
        'port_name': ['PUERTO_SUPE'],
        'daily_wages': [avg_wages],
        'n_fishermen': [avg_fishermen],
        'daily_losses': [avg_losses]
    })

    multi_row = pd.DataFrame({
        'port_name': ['TERMINAL_MULTIBOYAS'],
        'daily_wages': [avg_wages],
        'n_fishermen': [avg_fishermen],
        'daily_losses': [avg_losses]
    })

    df_wages_join = pd.concat([df_wages_join, chimbote_row, coishco_row, eldorado_row, elnuro_row, santa_row, ilo_row, callao_row, chorrillos_row, huarmey_row, samanco_row, supe_row, multi_row], ignore_index=True)

    # df_wages_join.to_csv(f'{path_out}/df_wages_join.csv')
    wage_port_list = df_wages_join['port_name'].unique()
    # --- Port group aggregation logic ---
    run_path = config.RUN_PATH if hasattr(config, 'RUN_PATH') else 'UNKNOWN'
    port_list = None
    if run_path == 'run_g1':
        port_list = ['CALETA_CANCAS', 'CALETA_GRAU', 'CALETA_ACAPULCO', 'CALETA_CRUZ', 
                    'PUERTO_PIZARRO', 'PUERTO_ZORRITOS']
    elif run_path == 'run_g2':
        port_list = ['BALNEARIO_DE_PUNTA_SAL', 'CALETA_MANCORA', 'CALETA_LOS_ORGANOS', 'CALETA_NURO',
                     'CALETA_CABO_BLANCO']
    elif run_path == 'run_g3':
        port_list = ['CALETA_COLAN', 'PUERTO_BAYOVAR', 'CALETA_YACILLA', 'CALETA_ISLILLA', 
                    'COLETA_TORTUGA', 'CALETA_CHULLILLACHE', 'CALETA_CONSTANTE', 'CALETA_MATACABALLO', 
                    'CALETA_TIERRA_COLORADA', 'CALETA_DELICIAS', 'CALETA_PARACHIQUE', 'CALETA_PUERTO_RICO', 
                    'PUERTO_PAITA', 'ENSENADA_SECHURA']
    elif run_path == 'run_g4':
        port_list = ['CALETA_DE_SAN_JOSE', 'PUERTO_ETEN', 'PUERTO_PIMENTEL', 'CALETA_DE_SANTA_ROSA']
    elif run_path == 'run_g5':
        port_list = ['PUERTO_CHIMBOTE', 'CALETA_COISHCO', 'CALETA_DORADO', 'CALETA_ÑURO', 
                    'CALETA_SANTA', 'COLETA_CULEBRAS', 'CALETA_CHIMUS', 'PUERTO_HUARMEY', 
                    'PUERTO_SAMANCO', 'PUERTO_CASMA', 'CALETA_VIDAL', 'CALETA_GRAMITA',
                    'COLETA_TORTUGAS']
    elif run_path == 'run_g6':
        port_list = ['ANCON', 'DPA_CALLAO', 'DPA_CHORRILLOS', 'PUCUSANA']
    elif run_path == 'run_g7':
        port_list = ['CALETA_NAZCA', 'PUERTO_SAN_NICOLAS', 'PUERTO_SAN_JUAN', 'CALETA_LOMAS', 
                    'CALETA_TANAKA', 'CALETA_PUERTO_VIEJO', 'CALETA_CHALA']
    elif run_path == 'run_g8':
        port_list = ['CALETA_NAZCA', 'PUERTO_SAN_NICOLAS', 'PUERTO_SAN_JUAN', 'CALETA_LOMAS',
                     'CALETA_TANAKA', 'CALETA_CHALA']
    elif run_path == 'run_g9':
        port_list = ['CALETA_ATICO', 'CALETA_PLANCHADA', 'CALETA_QUILCA', 'MUELLE_OCEAN_FISH',
                     'CALETA_FARO']
    elif run_path == 'run_g10':
        port_list = ['DPA_ILO', 'MUELLE_FISCAL_ILO', 'TERMINAL_PESQUERO_PUNTA_PICATA', 'DPA_MORRO_SAMA', 
                    'DPA_VILA_VILA']

    if port_list is not None:
        N_PARAM = df_wages_join[df_wages_join['port_name'].isin(port_list)]['n_fishermen'].sum()
        # Use the same averaging formula as user: average daily_wages across group, then divide by 3.5
        W_PARAM = (df_wages_join[df_wages_join['port_name'].isin(port_list)]['daily_wages'].sum()/len(port_list))/3.5
        print(f"[AEP] Aggregated N_PARAM (fishermen): {N_PARAM}")
        print(f"[AEP] Aggregated W_PARAM (wage): {W_PARAM}")
    else:
        # Fallback: single port matching as before
        port_name = config.reference_port.upper().replace(' ', '_')
        row = df_wages_join[df_wages_join['port_name'] == port_name]
        if not row.empty:
            N_PARAM = float(row.iloc[0]['n_fishermen'])
            W_PARAM = float(row.iloc[0]['daily_wages'])
        else:
            N_PARAM = 1
            W_PARAM = 1
            print(f"[AEP] Warning: Port '{port_name}' not found in wage data, using defaults.")
except Exception as e:
    N_PARAM = 1
    W_PARAM = 1
    print(f"[AEP] Warning: Could not load wage data: {e}. Using defaults.")

# --- JIT and simulation functions (ported from legacy) ---
from datetime import timedelta
from numba import jit
import numpy as np

@jit(nopython=True, cache=True)
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

@jit(nopython=True, cache=True)
def find_events_in_category_jit(predicted, observed, category_code):
    durations = []
    current_event_length = 0
    for i in range(len(predicted)):
        in_event = False
        if category_code == 0:
            in_event = (predicted[i] == 1) and (observed[i] == 0)
        elif category_code == 1:
            in_event = (predicted[i] == 0) and (observed[i] == 1)
        elif category_code == 2:
            in_event = (predicted[i] == 1) and (observed[i] == 1)
        if in_event:
            current_event_length += 1
        else:
            if current_event_length > 0:
                durations.append(current_event_length)
            current_event_length = 0
    if current_event_length > 0:
        durations.append(current_event_length)
    return durations

@jit(nopython=True, cache=True)
def calculate_cm_costs_jit(predicted_events, observed_events, N, W, min_days):
    fp_durations = find_events_in_category_jit(predicted_events, observed_events, 0)
    fn_durations = find_events_in_category_jit(predicted_events, observed_events, 1)
    tp_durations = find_events_in_category_jit(predicted_events, observed_events, 2)
    fp_cost = 0
    for duration in fp_durations:
        if duration >= min_days:
            fp_cost += N * W * duration
    fn_cost = 0
    for duration in fn_durations:
        if duration >= min_days:
            fn_cost += N * W * duration
    tp_cost = 0
    for duration in tp_durations:
        if duration >= min_days:
            tp_cost += N * W * duration
    return fp_cost, fn_cost, tp_cost

def vectorized_block_bootstrap(daily_clean, n_simulations, block_length=7, window_days=20, days_per_year=365):
    from datetime import datetime
    available_years = sorted(daily_clean.index.year.unique())
    n_days = len(daily_clean)
    print("  Pre-computing valid block positions...")
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
    print("  Generating all simulation indices...")
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

def process_simulation_batch_threaded(batch_indices, trigger_values_matrix, observed_matrix, N, W, min_days, trigger_threshold, has_observed):
    batch_losses = []
    batch_fp_costs = []
    batch_fn_costs = []
    batch_tp_costs = []
    batch_event_counts = []
    for sim_indices in batch_indices:
        try:
            trigger_vals = trigger_values_matrix[sim_indices]
            predicted_events = (trigger_vals > trigger_threshold).astype(np.int32)
            total_loss = calculate_annual_loss_jit(predicted_events, N, W, min_days)
            batch_losses.append(total_loss)
            n_events = np.sum(predicted_events)
            if has_observed:
                observed_vals = observed_matrix[sim_indices]
                fp_cost, fn_cost, tp_cost = calculate_cm_costs_jit(predicted_events, observed_vals, N, W, min_days)
                batch_fp_costs.append(fp_cost)
                batch_fn_costs.append(fn_cost)
                batch_tp_costs.append(tp_cost)
                fp_events = len(find_events_in_category_jit(predicted_events, observed_vals, 0))
                fn_events = len(find_events_in_category_jit(predicted_events, observed_vals, 1))
                tp_events = len(find_events_in_category_jit(predicted_events, observed_vals, 2))
                batch_event_counts.append({'fp_events': fp_events, 'fn_events': fn_events, 'tp_events': tp_events, 'n_events': n_events, 'tn_events': 0})
            else:
                # No observed, so just store n_events
                batch_event_counts.append({'fp_events': 0, 'fn_events': 0, 'tp_events': 0, 'tn_events': 0, 'n_events': n_events})
        except Exception as e:
            batch_losses.append(0)
            if has_observed:
                batch_fp_costs.append(0)
                batch_fn_costs.append(0)
                batch_tp_costs.append(0)
                batch_event_counts.append({'fp_events': 0, 'fn_events': 0, 'tp_events': 0, 'tn_events': 0, 'n_events': 0})
            else:
                batch_event_counts.append({'fp_events': 0, 'fn_events': 0, 'tp_events': 0, 'tn_events': 0, 'n_events': 0})
    return batch_losses, batch_fp_costs, batch_fn_costs, batch_tp_costs, batch_event_counts

def calculate_aep_curve_efficient(annual_losses, trigger_feature, trigger_threshold, min_days, N, W):
    losses_sorted = np.sort(annual_losses)[::-1]
    exceedance_prob = np.arange(1, len(losses_sorted)+1) / (len(losses_sorted)+1)
    return pd.DataFrame({'loss': losses_sorted, 'probability': exceedance_prob})

def calculate_aep_curve_cm(costs, cost_type, trigger_feature, trigger_threshold, min_days, N, W):
    costs_sorted = np.sort(costs)[::-1]
    exceedance_prob = np.arange(1, len(costs_sorted)+1) / (len(costs_sorted)+1)
    return pd.DataFrame({'cost_type': cost_type, 'loss': costs_sorted, 'probability': exceedance_prob})

def calculate_unified_aep_analysis_fast(swh_data, trigger_feature, trigger_threshold, N, W, min_days=None, n_simulations=None, observed_events=None, block_length=None, window_days=None, n_jobs=-1):
    print(f"🚀 SPEED-OPTIMIZED UNIFIED AEP ANALYSIS")
    print("=" * 50)
    print(f"  Data: {len(swh_data)} observations")
    print(f"  Trigger: {trigger_feature} > {trigger_threshold}")
    print(f"  Port: {N} fishermen × ${W}/day")
    print(f"  Min event: {min_days} days")
    print(f"  Block length: {block_length} days")
    print(f"  Simulations: {n_simulations}")
    if trigger_feature not in swh_data.columns:
        print(f"ERROR: {trigger_feature} not found in input DataFrame!")
        return None
    daily_clean = swh_data.copy().sort_index()
    daily_clean = daily_clean.dropna(subset=[trigger_feature])
    if len(daily_clean) == 0:
        print("ERROR: No valid rows with trigger feature after cleaning!")
        return None
    print(f"  Using {len(daily_clean)} days for simulation.")
    has_observed = observed_events is not None
    if has_observed:
        observed_aligned = observed_events.reindex(daily_clean.index, fill_value=0)
        print(f"  Observed events: {observed_aligned.sum()} out of {len(observed_aligned)} days")
        observed_matrix = observed_aligned.values.astype(np.int32)
    else:
        print("  No observed events provided - confusion matrix analysis will be skipped.")
        observed_matrix = None
    print("  Pre-computing trigger values...")
    trigger_values_matrix = daily_clean[trigger_feature].values.astype(np.float32)
    print("  Generating block bootstrap samples...")
    all_simulation_indices = vectorized_block_bootstrap(
        daily_clean, n_simulations, block_length, window_days, days_per_year=365
    )
    if n_jobs == -1:
        import multiprocessing as mp
        n_jobs = mp.cpu_count()
    n_jobs = min(n_jobs, n_simulations)
    print(f"  Processing {n_simulations} simulations using {n_jobs} parallel workers...")
    batch_size = max(1, n_simulations // n_jobs)
    simulation_batches = []
    for i in range(0, n_simulations, batch_size):
        end_idx = min(i + batch_size, n_simulations)
        batch_indices = all_simulation_indices[i:end_idx]
        simulation_batches.append(batch_indices)
    all_losses = []
    all_fp_costs = []
    all_fn_costs = []
    all_tp_costs = []
    all_event_counts = []
    if n_jobs == 1:
        print("  Using single-threaded processing...")
        for batch_indices in tqdm(simulation_batches, desc="Processing batches"):
            batch_losses, batch_fp, batch_fn, batch_tp, batch_events = process_simulation_batch_threaded(
                batch_indices, trigger_values_matrix, observed_matrix, 
                N, W, min_days, trigger_threshold, has_observed
            )
            all_losses.extend(batch_losses)
            if has_observed:
                all_fp_costs.extend(batch_fp)
                all_fn_costs.extend(batch_fn)
                all_tp_costs.extend(batch_tp)
                all_event_counts.extend(batch_events)
    else:
        from concurrent.futures import ThreadPoolExecutor
        print(f"  Using {n_jobs} threads for parallel processing...")
        def process_batch_wrapper(batch_indices):
            return process_simulation_batch_threaded(
                batch_indices, trigger_values_matrix, observed_matrix, 
                N, W, min_days, trigger_threshold, has_observed
            )
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            from tqdm.auto import tqdm
            results = list(tqdm(
                executor.map(process_batch_wrapper, simulation_batches),
                total=len(simulation_batches),
                desc="Processing batches"
            ))
        for batch_losses, batch_fp, batch_fn, batch_tp, batch_events in results:
            all_losses.extend(batch_losses)
            if has_observed:
                all_fp_costs.extend(batch_fp)
                all_fn_costs.extend(batch_fn)
                all_tp_costs.extend(batch_tp)
                all_event_counts.extend(batch_events)
    annual_losses = np.array(all_losses)
    print(f"  Completed {len(annual_losses)} simulations successfully.")
    print("  Calculating standard AEP curve...")
    standard_aep_curve = calculate_aep_curve_efficient(
        annual_losses, trigger_feature, trigger_threshold, min_days, N, W
    )
    standard_summary = {
        'mean_loss': float(np.mean(annual_losses)),
        'std_loss': float(np.std(annual_losses)),
        'max_loss': float(np.max(annual_losses)),
        'zero_prob': float(np.mean(annual_losses == 0)),
        'method': 'speed_optimized_block_bootstrap',
        'block_length': block_length,
        'window_days': window_days,
        'trigger_feature': trigger_feature,
        'trigger_threshold': trigger_threshold,
        'min_days': min_days,
        'n_fishermen': N,
        'daily_wage': W,
        'n_simulations': len(annual_losses),
        'n_jobs': n_jobs
    }
    print(f"\n📊 Standard AEP Results:")
    print(f"  Mean annual loss: ${standard_summary['mean_loss']:,.0f}")
    print(f"  Max annual loss: ${standard_summary['max_loss']:,.0f}")
    print(f"  Zero loss probability: {standard_summary['zero_prob']:.1%}")
    confusion_matrix_results = None
    if has_observed and all_fp_costs:
        print("  Calculating confusion matrix AEP curves...")
        fp_annual_costs = np.array(all_fp_costs)
        fn_annual_costs = np.array(all_fn_costs)
        tp_annual_costs = np.array(all_tp_costs)
        fp_aep = calculate_aep_curve_cm(fp_annual_costs, 'FP_Cost', trigger_feature, trigger_threshold, min_days, N, W)
        fn_aep = calculate_aep_curve_cm(fn_annual_costs, 'FN_Cost', trigger_feature, trigger_threshold, min_days, N, W)
        tp_aep = calculate_aep_curve_cm(tp_annual_costs, 'TP_Cost', trigger_feature, trigger_threshold, min_days, N, W)
        insurance_annual_costs = fp_annual_costs + tp_annual_costs
        insurance_aep = calculate_aep_curve_cm(insurance_annual_costs, 'Insurance_Cost', trigger_feature, trigger_threshold, min_days, N, W)
        cm_summary = {
            'fp_costs': {
                'mean': float(np.mean(fp_annual_costs)),
                'std': float(np.std(fp_annual_costs)),
                'max': float(np.max(fp_annual_costs)),
                'zero_prob': float(np.mean(fp_annual_costs == 0))
            },
            'fn_costs': {
                'mean': float(np.mean(fn_annual_costs)),
                'std': float(np.std(fn_annual_costs)),
                'max': float(np.max(fn_annual_costs)),
                'zero_prob': float(np.mean(fn_annual_costs == 0))
            },
            'tp_costs': {
                'mean': float(np.mean(tp_annual_costs)),
                'std': float(np.std(tp_annual_costs)),
                'max': float(np.max(tp_annual_costs)),
                'zero_prob': float(np.mean(tp_annual_costs == 0))
            },
            'insurance_costs': {
                'mean': float(np.mean(insurance_annual_costs)),
                'std': float(np.std(insurance_annual_costs)),
                'max': float(np.max(insurance_annual_costs)),
                'zero_prob': float(np.mean(insurance_annual_costs == 0))
            }
        }
        print(f"\n🔍 Confusion Matrix Results:")
        print(f"  FP Cost: ${cm_summary['fp_costs']['mean']:,.0f}")
        print(f"  FN Cost: ${cm_summary['fn_costs']['mean']:,.0f}")
        print(f"  TP Cost: ${cm_summary['tp_costs']['mean']:,.0f}")
        print(f"  Total Insurance Cost: ${cm_summary['insurance_costs']['mean']:,.0f}")
        if all_event_counts:
            fp_events_avg = np.mean([ec['fp_events'] for ec in all_event_counts])
            fn_events_avg = np.mean([ec['fn_events'] for ec in all_event_counts])
            tp_events_avg = np.mean([ec['tp_events'] for ec in all_event_counts])
            print(f"\n📊 Average Event Counts per Simulation:")
            print(f"  FP Events (False Alarms): {fp_events_avg:.1f}")
            print(f"  FN Events (Missed): {fn_events_avg:.1f}")
            print(f"  TP Events (Correct): {tp_events_avg:.1f}")
            print(f"  Total Events: {fp_events_avg + fn_events_avg + tp_events_avg:.1f}")
        # Compute observed stats if possible
        observed_stats = None
        if has_observed and observed_matrix is not None:
            # Calculate observed TP, FP, TN, FN for the entire period
            predicted = (trigger_values_matrix > trigger_threshold).astype(int)
            observed = observed_matrix
            tp = np.sum((predicted == 1) & (observed == 1))
            fp = np.sum((predicted == 1) & (observed == 0))
            tn = np.sum((predicted == 0) & (observed == 0))
            fn = np.sum((predicted == 0) & (observed == 1))
            observed_stats = {
                'obs_tp': int(tp),
                'obs_fp': int(fp),
                'obs_tn': int(tn),
                'obs_fn': int(fn)
            }
        confusion_matrix_results = {
            'fp_aep': fp_aep,
            'fn_aep': fn_aep,
            'tp_aep': tp_aep,
            'insurance_aep': insurance_aep,
            'fp_annual_costs': fp_annual_costs,
            'fn_annual_costs': fn_annual_costs,
            'tp_annual_costs': tp_annual_costs,
            'insurance_annual_costs': insurance_annual_costs,
            'summary': cm_summary,
            'all_event_counts': all_event_counts if all_event_counts else [],
            'observed_stats': observed_stats
        }
    unified_results = {
        'standard_summary': standard_summary,
        'standard_aep_curve': standard_aep_curve,
        'confusion_matrix_results': confusion_matrix_results
    }

    # Observed yearly losses (if observed_events is provided)
    obs_yearly_losses = None
    if has_observed:
        observed_aligned = observed_events.reindex(daily_clean.index, fill_value=0)
        observed_years = daily_clean.index.year
        obs_yearly_losses = {}
        for year in np.unique(observed_years):
            mask = (observed_years == year)
            obs_loss = calculate_annual_loss_jit(observed_aligned.values[mask], N, W, min_days)
            obs_yearly_losses[int(year)] = float(obs_loss)
    if obs_yearly_losses is not None:
        unified_results['obs_yearly_losses'] = obs_yearly_losses

    return unified_results

# --- Dual-condition rule parser ---
def parse_dual_condition_rule(rule_str):
    """
    Parse a dual-condition rule string of the form:
    'feature1 > t1 AND feature2 > t2' or 'feature1 > t1 OR feature2 > t2'
    Returns (feature1, logic, feature2) where logic is 'AND' or 'OR'.
    """
    import re
    # Regex to match: <feature1> > t1 <logic> <feature2> > t2
    pattern = r"(.+?) > t1 (AND|OR) (.+?) > t2"
    match = re.match(pattern, rule_str.strip())
    if not match:
        raise ValueError(f"Could not parse dual-condition rule: {rule_str}")
    feature1 = match.group(1).strip()
    logic = match.group(2).strip()
    feature2 = match.group(3).strip()
    return feature1, logic, feature2

# --- Helper: Find best single rule and all thresholds ---
def load_best_single_rule_and_all_thresholds(cv_results_path, folds_dir):
    import pandas as pd
    import numpy as np
    import os

    cv_df = pd.read_csv(cv_results_path)
    single_rules = cv_df[cv_df['rule_type'] == 'single'].copy()
    if single_rules.empty:
        raise ValueError("No single-feature rules found in CV results.")
    # Sort by f1_mean, then precision_mean, then recall_mean
    single_rules = single_rules.sort_values(['f1_mean', 'precision_mean', 'recall_mean'], ascending=False)
    best_rule = single_rules.iloc[0]
    rule_name = best_rule['rule_name']
    feature = rule_name.replace(' > threshold', '').replace('Single: ', '').strip()

    # Now get all fold thresholds for this rule from fold_thresholds file
    fold_files = [f for f in os.listdir(folds_dir) if f.startswith('fold_thresholds_') and f.endswith('.csv')]
    if not fold_files:
        raise ValueError("No fold_thresholds_*.csv file found in results directory. Run rule_evaluation.py first.")
    latest_fold = sorted(fold_files)[-1]
    folds_df = pd.read_csv(os.path.join(folds_dir, latest_fold))
    folds = folds_df[(folds_df['rule_type'] == 'single') & (folds_df['rule_name'] == rule_name)]
    if folds.empty:
        raise ValueError(f"No fold thresholds found for rule: {rule_name}")
    # The threshold is in the column named after the feature
    if feature in folds.columns:
        thresholds = folds[feature].values.astype(float)
        thresholds = thresholds[~np.isnan(thresholds)]
    else:
        raise ValueError(f"No threshold column for feature '{feature}' found in fold thresholds file. Columns are: {list(folds.columns)}")
    return feature, thresholds

# --- Helper: Plot rule condition (for validation) ---
def plot_rule_condition(df, feature, threshold, event_col='event_dummy_1', save_path=None):
    plt.figure(figsize=(12, 5))
    plt.plot(df['date'], df[feature], label=feature)
    plt.axhline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.2f}')
    plt.scatter(df['date'], df[event_col], color='orange', alpha=0.3, label='Events')
    plt.legend()
    plt.title(f'Best Rule: {feature} > {threshold:.2f}')
    plt.xlabel('Date')
    plt.ylabel(feature)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

#-----------------------------------------------
def identify_if_derived_feature(trigger_feature):
    """
    Determine if a trigger feature is derived and needs full pipeline processing
    """
    derived_patterns = [
        '_deseasonalized', '_detrended', '_processed',
        '_persistence_', '_trend_', '_change_', '_lag_',
        '_rolling_mean_', 'anom_', '_abs_change_', '_rel_change_'
    ]
    
    is_derived = any(pattern in trigger_feature for pattern in derived_patterns)
    
    if is_derived:
        print(f"🔧 DERIVED FEATURE DETECTED: {trigger_feature}")
        print("   Will use full pipeline bootstrap approach")
    else:
        print(f"✅ RAW FEATURE DETECTED: {trigger_feature}")
        print("   Will use fast direct sampling approach")
    
    return is_derived

def extract_base_features_for_pipeline(df_merged):
    """
    Fixed version that correctly identifies base features based on your data_preparation_1.py structure
    """
    print(f"🔍 DEBUGGING: Available columns in data:")
    print(f"   Total columns: {len(df_merged.columns)}")
    
    # Show sample of column names to understand the pattern
    sample_cols = [col for col in df_merged.columns if 'swh' in col.lower()][:20]
    print(f"   Sample SWH columns: {sample_cols}")
    
    base_features = []
    
    # Strategy 1: Look for base features that can be used in the pipeline
    # Based on your data_preparation_1.py, the base features should be:
    # - Raw wave features (before _deseasonalized, _detrended suffixes)
    # - Features that end with specific patterns from SWAN/WAVERYS processing
    
    # Common wave feature patterns from SWAN processing
    wave_base_patterns = [
        # SWAN patterns (assuming these are your base features)
        'swh_mean_sw', 'swh_max_sw', 'swh_median_sw', 'swh_p80_sw', 'swh_p75_sw', 'swh_p25_sw',
        'swh_p90_sw', 'swh_p95_sw', 'swh_p10_sw', 'swh_p20_sw', 'swh_std_sw', 'swh_min_sw',
        'swh_mean_wa', 'swh_max_wa', 'swh_median_wa', 'swh_p80_wa', 'swh_p75_wa', 'swh_p25_wa',
        'swh_p90_wa', 'swh_p95_wa', 'swh_p10_wa', 'swh_p20_wa', 'swh_std_wa', 'swh_min_wa',
        
        # Alternative patterns without _sw/_wa
        'swh_mean', 'swh_max', 'swh_median', 'swh_p80', 'swh_p75', 'swh_p25',
        'swh_p90', 'swh_p95', 'swh_p10', 'swh_p20', 'swh_std', 'swh_min',
        
        # Climate/reference patterns
        'clima_swh_mean_sw', 'clima_swh_mean_wa', 'clima_swh_max_sw', 'clima_swh_max_wa',
        
        # SWE (wind wave) patterns if they exist
        'swe_mean_sw', 'swe_max_sw', 'swe_median_sw', 'swe_p80_sw', 'swe_p75_sw', 'swe_p25_sw',
        'swe_mean_wa', 'swe_max_wa', 'swe_median_wa', 'swe_p80_wa', 'swe_p75_wa', 'swe_p25_wa',
    ]
    
    # Find features matching these patterns
    strategy1_features = [feat for feat in wave_base_patterns if feat in df_merged.columns]
    base_features.extend(strategy1_features)
    print(f"   Strategy 1 (known wave patterns): {len(strategy1_features)} features")
    if strategy1_features:
        print(f"     Examples: {strategy1_features[:5]}")
    
    # Strategy 2: Auto-detect base features by finding wave features WITHOUT derived suffixes
    derived_suffixes = [
        '_deseasonalized', '_detrended', '_processed', '_persistence_', '_trend_', 
        '_change_', '_lag_', '_rolling_mean_', 'anom_', '_abs_change_', '_rel_change_'
    ]
    
    potential_base = []
    for col in df_merged.columns:
        # Must contain wave identifiers
        if any(wave_type in col.lower() for wave_type in ['swh', 'swe']) and col != 'date':
            # Check if it's NOT a derived feature
            is_derived = any(suffix in col for suffix in derived_suffixes)
            if not is_derived:
                # Additional filter: exclude non-base columns
                exclude_patterns = ['event_dummy', 'total_obs', 'port_name', 'year']
                is_excluded = any(pattern in col for pattern in exclude_patterns)
                if not is_excluded:
                    potential_base.append(col)
    
    strategy2_features = list(set(potential_base) - set(base_features))
    base_features.extend(strategy2_features)
    print(f"   Strategy 2 (auto-detect base): {len(strategy2_features)} features")
    if strategy2_features:
        print(f"     Examples: {strategy2_features[:5]}")
    
    # Strategy 3: Look for percentage features (pct_)
    pct_features = [col for col in df_merged.columns if col.startswith('pct_')]
    strategy3_features = list(set(pct_features) - set(base_features))
    base_features.extend(strategy3_features)
    print(f"   Strategy 3 (percentage features): {len(strategy3_features)} features")
    if strategy3_features:
        print(f"     Examples: {strategy3_features[:5]}")
    
    # Strategy 4: Extract root feature from derived feature name
    # For 'anom_swh_p25_persistence_2', try to find 'swh_p25' variations
    if len(base_features) == 0:
        print("   Strategy 4: Attempting to extract base from derived feature names...")
        
        # Get all derived features to understand the pattern
        all_derived = [col for col in df_merged.columns 
                      if any(suffix in col for suffix in derived_suffixes)]
        
        root_candidates = set()
        for derived_col in all_derived:
            # Try to extract the root
            # Remove common prefixes
            root = derived_col
            if root.startswith('anom_'):
                root = root[5:]  # Remove 'anom_'
            
            # Remove suffixes to find the base
            for suffix in derived_suffixes:
                if suffix in root:
                    # For patterns like '_persistence_2', remove the whole thing
                    if '_persistence_' in root:
                        root = root.split('_persistence_')[0]
                    elif '_trend_' in root:
                        root = root.split('_trend_')[0]
                    elif '_change_' in root:
                        root = root.split('_change_')[0]
                    elif '_lag_' in root:
                        root = root.split('_lag_')[0]
                    elif '_rolling_mean_' in root:
                        root = root.split('_rolling_mean_')[0]
                    else:
                        root = root.replace(suffix, '')
                    break
            
            # Clean up the root
            root = root.strip('_')
            if root and any(wave_type in root for wave_type in ['swh', 'swe']):
                root_candidates.add(root)
        
        # Now look for these roots in the actual columns
        strategy4_features = []
        for root in root_candidates:
            # Look for exact matches or close matches
            matches = [col for col in df_merged.columns if col == root]
            if not matches:
                # Look for variations (with _sw, _wa suffixes)
                variations = [f"{root}_sw", f"{root}_wa", f"{root}_mean", f"{root}_max"]
                matches = [col for col in df_merged.columns if col in variations]
            strategy4_features.extend(matches)
        
        strategy4_features = list(set(strategy4_features) - set(base_features))
        base_features.extend(strategy4_features)
        print(f"   Strategy 4 (root extraction): {len(strategy4_features)} features")
        if strategy4_features:
            print(f"     Examples: {strategy4_features[:5]}")
    
    # Remove duplicates and validate
    base_features = list(set(base_features))
    valid_base_features = [f for f in base_features if f in df_merged.columns and f != 'date']
    
    print(f"\n📊 Final base features available for pipeline: {len(valid_base_features)}")
    if len(valid_base_features) > 0:
        for feat in valid_base_features[:15]:  # Show first 15
            print(f"  • {feat}")
        if len(valid_base_features) > 15:
            print(f"  ... and {len(valid_base_features) - 15} more")
    else:
        print("  ❌ NO BASE FEATURES FOUND!")
        print("  📋 Available columns for debugging:")
        all_cols = list(df_merged.columns)
        for i in range(0, len(all_cols), 5):
            print(f"     {all_cols[i:i+5]}")
    
    return valid_base_features


def debug_feature_structure(df_merged, trigger_feature):
    """
    Enhanced debug function to understand your exact data structure
    """
    print(f"\n🔍 DETAILED FEATURE STRUCTURE ANALYSIS")
    print(f"=" * 60)
    print(f"Target trigger feature: {trigger_feature}")
    print(f"Total columns: {len(df_merged.columns)}")
    
    # Categorize all columns
    categories = {
        'date_cols': [],
        'event_cols': [],
        'swh_base': [],
        'swh_deseasonalized': [],
        'swh_detrended': [],
        'swh_persistence': [],
        'swh_trend': [],
        'swh_change': [],
        'swh_lag': [],
        'swh_rolling': [],
        'anom_features': [],
        'pct_features': [],
        'other_features': []
    }
    
    for col in df_merged.columns:
        if col == 'date' or 'date' in col:
            categories['date_cols'].append(col)
        elif 'event' in col:
            categories['event_cols'].append(col)
        elif 'pct_' in col:
            categories['pct_features'].append(col)
        elif 'anom_' in col:
            categories['anom_features'].append(col)
        elif any(wave in col for wave in ['swh', 'swe']):
            if '_deseasonalized' in col and '_detrended' not in col:
                categories['swh_deseasonalized'].append(col)
            elif '_detrended' in col:
                categories['swh_detrended'].append(col)
            elif '_persistence_' in col:
                categories['swh_persistence'].append(col)
            elif '_trend_' in col:
                categories['swh_trend'].append(col)
            elif '_change_' in col:
                categories['swh_change'].append(col)
            elif '_lag_' in col:
                categories['swh_lag'].append(col)
            elif '_rolling_' in col:
                categories['swh_rolling'].append(col)
            else:
                categories['swh_base'].append(col)
        else:
            categories['other_features'].append(col)
    
    # Print categorized results
    for category, features in categories.items():
        if features:
            print(f"\n{category.upper().replace('_', ' ')}: {len(features)} features")
            for feat in features[:10]:  # Show first 10
                print(f"  • {feat}")
            if len(features) > 10:
                print(f"  ... and {len(features) - 10} more")
    
    # Analyze the trigger feature specifically
    print(f"\n🎯 TRIGGER FEATURE ANALYSIS: {trigger_feature}")
    if trigger_feature in df_merged.columns:
        print(f"  ✅ Feature exists in dataset")
        # Try to decompose it
        if trigger_feature.startswith('anom_'):
            base_part = trigger_feature[5:]  # Remove 'anom_'
            print(f"  📊 Base part (after removing 'anom_'): {base_part}")
            
            if '_persistence_' in base_part:
                root_part = base_part.split('_persistence_')[0]
                window_part = base_part.split('_persistence_')[1]
                print(f"  🔍 Root feature: {root_part}")
                print(f"  🔍 Persistence window: {window_part}")
                
                # Look for the root in base features
                possible_roots = [col for col in df_merged.columns if root_part in col and 'anom' not in col and 'persistence' not in col]
                print(f"  🔍 Possible root features found: {possible_roots}")
    else:
        print(f"  ❌ Feature NOT found in dataset")
    
    return categories


# Add this fallback function to the AEP script
def create_fallback_base_features(df_merged, trigger_feature):
    """
    Fallback: if no base features found, try to create minimal base features 
    from the available data for the specific trigger feature
    """
    print(f"\n🚨 FALLBACK: Creating minimal base features for trigger: {trigger_feature}")
    
    # For anom_swh_p25_persistence_2, we need swh_p25 as base
    if trigger_feature.startswith('anom_') and '_persistence_' in trigger_feature:
        # Extract the root feature name
        base_part = trigger_feature[5:]  # Remove 'anom_'
        root_feature = base_part.split('_persistence_')[0]
        
        print(f"  🔍 Need base feature: {root_feature}")
        
        # Look for this feature or similar in the dataset
        candidates = []
        for col in df_merged.columns:
            if root_feature in col and 'anom' not in col and 'persistence' not in col:
                candidates.append(col)
        
        if candidates:
            print(f"  ✅ Found candidate base features: {candidates}")
            return candidates
        else:
            # Try to find any swh features that could work
            swh_features = [col for col in df_merged.columns 
                          if 'swh' in col and 'p25' in col and 'anom' not in col]
            if swh_features:
                print(f"  ⚠️  Using similar SWH features: {swh_features}")
                return swh_features
    
    # Generic fallback - find any base wave features
    fallback_features = []
    for col in df_merged.columns:
        if any(wave in col for wave in ['swh', 'swe']) and col != 'date':
            # Exclude derived features
            derived_suffixes = ['_deseasonalized', '_detrended', '_persistence_', '_trend_', 
                               '_change_', '_lag_', '_rolling_mean_', 'anom_', '_abs_change_', '_rel_change_']
            if not any(suffix in col for suffix in derived_suffixes):
                fallback_features.append(col)
    
    print(f"  🔄 Generic fallback features: {fallback_features[:10]}")
    return fallback_features[:10]  # Limit to 10 features


# Modified version of the extract function with fallback
def extract_base_features_for_pipeline_with_fallback(df_merged, trigger_feature=None):
    """
    Enhanced version with fallback mechanism
    """
    # Try the main extraction
    base_features = extract_base_features_for_pipeline(df_merged)
    
    # If no features found, use fallback
    if len(base_features) == 0 and trigger_feature:
        print(f"\n🚨 No base features found - attempting fallback...")
        base_features = create_fallback_base_features(df_merged, trigger_feature)
    
    return base_features

def apply_simplified_detrend_deseasonalize(sim_df):
    """
    Simplified version of your detrend & deseasonalize pipeline for bootstrap
    """
    df_processed = sim_df.copy()
    df_processed = df_processed.sort_values('date').reset_index(drop=True)
    
    # Get wave features
    wave_features = [col for col in df_processed.columns 
                    if any(wave_type in col for wave_type in ['swh', 'swe']) 
                    and col != 'date']
    
    if len(df_processed) < 30:
        return df_processed
    
    # Create day of year
    df_processed['day_of_year'] = df_processed['date'].dt.dayofyear
    
    # Deseasonalization
    for feature in wave_features:
        if feature in df_processed.columns:
            seasonal_avg = df_processed.groupby('day_of_year')[feature].transform('mean')
            feature_mean = df_processed[feature].mean()
            deseasonalized = df_processed[feature] - seasonal_avg + feature_mean
            df_processed[f"{feature}_deseasonalized"] = deseasonalized
    
    # Simple detrending
    for feature in wave_features:
        deseason_col = f"{feature}_deseasonalized"
        if deseason_col in df_processed.columns:
            try:
                y = df_processed[deseason_col].fillna(df_processed[deseason_col].mean()).values
                x = np.arange(len(y))
                z = np.polyfit(x, y, 1)
                trend = np.polyval(z, x)
                df_processed[f"{feature}_detrended"] = df_processed[deseason_col] - trend
            except:
                df_processed[f"{feature}_detrended"] = df_processed[deseason_col]
    
    return df_processed

def apply_simplified_enhanced_features(df_processed):
    """
    Simplified version of your enhanced features pipeline
    """
    df_enhanced = df_processed.copy()
    
    # Get base features to enhance
    base_features = []
    base_features.extend([col for col in df_enhanced.columns if col.endswith('_detrended')])
    base_features.extend([col for col in df_enhanced.columns 
                         if any(wave_type in col for wave_type in ['swh', 'swe']) 
                         and not col.endswith(('_deseasonalized', '_detrended'))
                         and col != 'date'])
    
    base_features = list(set(base_features))
    valid_base_features = [f for f in base_features if f in df_enhanced.columns]
    
    if len(valid_base_features) == 0:
        return df_enhanced
    
    # Configuration
    PERSISTENCE_WINDOWS = [2, 3, 5, 7, 14]
    TREND_WINDOWS = [3, 5, 7, 14]
    CHANGE_WINDOWS = [3, 5, 7, 14]
    LAG_WINDOWS = [1, 3, 5, 7, 14]
    
    # 1. Persistence features
    for window in PERSISTENCE_WINDOWS:
        for feature in valid_base_features:
            if feature in df_enhanced.columns:
                col_name = f'{feature}_persistence_{window}'
                df_enhanced[col_name] = df_enhanced[feature].rolling(window, min_periods=1).mean()
    
    # 2. Trend features (simplified)
    for window in TREND_WINDOWS:
        for feature in valid_base_features:
            if feature in df_enhanced.columns:
                # Rolling mean
                mean_col = f'{feature}_rolling_mean_{window}'
                df_enhanced[mean_col] = df_enhanced[feature].rolling(window, min_periods=2).mean()
                
                # Simple trend (rolling std as proxy)
                trend_col = f'{feature}_trend_{window}'
                df_enhanced[trend_col] = df_enhanced[feature].rolling(window, min_periods=2).std()
    
    # 3. Change features
    for window in CHANGE_WINDOWS:
        for feature in valid_base_features:
            if feature in df_enhanced.columns:
                # Absolute change
                abs_change_col = f'{feature}_abs_change_{window}'
                df_enhanced[abs_change_col] = df_enhanced[feature] - df_enhanced[feature].shift(window)
                
                # Relative change
                rel_change_col = f'{feature}_rel_change_{window}'
                past_values = df_enhanced[feature].shift(window)
                df_enhanced[rel_change_col] = np.where(
                    past_values != 0,
                    ((df_enhanced[feature] - past_values) / past_values) * 100,
                    0
                )
    
    # 4. Lag features (simplified)
    all_features_to_lag = valid_base_features.copy()
    
    # Add some engineered features
    for feature in valid_base_features:
        for window in PERSISTENCE_WINDOWS[:3]:  # Only first 3 for speed
            col_name = f'{feature}_persistence_{window}'
            if col_name in df_enhanced.columns:
                all_features_to_lag.append(col_name)
    
    for lag in LAG_WINDOWS:
        for feature in all_features_to_lag:
            if feature in df_enhanced.columns:
                lag_col = f'{feature}_lag_{lag}'
                df_enhanced[lag_col] = df_enhanced[feature].shift(lag)
    
    # 5. Create anomaly features (relative to simulation climatology)
    for feature in [f for f in valid_base_features if not f.endswith('_detrended')]:
        if feature in df_enhanced.columns and 'day_of_year' in df_enhanced.columns:
            climatology = df_enhanced.groupby('day_of_year')[feature].transform('mean')
            df_enhanced[f"anom_{feature}"] = df_enhanced[feature] - climatology
    
    return df_enhanced

def create_full_pipeline_simulation(base_data, base_features, simulation_length=365, 
                                   block_length=7, window_days=20):
    """
    Create one bootstrap simulation with full pipeline processing
    """
    # Step 1: Block bootstrap sample
    n_blocks = simulation_length // block_length + 1
    
    sampled_data = []
    for i in range(n_blocks):
        # Seasonal matching
        target_day_of_year = (i * block_length) % 365 + 1
        
        # Find valid blocks
        valid_starts = []
        for idx in range(len(base_data) - block_length):
            data_day_of_year = base_data.iloc[idx]['date'].timetuple().tm_yday
            if abs(data_day_of_year - target_day_of_year) <= window_days // 2:
                valid_starts.append(idx)
        
        if not valid_starts:
            start_idx = np.random.randint(0, len(base_data) - block_length)
        else:
            start_idx = np.random.choice(valid_starts)
        
        # Extract block
        block = base_data.iloc[start_idx:start_idx + block_length][base_features + ['date']]
        sampled_data.append(block)
        
        if len(sampled_data) * block_length >= simulation_length:
            break
    
    # Concatenate and assign new dates
    sim_base_df = pd.concat(sampled_data).head(simulation_length).reset_index(drop=True)
    sim_dates = pd.date_range('2025-01-01', periods=simulation_length)
    sim_base_df['date'] = sim_dates
    
    # Step 2: Apply pipeline
    sim_processed = apply_simplified_detrend_deseasonalize(sim_base_df)
    sim_enhanced = apply_simplified_enhanced_features(sim_processed)
    
    return sim_enhanced

def process_simulation_batch_full_pipeline(batch_indices, base_data, base_features, 
                                          trigger_feature, trigger_threshold,
                                          observed_matrix, N, W, min_days, has_observed,
                                          block_length, window_days):
    """
    Process a batch of simulations using full pipeline
    """
    batch_losses = []
    batch_fp_costs = []
    batch_fn_costs = []
    batch_tp_costs = []
    batch_event_counts = []
    
    for sim_idx in range(len(batch_indices)):
        try:
            # Create simulation with full pipeline
            sim_df = create_full_pipeline_simulation(
                base_data, base_features, simulation_length=365,
                block_length=block_length, window_days=window_days
            )
            
            # Check if trigger feature was created
            if trigger_feature not in sim_df.columns:
                batch_losses.append(0)
                if has_observed:
                    batch_fp_costs.append(0)
                    batch_fn_costs.append(0)
                    batch_tp_costs.append(0)
                    batch_event_counts.append({'fp_events': 0, 'fn_events': 0, 'tp_events': 0, 'tn_events': 0, 'n_events': 0})
                else:
                    batch_event_counts.append({'fp_events': 0, 'fn_events': 0, 'tp_events': 0, 'tn_events': 0, 'n_events': 0})
                continue
            
            # Apply trigger rule
            trigger_vals = sim_df[trigger_feature].fillna(0).values
            predicted_events = (trigger_vals > trigger_threshold).astype(np.int32)
            
            # Calculate loss
            total_loss = calculate_annual_loss_jit(predicted_events, N, W, min_days)
            batch_losses.append(total_loss)
            
            n_events = np.sum(predicted_events)
            
            if has_observed:
                # Use original observed data (this is a limitation - we can't perfectly align)
                # In practice, you might want to sample observed events too
                if sim_idx < len(observed_matrix):
                    observed_vals = observed_matrix[batch_indices[sim_idx]]
                    observed_vals = observed_vals[:len(predicted_events)]
                    if len(observed_vals) < len(predicted_events):
                        observed_vals = np.pad(observed_vals, (0, len(predicted_events) - len(observed_vals)), 'constant')
                    
                    fp_cost, fn_cost, tp_cost = calculate_cm_costs_jit(
                        predicted_events, observed_vals, N, W, min_days
                    )
                    batch_fp_costs.append(fp_cost)
                    batch_fn_costs.append(fn_cost)
                    batch_tp_costs.append(tp_cost)
                    
                    # Count events
                    from numba.typed import List
                    fp_events = len(find_events_in_category_jit(predicted_events, observed_vals, 0))
                    fn_events = len(find_events_in_category_jit(predicted_events, observed_vals, 1))
                    tp_events = len(find_events_in_category_jit(predicted_events, observed_vals, 2))
                    batch_event_counts.append({
                        'fp_events': fp_events, 'fn_events': fn_events, 
                        'tp_events': tp_events, 'n_events': n_events, 'tn_events': 0
                    })
                else:
                    batch_fp_costs.append(0)
                    batch_fn_costs.append(0)
                    batch_tp_costs.append(0)
                    batch_event_counts.append({'fp_events': 0, 'fn_events': 0, 'tp_events': 0, 'tn_events': 0, 'n_events': n_events})
            else:
                batch_event_counts.append({'fp_events': 0, 'fn_events': 0, 'tp_events': 0, 'tn_events': 0, 'n_events': n_events})
                
        except Exception as e:
            # Handle failures gracefully
            batch_losses.append(0)
            if has_observed:
                batch_fp_costs.append(0)
                batch_fn_costs.append(0)
                batch_tp_costs.append(0)
                batch_event_counts.append({'fp_events': 0, 'fn_events': 0, 'tp_events': 0, 'tn_events': 0, 'n_events': 0})
            else:
                batch_event_counts.append({'fp_events': 0, 'fn_events': 0, 'tp_events': 0, 'tn_events': 0, 'n_events': 0})
    
    return batch_losses, batch_fp_costs, batch_fn_costs, batch_tp_costs, batch_event_counts

def calculate_enhanced_aep_analysis(swh_data, trigger_feature, trigger_threshold, N, W, 
                                   min_days=None, n_simulations=None, observed_events=None, 
                                   block_length=None, window_days=None, n_jobs=-1):
    """
    Enhanced AEP analysis that automatically chooses between fast and full pipeline
    """
    print(f"🚀 ENHANCED AEP ANALYSIS WITH SMART PIPELINE SELECTION")
    print("=" * 60)
    print(f"  Data: {len(swh_data)} observations")
    print(f"  Trigger: {trigger_feature} > {trigger_threshold}")
    print(f"  Port: {N} fishermen × ${W}/day")
    print(f"  Min event: {min_days} days")
    print(f"  Block length: {block_length} days")
    print(f"  Simulations: {n_simulations}")
    
    # Step 1: Determine if we need full pipeline
    is_derived = identify_if_derived_feature(trigger_feature)
    
    if not is_derived:
        # Use your existing fast approach
        print("🚀 Using FAST direct sampling approach...")
        return calculate_unified_aep_analysis_fast(
            swh_data, trigger_feature, trigger_threshold, N, W,
            min_days, n_simulations, observed_events, 
            block_length, window_days, n_jobs
        )
    
    # Use full pipeline approach
    print("🔧 Using FULL PIPELINE bootstrap approach...")
    
    # Prepare data
    daily_clean = swh_data.copy().sort_index()
    if len(daily_clean) == 0:
        print("ERROR: No valid data!")
        return None
    
    # Extract base features
    base_features = extract_base_features_for_pipeline(daily_clean)
    if len(base_features) == 0:
        print("ERROR: No base features found for pipeline!")
        return None
    
    # Prepare base data for sampling
    base_data = daily_clean[base_features + ['date']].dropna()
    base_data = base_data.sort_values('date').reset_index(drop=True)
    
    print(f"  Base data for sampling: {len(base_data)} days")
    
    # Setup observed events
    has_observed = observed_events is not None
    if has_observed:
        observed_aligned = observed_events.reindex(daily_clean.index, fill_value=0)
        print(f"  Observed events: {observed_aligned.sum()} out of {len(observed_aligned)} days")
        observed_matrix = observed_aligned.values.astype(np.int32)
    else:
        print("  No observed events provided")
        observed_matrix = None
    
    # Setup parallel processing
    if n_jobs == -1:
        import multiprocessing as mp
        n_jobs = mp.cpu_count()
    n_jobs = min(n_jobs, n_simulations)
    
    print(f"  Processing {n_simulations} simulations using {n_jobs} workers...")
    
    # Create batches
    batch_size = max(1, n_simulations // n_jobs)
    simulation_batches = []
    for i in range(0, n_simulations, batch_size):
        end_idx = min(i + batch_size, n_simulations)
        batch_indices = list(range(i, end_idx))
        simulation_batches.append(batch_indices)
    
    # Process batches
    all_losses = []
    all_fp_costs = []
    all_fn_costs = []
    all_tp_costs = []
    all_event_counts = []
    
    if n_jobs == 1:
        print("  Using single-threaded processing...")
        for batch_indices in tqdm(simulation_batches, desc="Processing batches"):
            batch_losses, batch_fp, batch_fn, batch_tp, batch_events = process_simulation_batch_full_pipeline(
                batch_indices, base_data, base_features, trigger_feature, trigger_threshold,
                observed_matrix, N, W, min_days, has_observed, block_length, window_days
            )
            all_losses.extend(batch_losses)
            if has_observed:
                all_fp_costs.extend(batch_fp)
                all_fn_costs.extend(batch_fn)
                all_tp_costs.extend(batch_tp)
                all_event_counts.extend(batch_events)
    else:
        print(f"  Using {n_jobs} threads for parallel processing...")
        def process_batch_wrapper(batch_indices):
            return process_simulation_batch_full_pipeline(
                batch_indices, base_data, base_features, trigger_feature, trigger_threshold,
                observed_matrix, N, W, min_days, has_observed, block_length, window_days
            )
        
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            results = list(tqdm(
                executor.map(process_batch_wrapper, simulation_batches),
                total=len(simulation_batches),
                desc="Processing batches"
            ))
        
        for batch_losses, batch_fp, batch_fn, batch_tp, batch_events in results:
            all_losses.extend(batch_losses)
            if has_observed:
                all_fp_costs.extend(batch_fp)
                all_fn_costs.extend(batch_fn)
                all_tp_costs.extend(batch_tp)
                all_event_counts.extend(batch_events)
    
    # Process results (same as your existing code)
    annual_losses = np.array(all_losses)
    print(f"  Completed {len(annual_losses)} simulations successfully.")
    
    # Calculate AEP curve
    print("  Calculating AEP curve...")
    standard_aep_curve = calculate_aep_curve_efficient(
        annual_losses, trigger_feature, trigger_threshold, min_days, N, W
    )
    
    standard_summary = {
        'mean_loss': float(np.mean(annual_losses)),
        'std_loss': float(np.std(annual_losses)),
        'max_loss': float(np.max(annual_losses)),
        'zero_prob': float(np.mean(annual_losses == 0)),
        'method': 'enhanced_full_pipeline_bootstrap' if is_derived else 'fast_direct_sampling',
        'block_length': block_length,
        'window_days': window_days,
        'trigger_feature': trigger_feature,
        'trigger_threshold': trigger_threshold,
        'min_days': min_days,
        'n_fishermen': N,
        'daily_wage': W,
        'n_simulations': len(annual_losses),
        'n_jobs': n_jobs,
        'is_derived_feature': is_derived
    }
    
    print(f"\n📊 Enhanced AEP Results:")
    print(f"  Method: {standard_summary['method']}")
    print(f"  Mean annual loss: ${standard_summary['mean_loss']:,.0f}")
    print(f"  Max annual loss: ${standard_summary['max_loss']:,.0f}")
    print(f"  Zero loss probability: {standard_summary['zero_prob']:.1%}")
    
    # Confusion matrix processing (same as your existing code)
    confusion_matrix_results = None
    if has_observed and all_fp_costs:
        print("  Calculating confusion matrix AEP curves...")
        fp_annual_costs = np.array(all_fp_costs)
        fn_annual_costs = np.array(all_fn_costs)
        tp_annual_costs = np.array(all_tp_costs)
        
        fp_aep = calculate_aep_curve_cm(fp_annual_costs, 'FP_Cost', trigger_feature, trigger_threshold, min_days, N, W)
        fn_aep = calculate_aep_curve_cm(fn_annual_costs, 'FN_Cost', trigger_feature, trigger_threshold, min_days, N, W)
        tp_aep = calculate_aep_curve_cm(tp_annual_costs, 'TP_Cost', trigger_feature, trigger_threshold, min_days, N, W)
        
        insurance_annual_costs = fp_annual_costs + tp_annual_costs
        insurance_aep = calculate_aep_curve_cm(insurance_annual_costs, 'Insurance_Cost', trigger_feature, trigger_threshold, min_days, N, W)
        
        cm_summary = {
            'fp_costs': {
                'mean': float(np.mean(fp_annual_costs)),
                'std': float(np.std(fp_annual_costs)),
                'max': float(np.max(fp_annual_costs)),
                'zero_prob': float(np.mean(fp_annual_costs == 0))
            },
            'fn_costs': {
                'mean': float(np.mean(fn_annual_costs)),
                'std': float(np.std(fn_annual_costs)),
                'max': float(np.max(fn_annual_costs)),
                'zero_prob': float(np.mean(fn_annual_costs == 0))
            },
            'tp_costs': {
                'mean': float(np.mean(tp_annual_costs)),
                'std': float(np.std(tp_annual_costs)),
                'max': float(np.max(tp_annual_costs)),
                'zero_prob': float(np.mean(tp_annual_costs == 0))
            },
            'insurance_costs': {
                'mean': float(np.mean(insurance_annual_costs)),
                'std': float(np.std(insurance_annual_costs)),
                'max': float(np.max(insurance_annual_costs)),
                'zero_prob': float(np.mean(insurance_annual_costs == 0))
            }
        }
        
        confusion_matrix_results = {
            'fp_aep': fp_aep,
            'fn_aep': fn_aep,
            'tp_aep': tp_aep,
            'insurance_aep': insurance_aep,
            'fp_annual_costs': fp_annual_costs,
            'fn_annual_costs': fn_annual_costs,
            'tp_annual_costs': tp_annual_costs,
            'insurance_annual_costs': insurance_annual_costs,
            'summary': cm_summary,
            'all_event_counts': all_event_counts if all_event_counts else []
        }
    
    unified_results = {
        'standard_summary': standard_summary,
        'standard_aep_curve': standard_aep_curve,
        'confusion_matrix_results': confusion_matrix_results
    }
    
    return unified_results

def parse_multi_rule(rule_str):
    """
    Parse multi-condition rules of the form:
    - 'feature1 > t1 AND feature2 > t2'
    - 'feature1 > t1 OR feature2 > t2'
    - 'feature1 > t1 AND feature2 > t2 AND feature3 > t3'
    """
    import re
    
    rule_str = rule_str.strip()
    parts = re.split(r'\s+(AND|OR)\s+', rule_str)
    
    features = []
    operators = []
    logic_ops = []
    
    for i, part in enumerate(parts):
        if part in ['AND', 'OR']:
            logic_ops.append(part)
        else:
            if ' > ' in part:
                feature = part.split(' > ')[0].strip()
                operator = '>'
            elif ' < ' in part:
                feature = part.split(' < ')[0].strip()
                operator = '<'
            elif ' >= ' in part:
                feature = part.split(' >= ')[0].strip()
                operator = '>='
            elif ' <= ' in part:
                feature = part.split(' <= ')[0].strip()
                operator = '<='
            else:
                raise ValueError(f"Cannot parse condition: {part}")
            
            features.append(feature)
            operators.append(operator)
    
    return features, operators, logic_ops

@jit(nopython=True, cache=True)
def apply_multi_rule_jit(feature_values_matrix, thresholds, operators_encoded, logic_ops_encoded):
    """Fast JIT-compiled multi-rule evaluation"""
    n_days = feature_values_matrix.shape[0]
    n_features = len(thresholds)
    
    # Evaluate individual conditions
    conditions = np.zeros((n_days, n_features), dtype=np.bool_)
    
    for f_idx in range(n_features):
        for day_idx in range(n_days):
            value = feature_values_matrix[day_idx, f_idx]
            threshold = thresholds[f_idx]
            op = operators_encoded[f_idx]
            
            if op == 0:  # >
                conditions[day_idx, f_idx] = value > threshold
            elif op == 1:  # <
                conditions[day_idx, f_idx] = value < threshold
            elif op == 2:  # >=
                conditions[day_idx, f_idx] = value >= threshold
            elif op == 3:  # <=
                conditions[day_idx, f_idx] = value <= threshold
    
    # Combine conditions with logic operators
    result = np.zeros(n_days, dtype=np.bool_)
    
    for day_idx in range(n_days):
        if n_features == 1:
            result[day_idx] = conditions[day_idx, 0]
        else:
            current_result = conditions[day_idx, 0]
            
            for logic_idx in range(len(logic_ops_encoded)):
                next_condition = conditions[day_idx, logic_idx + 1]
                
                if logic_ops_encoded[logic_idx] == 0:  # AND
                    current_result = current_result and next_condition
                else:  # OR
                    current_result = current_result or next_condition
            
            result[day_idx] = current_result
    
    return result.astype(np.int32)

def evaluate_multi_rule_fast(df, features, operators, logic_ops, thresholds):
    """Fast evaluation of multi-rule combinations"""
    op_mapping = {'>': 0, '<': 1, '>=': 2, '<=': 3}
    logic_mapping = {'AND': 0, 'OR': 1}
    
    operators_encoded = np.array([op_mapping[op] for op in operators])
    logic_ops_encoded = np.array([logic_mapping[op] for op in logic_ops])
    
    # Extract feature values
    feature_values = []
    for feature in features:
        if feature in df.columns:
            values = df[feature].fillna(0).values
        else:
            print(f"Warning: Feature {feature} not found, using zeros")
            values = np.zeros(len(df))
        feature_values.append(values)
    
    feature_values_matrix = np.column_stack(feature_values).astype(np.float64)
    thresholds_array = np.array(thresholds, dtype=np.float64)
    
    predictions = apply_multi_rule_jit(
        feature_values_matrix, thresholds_array, 
        operators_encoded, logic_ops_encoded
    )
    
    return predictions

def generate_top_single_rules(cv_results_path, top_k=10):
    """Extract top K single rules from CV results"""
    cv_df = pd.read_csv(cv_results_path)
    single_rules = cv_df[cv_df['rule_type'] == 'single'].copy()
    
    if single_rules.empty:
        print("No single rules found in CV results")
        return []
    
    single_rules = single_rules.sort_values(
        ['f1_mean', 'precision_mean', 'recall_mean'], 
        ascending=False
    )
    
    top_rules = []
    for _, row in single_rules.head(top_k).iterrows():
        rule_name = row['rule_name']
        feature = rule_name.replace(' > threshold', '').replace('Single: ', '').strip()
        top_rules.append({
            'feature': feature,
            'f1_score': row['f1_mean'],
            'precision': row['precision_mean'],
            'recall': row['recall_mean']
        })
    
    return top_rules

def generate_double_rule_combinations(top_rules, max_combinations=50):
    """Generate double rule combinations from top single rules"""
    combinations = []
    
    for i, rule1 in enumerate(top_rules):
        for j, rule2 in enumerate(top_rules[i+1:], i+1):
            # AND combination
            combinations.append({
                'features': [rule1['feature'], rule2['feature']],
                'operators': ['>', '>'],
                'logic_ops': ['AND'],
                'description': f"{rule1['feature']} > t1 AND {rule2['feature']} > t2",
                'type': 'double_AND'
            })
            
            # OR combination
            combinations.append({
                'features': [rule1['feature'], rule2['feature']],
                'operators': ['>', '>'],
                'logic_ops': ['OR'],
                'description': f"{rule1['feature']} > t1 OR {rule2['feature']} > t2",
                'type': 'double_OR'
            })
            
            if len(combinations) >= max_combinations:
                break
        
        if len(combinations) >= max_combinations:
            break
    
    return combinations[:max_combinations]

def optimize_thresholds_fast(df, rule_combination, observed_events, 
                           threshold_grid_size=5, metric='f1'):
    """Fast threshold optimization for multi-rule combinations"""
    features = rule_combination['features']
    operators = rule_combination['operators']
    logic_ops = rule_combination['logic_ops']
    
    # Create threshold grids for each feature
    threshold_grids = []
    for feature in features:
        if feature in df.columns:
            values = df[feature].dropna()
            percentiles = np.linspace(5, 95, threshold_grid_size)
            thresholds = np.percentile(values, percentiles)
            threshold_grids.append(thresholds)
        else:
            threshold_grids.append(np.array([0.0]))
    
    # Generate all threshold combinations
    all_combinations = list(itertools.product(*threshold_grids))
    
    if len(all_combinations) > 200:  # Limit combinations for speed
        np.random.seed(42)
        selected_indices = np.random.choice(len(all_combinations), 200, replace=False)
        all_combinations = [all_combinations[i] for i in selected_indices]
    
    best_score = -1
    best_thresholds = None
    
    # Prepare observed events
    observed_aligned = observed_events.reindex(df.set_index('date').index, fill_value=0).values
    
    for thresholds in all_combinations:
        # Evaluate rule
        predictions = evaluate_multi_rule_fast(
            df, features, operators, logic_ops, thresholds
        )
        
        # Calculate metric
        tp = np.sum((predictions == 1) & (observed_aligned == 1))
        fp = np.sum((predictions == 1) & (observed_aligned == 0))
        fn = np.sum((predictions == 0) & (observed_aligned == 1))
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
            
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0
            
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        
        if metric == 'f1':
            score = f1
        elif metric == 'precision':
            score = precision
        elif metric == 'recall':
            score = recall
        else:
            score = f1
        
        if score > best_score:
            best_score = score
            best_thresholds = thresholds
    
    return best_thresholds, best_score

@jit(nopython=True, cache=True)
def generate_bootstrap_indices_simple(n_days, n_simulations, block_length):
    """Simple bootstrap index generation"""
    simulation_indices = np.zeros((n_simulations, 365), dtype=np.int32)
    
    for sim in range(n_simulations):
        np.random.seed(sim)
        sim_indices = []
        current_day = 0
        
        while current_day < 365:
            start_idx = np.random.randint(0, max(1, n_days - block_length))
            block_indices = np.arange(start_idx, min(start_idx + block_length, n_days))
            
            for idx in block_indices:
                if len(sim_indices) < 365:
                    sim_indices.append(idx)
            
            current_day += block_length
        
        for i in range(min(len(sim_indices), 365)):
            simulation_indices[sim, i] = sim_indices[i]
    
    return simulation_indices

def calculate_fast_multi_rule_aep(df, rule_features, rule_operators, rule_logic_ops, 
                                  thresholds, N, W, min_days, n_simulations=config.N_SIMULATIONS):
    """Fast AEP analysis for multi-rule combinations"""
    print(f"  🚀 Fast multi-rule AEP: {len(rule_features)} features, {n_simulations} sims")
    
    # Prepare data
    df_clean = df.copy().sort_values('date').reset_index(drop=True)
    
    # Check if all features exist
    missing_features = [f for f in rule_features if f not in df_clean.columns]
    if missing_features:
        print(f"    ❌ Missing features: {missing_features}")
        return None
    
    # Extract feature matrices for simulation
    feature_matrices = {}
    for feature in rule_features:
        feature_matrices[feature] = df_clean[feature].fillna(0).values.astype(np.float32)
    
    # Generate simulation indices
    simulation_indices = generate_bootstrap_indices_simple(
        len(df_clean), n_simulations, block_length=7
    )
    
    # Run simulations
    annual_losses = []
    for sim_idx in range(n_simulations):
        try:
            # Sample feature values
            sampled_features = {}
            for feature in rule_features:
                sampled_features[feature] = feature_matrices[feature][simulation_indices[sim_idx]]
            
            # Create temp dataframe
            sim_df = pd.DataFrame(sampled_features)
            
            # Evaluate rule
            predictions = evaluate_multi_rule_fast(
                sim_df, rule_features, rule_operators, rule_logic_ops, thresholds
            )
            
            # Calculate loss
            total_loss = calculate_annual_loss_jit(predictions, N, W, min_days)
            annual_losses.append(total_loss)
            
        except Exception:
            annual_losses.append(0)
    
    # Calculate AEP curve
    annual_losses = np.array(annual_losses)
    losses_sorted = np.sort(annual_losses)[::-1]
    exceedance_prob = np.arange(1, len(losses_sorted)+1) / (len(losses_sorted)+1)
    
    aep_curve = pd.DataFrame({
        'loss': losses_sorted,
        'probability': exceedance_prob
    })
    
    # Summary
    summary = {
        'mean_loss': float(np.mean(annual_losses)),
        'std_loss': float(np.std(annual_losses)),
        'max_loss': float(np.max(annual_losses)),
        'zero_prob': float(np.mean(annual_losses == 0)),
        'method': 'fast_multi_rule_bootstrap',
        'rule_features': rule_features,
        'thresholds': thresholds,
        'min_days': min_days,
        'n_fishermen': N,
        'daily_wage': W,
        'n_simulations': len(annual_losses)
    }
    
    return {
        'summary': summary,
        'aep_curve': aep_curve,
        'annual_losses': annual_losses
    }

# STEP 3: ADD THE MAIN ANALYSIS FUNCTION
# Add this function to your script:

def fast_multi_rule_main():
    """
    Fast multi-rule AEP analysis - replacement for enhanced_main()
    """
    print("\n🚀 FAST MULTI-RULE AEP ANALYSIS")
    print(f"Run: {config.RUN_PATH}")
    print(f"Reference port: {config.reference_port}")

    # Load input data (same as your existing code)
    merged_path = os.path.join(config.run_output_dir, 'df_swan_waverys_merged.csv')
    if not os.path.exists(merged_path):
        print(f"❌ Input file not found: {merged_path}")
        return
    df = pd.read_csv(merged_path, parse_dates=['date'])

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = config.results_output_dir

    # Locate CV results
    cv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv') and 'cv_results' in f]
    if not cv_files:
        print(f"❌ No CV results found in {results_dir}")
        return
    cv_results_path = os.path.join(results_dir, sorted(cv_files)[-1])

    # Setup observed events
    observed_col = 'event_dummy_1' if 'event_dummy_1' in df.columns else None
    observed_events = df.set_index('date')[observed_col] if observed_col else None

    # Get top rules
    print("\n📋 Generating rule combinations...")
    top_rules = generate_top_single_rules(cv_results_path, top_k=6)
    if len(top_rules) == 0:
        print("❌ No single rules found")
        return
    
    print(f"  Top features: {[r['feature'] for r in top_rules[:3]]}")
    
    # Generate combinations
    double_combinations = generate_double_rule_combinations(top_rules, max_combinations=15)
    
    print(f"  Testing {len(double_combinations)} double rule combinations...")

    # Analyze combinations
    results = []
    for i, combination in enumerate(double_combinations):
        print(f"\n--- {i+1}/{len(double_combinations)}: {combination['type']} ---")
        
        try:
            # Optimize thresholds
            if observed_events is not None:
                best_thresholds, best_score = optimize_thresholds_fast(
                    df, combination, observed_events, threshold_grid_size=4
                )
                print(f"    F1: {best_score:.3f}")
            else:
                best_thresholds = []
                for feature in combination['features']:
                    if feature in df.columns:
                        best_thresholds.append(np.percentile(df[feature].dropna(), 75))
                    else:
                        best_thresholds.append(0.0)
                best_score = 0.0
            
            # Run AEP analysis
            aep_results = calculate_fast_multi_rule_aep(
                df, combination['features'], combination['operators'], 
                combination['logic_ops'], best_thresholds, N_PARAM, W_PARAM, 
                config.MIN_DAYS, n_simulations=config.N_SIMULATIONS
            )
            
            if aep_results is not None:
                result = {
                    'combination_id': i,
                    'type': combination['type'],
                    'description': combination['description'],
                    'features': combination['features'],
                    'thresholds': best_thresholds,
                    'f1_score': best_score,
                    'mean_loss': aep_results['summary']['mean_loss'],
                    'max_loss': aep_results['summary']['max_loss'],
                    'zero_prob': aep_results['summary']['zero_prob']
                }
                results.append(result)
                print(f"    ✅ Mean loss: ${result['mean_loss']:,.0f}")
            else:
                print(f"    ❌ Failed")
                
        except Exception as e:
            print(f"    ❌ Error: {e}")

    # Save and display results
    if results:
        results_df = pd.DataFrame(results).sort_values('mean_loss')
        
        # Save summary
        summary_path = os.path.join(results_dir, f'fast_multi_rule_summary_{timestamp}.csv')
        results_df.to_csv(summary_path, index=False)
        
        print(f"\n🏆 TOP 5 MULTI-RULE COMBINATIONS:")
        print("=" * 70)
        for i, (_, row) in enumerate(results_df.head(5).iterrows(), 1):
            print(f"{i}. {row['type']}: {row['description']}")
            print(f"   F1: {row['f1_score']:.3f} | Mean Loss: ${row['mean_loss']:,.0f}")
            print()
        
        print(f"✅ Results saved: {summary_path}")
        return results_df
    else:
        print("❌ No successful analyses")
        return None

# ============================================================================
# INTEGRATION WITH YOUR EXISTING MAIN FUNCTION
# ============================================================================

def enhanced_main():
    """
    Enhanced version of your main() function using smart pipeline selection
    """
    print("\n🔧 ENHANCED AEP_CALCULATION.PY - Smart Pipeline Selection")
    print(f"Run: {config.RUN_PATH}")
    print(f"Reference port: {config.reference_port}")

    # Load input data (same as your existing code)
    merged_path = os.path.join(config.run_output_dir, 'df_swan_waverys_merged.csv')
    if not os.path.exists(merged_path):
        print(f"❌ Input file not found: {merged_path}")
        return
    df = pd.read_csv(merged_path, parse_dates=['date'])

    # Set timestamp and results_dir
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = config.results_output_dir

    # Locate CV results (same as your existing code)
    cv_files = [f for f in os.listdir(results_dir) if (f == 'rule_cv_results.csv' or (f.startswith('cv_results_') and f.endswith('.csv')))]
    if not cv_files:
        print(f"❌ No CV results found in {results_dir}")
        return
    latest_cv = sorted(cv_files)[-1]
    cv_results_path = os.path.join(results_dir, latest_cv)

    # Select best rule (same as your existing code)
    feature, fold_thresholds = load_best_single_rule_and_all_thresholds(cv_results_path, results_dir)
    print(f"✅ Best single rule: {feature}")
    print(f"Fold thresholds: {fold_thresholds}")

    # Compute candidate thresholds (same as your existing code)
    candidate_thresholds = []
    candidate_labels = []
    for i, t in enumerate(fold_thresholds):
        candidate_thresholds.append(t)
        candidate_labels.append(f'fold{i}')
    mean_thr = np.mean(fold_thresholds)
    median_thr = np.median(fold_thresholds)
    p75_thr = np.percentile(fold_thresholds, 75)
    candidate_thresholds.extend([mean_thr, median_thr, p75_thr])
    candidate_labels.extend(['mean', 'median', 'p75'])

    # Run enhanced AEP simulation
    print("\n🚀 Running ENHANCED AEP simulation with smart pipeline selection...")
    observed_col = 'event_dummy_1' if 'event_dummy_1' in df.columns else None
    observed_events = df.set_index('date')[observed_col] if observed_col else None
    # swh_data = df.set_index('date')
    swh_data = df  # Keep date as column - analysis functions handle indexing internally

    summary_rows = []
    for thr, label in zip(candidate_thresholds, candidate_labels):
        print(f"\n--- Threshold ({label}): {thr:.5f} ---")
        
        # Use enhanced AEP analysis (automatically selects approach)
        aep_results = calculate_enhanced_aep_analysis(
            swh_data,
            trigger_feature=feature,
            trigger_threshold=thr,
            N=N_PARAM,
            W=W_PARAM,
            min_days=config.MIN_DAYS,
            n_simulations=min(config.N_SIMULATIONS, 1000),  # Start smaller for testing
            observed_events=observed_events,
            block_length=config.BLOCK_LENGTH,
            window_days=config.WINDOW_DAYS,
            n_jobs=-1
        )
        
        if aep_results is None:
            print(f"❌ AEP simulation failed for threshold {thr:.5f} ({label})")
            continue
        
        # Print summary
        mean_loss = aep_results['standard_summary']['mean_loss']
        method = aep_results['standard_summary']['method']
        print(f"Enhanced method ({label}) using {method}: mean annual loss = ${mean_loss:,.0f}")
        
        # Save results (same structure as your existing code)
        summary_path = os.path.join(results_dir, f'enhanced_aep_summary_{label}_{timestamp}.csv')
        aep_curve_path = os.path.join(results_dir, f'enhanced_aep_curve_{label}_{timestamp}.csv')
        pd.DataFrame([aep_results['standard_summary']]).to_csv(summary_path, index=False)
        pd.DataFrame(aep_results['standard_aep_curve']).to_csv(aep_curve_path, index=False)
        
        # Store for comparison
        row = dict(threshold_label=label, threshold_value=thr)
        row['W_param'] = W_PARAM
        row['N_param'] = N_PARAM
        row['min_days'] = config.MIN_DAYS
        row.update(aep_results['standard_summary'])
        summary_rows.append(row)

    # Save combined summary
    if summary_rows:
        all_summary_path = os.path.join(results_dir, f'enhanced_aep_multi_threshold_summary_{timestamp}.csv')
        pd.DataFrame(summary_rows).to_csv(all_summary_path, index=False)
        print(f"\n✅ Enhanced AEP analysis complete!")
        print(f"📊 Results saved with timestamp: {timestamp}")
        
        # Print comparison
        comparison_df = pd.DataFrame(summary_rows).sort_values('mean_loss')
        print(f"\n🏆 ENHANCED RESULTS COMPARISON:")
        print("=" * 80)
        
        for i, (_, row) in enumerate(comparison_df.head(5).iterrows(), 1):
            method_type = "🔧 FULL PIPELINE" if row.get('is_derived_feature', False) else "⚡ FAST DIRECT"
            print(f"{i}. {row['threshold_label']} | {method_type}")
            print(f"   Mean Loss: ${row['mean_loss']:,.0f}")
            print(f"   Method: {row['method']}")
            print(f"   Zero Loss Prob: {row['zero_prob']:.1%}")
            print()
        
        return comparison_df
    
    else:
        print("❌ No successful enhanced AEP analyses")
        return None
#-----------------------------------------------

# ENHANCED ANALYSIS AND NEXT STEPS
# =================================

def analyze_top_combinations_detailed(results_df, df, observed_events, timestamp, results_dir):
    """
    Detailed analysis of your top combinations with specific threshold values
    """
    print("\n📊 DETAILED ANALYSIS OF TOP COMBINATIONS")
    print("=" * 60)
    
    # Get the top 3 combinations for detailed analysis
    top_3 = results_df.head(3)
    
    detailed_results = []
    
    for i, (_, row) in enumerate(top_3.iterrows(), 1):
        print(f"\n🔍 DETAILED ANALYSIS #{i}: {row['type']}")
        print(f"Features: {row['features']}")
        print(f"Thresholds: {[f'{t:.4f}' for t in row['thresholds']]}")
        print(f"F1 Score: {row['f1_score']:.3f}")
        print(f"Mean Annual Loss: ${row['mean_loss']:,.0f}")
        
        # Get the specific thresholds for this combination
        features = row['features']
        thresholds = row['thresholds']
        
        # Evaluate the rule on actual data to get detailed stats
        predictions = evaluate_multi_rule_fast(
            df, features, ['>', '>'], ['AND'], thresholds
        )
        
        # Calculate detailed performance metrics
        if observed_events is not None:
            observed_aligned = observed_events.reindex(df.set_index('date').index, fill_value=0).values
            
            tp = np.sum((predictions == 1) & (observed_aligned == 1))
            fp = np.sum((predictions == 1) & (observed_aligned == 0))
            tn = np.sum((predictions == 0) & (observed_aligned == 0))
            fn = np.sum((predictions == 0) & (observed_aligned == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            print(f"  Detailed Performance:")
            print(f"    True Positives: {tp}")
            print(f"    False Positives: {fp}")
            print(f"    True Negatives: {tn}")
            print(f"    False Negatives: {fn}")
            print(f"    Precision: {precision:.3f}")
            print(f"    Recall: {recall:.3f}")
            print(f"    Specificity: {specificity:.3f}")
        
        # Prediction statistics
        total_predictions = np.sum(predictions)
        total_days = len(predictions)
        prediction_rate = total_predictions / total_days
        
        print(f"  Prediction Statistics:")
        print(f"    Total predicted events: {total_predictions} days")
        print(f"    Prediction rate: {prediction_rate:.1%}")
        print(f"    Average event frequency: {365 * prediction_rate:.1f} days/year")
        
        # Feature threshold interpretation
        print(f"  Rule Interpretation:")
        for j, (feature, threshold) in enumerate(zip(features, thresholds)):
            if feature in df.columns:
                feature_stats = df[feature].describe()
                percentile = (df[feature] <= threshold).mean() * 100
                print(f"    {feature} > {threshold:.4f}")
                print(f"      (>{percentile:.1f}th percentile, mean={feature_stats['mean']:.4f})")
        
        detailed_results.append({
            'rank': i,
            'features': features,
            'thresholds': thresholds,
            'f1_score': row['f1_score'],
            'mean_loss': row['mean_loss'],
            'prediction_rate': prediction_rate,
            'total_predictions': total_predictions
        })
    
    # Save detailed analysis
    detailed_df = pd.DataFrame(detailed_results)
    detailed_path = os.path.join(results_dir, f'detailed_top_combinations_{timestamp}.csv')
    detailed_df.to_csv(detailed_path, index=False)
    print(f"\n✅ Saved detailed analysis: {detailed_path}")
    
    return detailed_results


def create_comparison_with_baseline(results_df, df, cv_results_path, observed_events, 
                                   timestamp, results_dir):
    """
    Compare multi-rule results with the best single rule baseline
    """
    print("\n📈 COMPARISON WITH SINGLE RULE BASELINE")
    print("=" * 50)
    
    # Get the best single rule for comparison
    try:
        feature, fold_thresholds = load_best_single_rule_and_all_thresholds(cv_results_path, results_dir)
        mean_threshold = np.mean(fold_thresholds)
        
        print(f"Best single rule: {feature} > {mean_threshold:.4f}")
        
        # Quick evaluation of single rule
        if feature in df.columns:
            single_predictions = (df[feature] > mean_threshold).astype(int)
            
            if observed_events is not None:
                observed_aligned = observed_events.reindex(df.set_index('date').index, fill_value=0).values
                
                tp = np.sum((single_predictions == 1) & (observed_aligned == 1))
                fp = np.sum((single_predictions == 1) & (observed_aligned == 0))
                fn = np.sum((single_predictions == 0) & (observed_aligned == 1))
                
                single_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                single_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                single_f1 = 2 * (single_precision * single_recall) / (single_precision + single_recall) if (single_precision + single_recall) > 0 else 0
                
                print(f"Single rule F1: {single_f1:.3f}")
                
                # Quick AEP estimate for single rule
                single_aep = calculate_fast_multi_rule_aep(
                    df, [feature], ['>'], [], [mean_threshold], 
                    N_PARAM, W_PARAM, config.MIN_DAYS, n_simulations=200
                )
                
                if single_aep:
                    single_mean_loss = single_aep['summary']['mean_loss']
                    print(f"Single rule mean loss: ${single_mean_loss:,.0f}")
                    
                    # Compare with top multi-rules
                    print(f"\n🏆 IMPROVEMENT ANALYSIS:")
                    print("=" * 40)
                    
                    for i, (_, row) in enumerate(results_df.head(3).iterrows(), 1):
                        improvement = ((single_mean_loss - row['mean_loss']) / single_mean_loss) * 100
                        f1_improvement = ((row['f1_score'] - single_f1) / single_f1) * 100 if single_f1 > 0 else 0
                        
                        print(f"{i}. {row['type']}")
                        print(f"   Loss improvement: {improvement:+.1f}% (${single_mean_loss - row['mean_loss']:+,.0f})")
                        print(f"   F1 improvement: {f1_improvement:+.1f}%")
                        print()
                    
                    # Save comparison
                    comparison_data = {
                        'analysis_type': ['single_rule'] + [f"multi_rule_top_{i}" for i in range(1, 4)],
                        'description': [f"{feature} > {mean_threshold:.4f}"] + [row['description'] for _, row in results_df.head(3).iterrows()],
                        'f1_score': [single_f1] + list(results_df.head(3)['f1_score']),
                        'mean_loss': [single_mean_loss] + list(results_df.head(3)['mean_loss']),
                        'improvement_vs_single': [0] + [((single_mean_loss - row['mean_loss']) / single_mean_loss) * 100 for _, row in results_df.head(3).iterrows()]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    comparison_path = os.path.join(results_dir, f'baseline_comparison_{timestamp}.csv')
                    comparison_df.to_csv(comparison_path, index=False)
                    print(f"✅ Saved comparison: {comparison_path}")
                    
                    return comparison_df
        
    except Exception as e:
        print(f"Could not load single rule baseline: {e}")
        return None


def generate_triple_rule_analysis(top_rules, df, observed_events, timestamp, results_dir):
    """
    Quick analysis of promising triple rule combinations
    """
    print("\n🚀 TRIPLE RULE EXPLORATION")
    print("=" * 40)
    
    # Take top 4 features from single rules for triple combinations
    if len(top_rules) >= 4:
        top_features = [rule['feature'] for rule in top_rules[:4]]
        print(f"Testing triple combinations with: {top_features}")
        
        # Generate a few promising triple combinations
        triple_combinations = [
            {
                'features': [top_features[0], top_features[1], top_features[2]],
                'operators': ['>', '>', '>'],
                'logic_ops': ['AND', 'AND'],
                'description': f"{top_features[0]} AND {top_features[1]} AND {top_features[2]}",
                'type': 'triple_AND'
            },
            {
                'features': [top_features[0], top_features[1], top_features[3]],
                'operators': ['>', '>', '>'],
                'logic_ops': ['AND', 'OR'],
                'description': f"{top_features[0]} AND {top_features[1]} OR {top_features[3]}",
                'type': 'triple_AND_OR'
            }
        ]
        
        triple_results = []
        
        for i, combination in enumerate(triple_combinations):
            print(f"\n--- Triple {i+1}: {combination['type']} ---")
            
            try:
                # Quick threshold optimization
                if observed_events is not None:
                    best_thresholds, best_score = optimize_thresholds_fast(
                        df, combination, observed_events, threshold_grid_size=3
                    )
                    print(f"    F1: {best_score:.3f}")
                else:
                    best_thresholds = []
                    for feature in combination['features']:
                        if feature in df.columns:
                            best_thresholds.append(np.percentile(df[feature].dropna(), 75))
                        else:
                            best_thresholds.append(0.0)
                    best_score = 0.0
                
                # Quick AEP analysis (fewer simulations for speed)
                aep_results = calculate_fast_multi_rule_aep(
                    df, combination['features'], combination['operators'], 
                    combination['logic_ops'], best_thresholds, N_PARAM, W_PARAM, 
                    config.MIN_DAYS, n_simulations=200
                )
                
                if aep_results is not None:
                    result = {
                        'type': combination['type'],
                        'description': combination['description'],
                        'f1_score': best_score,
                        'mean_loss': aep_results['summary']['mean_loss'],
                        'zero_prob': aep_results['summary']['zero_prob']
                    }
                    triple_results.append(result)
                    print(f"    ✅ Mean loss: ${result['mean_loss']:,.0f}")
                else:
                    print(f"    ❌ Failed")
                    
            except Exception as e:
                print(f"    ❌ Error: {e}")
        
        if triple_results:
            print(f"\n🏆 TRIPLE RULE RESULTS:")
            for i, result in enumerate(triple_results, 1):
                print(f"{i}. {result['type']}: {result['description']}")
                print(f"   F1: {result['f1_score']:.3f} | Mean Loss: ${result['mean_loss']:,.0f}")
            
            # Save triple results
            triple_df = pd.DataFrame(triple_results)
            triple_path = os.path.join(results_dir, f'triple_rule_results_{timestamp}.csv')
            triple_df.to_csv(triple_path, index=False)
            print(f"\n✅ Saved triple results: {triple_path}")
            
            return triple_results
    
    return []


# ENHANCED MAIN FUNCTION WITH COMPREHENSIVE ANALYSIS
def enhanced_fast_multi_rule_main():
    """
    Enhanced version with comprehensive analysis and comparisons
    """
    print("\n🚀 ENHANCED FAST MULTI-RULE AEP ANALYSIS")
    print(f"Run: {config.RUN_PATH}")
    print(f"Reference port: {config.reference_port}")

    # Load input data
    merged_path = os.path.join(config.run_output_dir, 'df_swan_waverys_merged.csv')
    if not os.path.exists(merged_path):
        print(f"❌ Input file not found: {merged_path}")
        return
    df = pd.read_csv(merged_path, parse_dates=['date'])

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = config.results_output_dir

    # Locate CV results
    cv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv') and 'cv_results' in f]
    if not cv_files:
        print(f"❌ No CV results found in {results_dir}")
        return
    cv_results_path = os.path.join(results_dir, sorted(cv_files)[-1])

    # Setup observed events
    observed_col = 'event_dummy_1' if 'event_dummy_1' in df.columns else None
    observed_events = df.set_index('date')[observed_col] if observed_col else None

    # Step 1: Run the fast multi-rule analysis (as before)
    print("\n📋 Generating rule combinations...")
    top_rules = generate_top_single_rules(cv_results_path, top_k=8)
    double_combinations = generate_double_rule_combinations(top_rules, max_combinations=20)
    
    print(f"  Testing {len(double_combinations)} double rule combinations...")

    results = []
    for i, combination in enumerate(double_combinations):
        print(f"\n--- {i+1}/{len(double_combinations)}: {combination['type']} ---")
        
        try:
            if observed_events is not None:
                best_thresholds, best_score = optimize_thresholds_fast(
                    df, combination, observed_events, threshold_grid_size=4
                )
            else:
                best_thresholds = []
                for feature in combination['features']:
                    if feature in df.columns:
                        best_thresholds.append(np.percentile(df[feature].dropna(), 75))
                    else:
                        best_thresholds.append(0.0)
                best_score = 0.0
            
            aep_results = calculate_fast_multi_rule_aep(
                df, combination['features'], combination['operators'], 
                combination['logic_ops'], best_thresholds, N_PARAM, W_PARAM, 
                config.MIN_DAYS, n_simulations=config.N_SIMULATIONS
            )
            
            if aep_results is not None:
                result = {
                    'combination_id': i,
                    'type': combination['type'],
                    'description': combination['description'],
                    'features': combination['features'],
                    'thresholds': best_thresholds,
                    'f1_score': best_score,
                    'mean_loss': aep_results['summary']['mean_loss'],
                    'max_loss': aep_results['summary']['max_loss'],
                    'zero_prob': aep_results['summary']['zero_prob']
                }
                results.append(result)
                print(f"    ✅ Mean loss: ${result['mean_loss']:,.0f}")
                
        except Exception as e:
            print(f"    ❌ Error: {e}")

    if results:
        results_df = pd.DataFrame(results).sort_values('mean_loss')
        
        # Step 2: Enhanced analysis
        print(f"\n🏆 TOP 5 MULTI-RULE COMBINATIONS:")
        print("=" * 70)
        for i, (_, row) in enumerate(results_df.head(5).iterrows(), 1):
            print(f"{i}. {row['type']}: {row['description']}")
            print(f"   F1: {row['f1_score']:.3f} | Mean Loss: ${row['mean_loss']:,.0f}")
            print()
        
        # Step 3: Detailed analysis of top combinations
        detailed_results = analyze_top_combinations_detailed(
            results_df, df, observed_events, timestamp, results_dir
        )
        
        # Step 4: Comparison with single rule baseline
        comparison_results = create_comparison_with_baseline(
            results_df, df, cv_results_path, observed_events, timestamp, results_dir
        )
        
        # Step 5: Quick triple rule exploration
        triple_results = generate_triple_rule_analysis(
            top_rules, df, observed_events, timestamp, results_dir
        )
        
        # Step 6: Save main results
        summary_path = os.path.join(results_dir, f'enhanced_multi_rule_summary_{timestamp}.csv')
        results_df.to_csv(summary_path, index=False)
        
        print(f"\n✅ ENHANCED ANALYSIS COMPLETE!")
        print(f"📁 All results saved with timestamp: {timestamp}")
        
        return results_df, detailed_results, comparison_results, triple_results
    else:
        print("❌ No successful analyses")
        return None, None, None, None

# --- MAIN EXECUTION ---
def main():
    print("\n🔧 AEP_CALCULATION.PY - Final AEP Analysis (Multi-threshold Experiment)")
    print(f"Run: {config.RUN_PATH}")
    print(f"Reference port: {config.reference_port}")

    # --- Load input data ---
    merged_path = os.path.join(config.run_output_dir, 'df_swan_waverys_merged.csv')
    if not os.path.exists(merged_path):
        print(f"❌ Input file not found: {merged_path}")
        return
    df = pd.read_csv(merged_path, parse_dates=['date'])

    # --- Set timestamp and results_dir for output files ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = config.results_output_dir

    # --- Locate latest CV results file ---
    cv_files = [f for f in os.listdir(results_dir) if (f == 'rule_cv_results.csv' or (f.startswith('cv_results_') and f.endswith('.csv')))]
    if not cv_files:
        print(f"❌ No CV results found in {results_dir}")
        return
    latest_cv = sorted(cv_files)[-1]
    cv_results_path = os.path.join(results_dir, latest_cv)

    # --- Select best rule and gather all candidate thresholds ---
    feature, fold_thresholds = load_best_single_rule_and_all_thresholds(cv_results_path, results_dir)
    print(f"✅ Best single rule: {feature}")
    print(f"Fold thresholds: {fold_thresholds}")

    # Compute candidate thresholds
    candidate_thresholds = []
    candidate_labels = []
    # Individual fold thresholds
    for i, t in enumerate(fold_thresholds):
        candidate_thresholds.append(t)
        candidate_labels.append(f'fold{i}')
    # Mean, median, 75th percentile
    mean_thr = np.mean(fold_thresholds)
    median_thr = np.median(fold_thresholds)
    p75_thr = np.percentile(fold_thresholds, 75)
    candidate_thresholds.extend([mean_thr, median_thr, p75_thr])
    candidate_labels.extend(['mean', 'median', 'p75'])

    # --- Optional: Plot rule condition for validation ---
    if ENABLE_PLOTTING:
        plot_path = os.path.join(results_dir, f'best_rule_plot_{config.RUN_PATH}.png')
        plot_rule_condition(df, feature, mean_thr, save_path=plot_path)

    # --- Run AEP simulation for each candidate threshold ---
    print("\n🚀 Running speed-optimized AEP simulation for multiple thresholds...")
    observed_col = 'event_dummy_1' if 'event_dummy_1' in df.columns else None
    observed_events = df.set_index('date')[observed_col] if observed_col else None
    swh_data = df.set_index('date')

    summary_rows = []
    for thr, label in zip(candidate_thresholds, candidate_labels):
        print(f"\n--- Threshold ({label}): {thr:.5f} ---")
        # aep_results = calculate_unified_aep_analysis_fast(
        #     swh_data,
        #     trigger_feature=feature,
        #     trigger_threshold=thr,
        #     N=N_PARAM,
        #     W=W_PARAM,
        #     min_days=config.MIN_DAYS,
        #     n_simulations=config.N_SIMULATIONS,
        #     observed_events=observed_events,
        #     block_length=config.BLOCK_LENGTH,
        #     window_days=config.WINDOW_DAYS,
        #     n_jobs=-1
        # )
        aep_results = calculate_enhanced_aep_analysis(
            swh_data,
            trigger_feature=feature,
            trigger_threshold=thr,
            N=N_PARAM,
            W=W_PARAM,
            min_days=config.MIN_DAYS,
            n_simulations=config.N_SIMULATIONS,
            observed_events=observed_events,
            block_length=config.BLOCK_LENGTH,
            window_days=config.WINDOW_DAYS,
            n_jobs=-1   
        )
        if aep_results is None:
            print(f"❌ AEP simulation failed for threshold {thr:.5f} ({label})")
            continue
        # Print summary
        mean_loss = aep_results['standard_summary']['mean_loss']
        print(f"Rule-based method ({label}): mean annual loss = ${mean_loss:,.0f}")
        # Save results
        summary_path = os.path.join(results_dir, f'aep_summary_{label}_{timestamp}.csv')
        aep_curve_path = os.path.join(results_dir, f'aep_curve_{label}_{timestamp}.csv')
        pd.DataFrame([aep_results['standard_summary']]).to_csv(summary_path, index=False)
        pd.DataFrame(aep_results['standard_aep_curve']).to_csv(aep_curve_path, index=False)
        # --- Save observed yearly losses as CSV if available ---
        if 'obs_yearly_losses' in aep_results and aep_results['obs_yearly_losses']:
            obs_losses = aep_results['obs_yearly_losses']
            obs_losses_path = os.path.join(results_dir, f'observed_yearly_losses_{label}_{timestamp}.csv')
            if isinstance(obs_losses, dict):
                obs_df = pd.DataFrame({
                    'year': list(obs_losses.keys()),
                    'observed_loss': list(obs_losses.values())
                })
            elif isinstance(obs_losses, (list, tuple, pd.Series)):
                obs_df = pd.DataFrame({'observed_loss': list(obs_losses)})
            else:
                obs_df = None
            if obs_df is not None:
                obs_df.to_csv(obs_losses_path, index=False)
        if 'obs_yearly_losses' in aep_results:
            row = dict(threshold_label=label, threshold_value=thr)
            row['W_param'] = W_PARAM
            row['N_param'] = N_PARAM
            row['min_days'] = config.MIN_DAYS
            row.update(aep_results['standard_summary'])
            # Add confusion matrix results for later summary table
            row['confusion_matrix_results'] = aep_results.get('confusion_matrix_results', None)
            # Add observed yearly losses if present
            if 'obs_yearly_losses' in aep_results:
                row['observed_yearly_losses'] = aep_results['obs_yearly_losses']
            summary_rows.append(row)

    # --- Combined summary CSV for all thresholds ---
    if summary_rows:
        all_summary_path = os.path.join(results_dir, f'aep_multi_threshold_summary_MIN_DAYS_{config.MIN_DAYS}_{timestamp}.csv')
        pd.DataFrame(summary_rows).to_csv(all_summary_path, index=False)
        print(f"\n✅ Saved all-threshold summary: {all_summary_path}")

        # --- Print and save final full summary table for each threshold ---
        print("\n===== FINAL SUMMARY TABLE (full stats) =====")
        summary_table = []
        observed_stats = None
        observed_losses = None
        for thr, label in zip(candidate_thresholds, candidate_labels):
            aep_curve_path = os.path.join(results_dir, f'aep_curve_{label}_{timestamp}.csv')
            summary_path = os.path.join(results_dir, f'aep_summary_{label}_{timestamp}.csv')
            try:
                aep_curve = pd.read_csv(aep_curve_path)
                losses = aep_curve['loss'].values
                mean_loss = np.mean(losses)
                median_loss = np.median(losses)
                p99_loss = np.percentile(losses, 99)
            except Exception as e:
                print(f"[Warning] Could not load curve for {label}: {e}")
                mean_loss = median_loss = p99_loss = np.nan

            # Load confusion matrix and event stats from summary file
            try:
                summary_df = pd.read_csv(summary_path)
                n_sim = int(summary_df['n_simulations'].iloc[0]) if 'n_simulations' in summary_df else None
            except Exception as e:
                n_sim = None

            # Try to load event stats from confusion matrix results (if present)
            cm_stats = {'mean_events': np.nan, 'median_events': np.nan, 'p99_events': np.nan,
                        'mean_tp': np.nan, 'median_tp': np.nan, 'p99_tp': np.nan,
                        'mean_fp': np.nan, 'median_fp': np.nan, 'p99_fp': np.nan,
                        'mean_tn': np.nan, 'median_tn': np.nan, 'p99_tn': np.nan,
                        'mean_fn': np.nan, 'median_fn': np.nan, 'p99_fn': np.nan}
            obs_stats = {'obs_tp': np.nan, 'obs_fp': np.nan, 'obs_tn': np.nan, 'obs_fn': np.nan}
            obs_losses_row = {'obs_losses': np.nan}
            # Try to get these from the confusion matrix results if available
            try:
                # Try to load the confusion matrix results from the in-memory run
                # (This block assumes the run was not interrupted and aep_results is available)
                # Otherwise, skip
                pass  # We'll fill these in during the main loop below
            except Exception:
                pass

            # Instead, let's use the in-memory aep_results from the run (already in summary_rows)
            # Find the correct row in summary_rows
            this_row = next((r for r in summary_rows if r['threshold_label'] == label), None)
            if this_row and 'confusion_matrix_results' in this_row and this_row['confusion_matrix_results'] is not None:
                cmr = this_row['confusion_matrix_results']
                # Simulated event stats
                all_event_counts = cmr.get('all_event_counts', [])
                if all_event_counts and any([d.get('n_events', 0) > 0 for d in all_event_counts]):
                    tp_list = [d.get('tp_events', 0) for d in all_event_counts]
                    fp_list = [d.get('fp_events', 0) for d in all_event_counts]
                    tn_list = [d.get('tn_events', 0) for d in all_event_counts]
                    fn_list = [d.get('fn_events', 0) for d in all_event_counts]
                    n_events_list = [d.get('n_events', 0) for d in all_event_counts]
                else:
                    tp_list = fp_list = tn_list = fn_list = n_events_list = [0]
                cm_stats = {
                    'mean_events': np.mean(n_events_list),
                    'median_events': np.median(n_events_list),
                    'p99_events': np.percentile(n_events_list, 99),
                    'mean_tp': np.mean(tp_list),
                    'median_tp': np.median(tp_list),
                    'p99_tp': np.percentile(tp_list, 99),
                    'mean_fp': np.mean(fp_list),
                    'median_fp': np.median(fp_list),
                    'p99_fp': np.percentile(fp_list, 99),
                    'mean_tn': np.mean(tn_list),
                    'median_tn': np.median(tn_list),
                    'p99_tn': np.percentile(tn_list, 99),
                    'mean_fn': np.mean(fn_list),
                    'median_fn': np.median(fn_list),
                    'p99_fn': np.percentile(fn_list, 99),
                }
                # Losses for TP, FP, FN
                tp_annual_costs = cmr.get('tp_annual_costs', np.array([0]))
                fp_annual_costs = cmr.get('fp_annual_costs', np.array([0]))
                fn_annual_costs = cmr.get('fn_annual_costs', np.array([0]))
                loss_stats = {
                    'mean_tp_loss': np.mean(tp_annual_costs),
                    'median_tp_loss': np.median(tp_annual_costs),
                    'p99_tp_loss': np.percentile(tp_annual_costs, 99),
                    'mean_fp_loss': np.mean(fp_annual_costs),
                    'median_fp_loss': np.median(fp_annual_costs),
                    'p99_fp_loss': np.percentile(fp_annual_costs, 99),
                    'mean_fn_loss': np.mean(fn_annual_costs),
                    'median_fn_loss': np.median(fn_annual_costs),
                    'p99_fn_loss': np.percentile(fn_annual_costs, 99),
                }
                # Observed stats (repeat for all thresholds)
                obs_stats = cmr.get('observed_stats', obs_stats)
            # Observed yearly loss and events
            mean_obs_yearly_loss = np.nan
            mean_obs_yearly_events = np.nan
            obs_yearly_losses = this_row.get('obs_yearly_losses', None)
            if obs_yearly_losses and isinstance(obs_yearly_losses, dict):
                loss_vals = list(obs_yearly_losses.values())
                if len(loss_vals) > 0:
                    mean_obs_yearly_loss = np.mean(loss_vals)
            # For observed yearly events, need to get observed_events from original data
            # We'll approximate using obs_tp + obs_fn per year if available
            # But here, just use obs_tp + obs_fn (total) divided by number of years
            if 'obs_tp' in obs_stats and 'obs_fn' in obs_stats:
                n_years = len(obs_yearly_losses) if obs_yearly_losses else 1
                mean_obs_yearly_events = (obs_stats['obs_tp'] + obs_stats['obs_fn']) / n_years if n_years > 0 else (obs_stats['obs_tp'] + obs_stats['obs_fn'])
            # Compose final row
            row = {
                'variable': feature,
                'run_path': config.RUN_PATH,
                'W_param': W_PARAM,
                'N_param': N_PARAM,
                'min_days': config.MIN_DAYS,
                'threshold_label': label,
                'threshold_value': float(thr),
                'mean_loss': mean_loss,
                'median_loss': median_loss,
                'p99_loss': p99_loss,
                **cm_stats,
                **loss_stats,
                **obs_stats,
                'mean_obs_yearly_loss': mean_obs_yearly_loss,
                'mean_obs_yearly_events': mean_obs_yearly_events
            }
            summary_table.append(row)
        # Add mean_loss_minus_obs and sort by abs(mean_loss_minus_obs)
        for row in summary_table:
            row['mean_loss_minus_obs'] = row['mean_loss'] - row['mean_obs_yearly_loss'] if not pd.isna(row['mean_obs_yearly_loss']) else np.nan
        # Sort by abs(mean_loss_minus_obs)
        summary_table_sorted = sorted(summary_table, key=lambda r: abs(r['mean_loss_minus_obs']) if not pd.isna(r['mean_loss_minus_obs']) else float('inf'))
        # Print as table
        import tabulate
        float_cols = [k for k in summary_table_sorted[0].keys() if k != 'threshold_label']
        print(tabulate.tabulate(summary_table_sorted, headers="keys", floatfmt=(".3f",)+tuple([".2f"]*(len(float_cols)))))
        # Save to CSV
        full_summary_path = os.path.join(results_dir, f'aep_multi_threshold_full_summary_MIN_DAYS_{config.MIN_DAYS}_{timestamp}.csv')
        pd.DataFrame(summary_table_sorted).to_csv(full_summary_path, index=False)
        print(f"\n✅ Saved full summary table: {full_summary_path}")

        # --- Plot AEP curve for the best threshold (smallest abs(mean_loss_minus_obs)) ---
        best_row = summary_table_sorted[0]
        best_label = best_row['threshold_label']
        best_curve_path = os.path.join(results_dir, f'aep_curve_{best_label}_{timestamp}.csv')
        best_obs_losses = None
        if os.path.exists(best_curve_path):
            aep_curve = pd.read_csv(best_curve_path)
            plt.figure(figsize=(12, 7))
            plt.plot(aep_curve['loss'], aep_curve['probability'], color='blue', lw=2.5, marker='o', markersize=3)
            plt.xlabel('Loss ($)', fontsize=14)
            plt.ylabel('Exceedance Probability', fontsize=14)
            plt.title(f'Best AEP Curve: {best_label}', fontsize=16)
            # Overlay observed yearly losses from CSV if available (pipeline style)
            import seaborn as sns
            handles = []
            labels = []
            obs_losses_csv = os.path.join(results_dir, f'observed_yearly_losses_{best_label}_{timestamp}.csv')
            if os.path.exists(obs_losses_csv):
                obs_df = pd.read_csv(obs_losses_csv)
                print('DEBUG: observed losses loaded:')
                print(obs_df)
                for idx, row in obs_df.iterrows():
                    plt.axvline(row['observed_loss'], color='gray', linestyle='--', linewidth=2, alpha=0.8)
                    if 'year' in row:
                        # Remove .0 from label, shift slightly right
                        year_label = str(int(row['year']))
                        x_shift = 10  # pixels
                        plt.text(row['observed_loss'] + x_shift, 0.98, year_label, rotation=90, color='gray', fontsize=12, ha='center', va='top', fontweight='bold')
            else:
                print(f"DEBUG: Observed losses CSV not found: {obs_losses_csv}")
            # Add vertical line for AEP mean loss in dark green
            if 'mean_loss' in best_row and not np.isnan(best_row['mean_loss']):
                h_aep = plt.axvline(best_row['mean_loss'], color='forestgreen', linestyle=':', linewidth=2.5, label='AEP Mean')
                plt.text(best_row['mean_loss'], 0.85, f'AEP Mean\n${best_row["mean_loss"]/1000:.0f}K', color='forestgreen', fontsize=12, fontweight='bold', ha='center', va='top')
                plt.legend([h_aep], ['AEP Mean'], loc='upper right', fontsize=12)
            # Format x-axis as $X,XXXK
            import matplotlib.ticker as mticker
            def k_fmt(x, pos):
                return f'${int(x/1000):,}K'
            plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(k_fmt))
            # Use variable and threshold in title
            plot_variable = best_row.get('variable', best_label)
            plot_thr = best_row.get('threshold_value', None)
            if plot_thr is not None:
                plt.title(f'{plot_variable} | Threshold: {plot_thr:.5f}', fontsize=16)
            else:
                plt.title(str(plot_variable), fontsize=16)
            plt.tight_layout()
            out_path = os.path.join(results_dir, f'aep_curve_best_threshold_MIN_DAYS_{config.MIN_DAYS}_{timestamp}.png')
            plt.savefig(out_path)
            print(f"✅ Saved best AEP curve plot: {out_path}")
            plt.close()

    print("\n🎉 Multi-threshold AEP calculation completed!")


# DIRECT FIX: Enhanced Multi-Rule Pipeline with Full AEP Analysis
# Add this to your aep_calculation_experiment.py to replace the existing multi-rule function

def enhanced_multi_rule_main_with_full_aep():
    """
    Enhanced Multi-Rule analysis with FULL AEP pipeline - generates complete simulation outputs
    matching Single Rule format for proper comparison
    """
    print("\n🚀 ENHANCED MULTI-RULE AEP ANALYSIS WITH FULL SIMULATION")
    print(f"Run: {config.RUN_PATH}")
    print(f"Reference port: {config.reference_port}")

    # Load input data
    merged_path = os.path.join(config.run_output_dir, 'df_swan_waverys_merged.csv')
    if not os.path.exists(merged_path):
        print(f"❌ Input file not found: {merged_path}")
        return
    df = pd.read_csv(merged_path, parse_dates=['date'])

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = config.results_output_dir

    # Locate CV results
    cv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv') and 'cv_results' in f]
    if not cv_files:
        print(f"❌ No CV results found in {results_dir}")
        return
    cv_results_path = os.path.join(results_dir, sorted(cv_files)[-1])

    # Setup observed events
    observed_col = 'event_dummy_1' if 'event_dummy_1' in df.columns else None
    observed_events = df.set_index('date')[observed_col] if observed_col else None

    # Get top rules for combination testing
    print("\n📋 Generating rule combinations...")
    top_rules = generate_top_single_rules(cv_results_path, top_k=8)
    double_combinations = generate_double_rule_combinations(top_rules, max_combinations=20)
    
    print(f"  Testing {len(double_combinations)} double rule combinations...")

    # Find the best Multi-Rule combination first
    best_combination = None
    best_f1 = 0
    
    print("\n🔍 Finding best Multi-Rule combination...")
    for i, combination in enumerate(double_combinations):
        try:
            # Optimize thresholds for this combination
            if observed_events is not None:
                best_thresholds, f1_score = optimize_thresholds_fast(
                    df, combination, observed_events, threshold_grid_size=4
                )
                
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_combination = {
                        **combination,
                        'thresholds': best_thresholds,
                        'f1_score': f1_score
                    }
                    print(f"    🎯 New best: {combination['type']} F1={f1_score:.3f}")
            
        except Exception as e:
            print(f"    ❌ Error with combination {i}: {e}")
            continue
    
    if best_combination is None:
        print("❌ No valid Multi-Rule combination found")
        return
    
    print(f"\n✅ Best Multi-Rule: {best_combination['type']}")
    print(f"   Features: {best_combination['features']}")
    print(f"   Thresholds: {best_combination['thresholds']}")
    print(f"   F1 Score: {best_combination['f1_score']:.3f}")

    # NOW RUN FULL AEP ANALYSIS ON THE BEST COMBINATION
    print(f"\n🚀 Running FULL AEP analysis on best Multi-Rule combination...")
    
    # Create a custom multi-rule trigger function
    def create_multi_rule_trigger_series(df_data, combination):
        """Create a trigger series for multi-rule evaluation"""
        features = combination['features']
        thresholds = combination['thresholds'] 
        rule_type = combination['type']
        
        if len(features) != len(thresholds):
            raise ValueError("Features and thresholds length mismatch")
        
        # Apply rule logic
        if 'AND' in rule_type:
            trigger = df_data[features[0]] > thresholds[0]
            for i in range(1, len(features)):
                trigger = trigger & (df_data[features[i]] > thresholds[i])
        elif 'OR' in rule_type:
            trigger = df_data[features[0]] > thresholds[0]
            for i in range(1, len(features)):
                trigger = trigger | (df_data[features[i]] > thresholds[i])
        else:
            raise ValueError(f"Unknown rule type: {rule_type}")
        
        return trigger.astype(int)
    
    # Create the trigger series for the best combination
    try:
        swh_data = df.set_index('date')
        multi_rule_trigger = create_multi_rule_trigger_series(df, best_combination)
        
        # We need to adapt the AEP analysis to work with a pre-computed trigger series
        # instead of a single feature + threshold
        
        print(f"🔧 Running FULL AEP simulation with {config.N_SIMULATIONS} simulations...")
        
        # Use the enhanced AEP analysis but adapted for multi-rule
        multi_rule_aep_results = calculate_multi_rule_aep_with_full_simulation(
            swh_data=swh_data,
            trigger_series=multi_rule_trigger,
            combination_info=best_combination,
            N=N_PARAM,
            W=W_PARAM,
            min_days=config.MIN_DAYS,
            n_simulations=config.N_SIMULATIONS,
            observed_events=observed_events,
            block_length=config.BLOCK_LENGTH,
            window_days=config.WINDOW_DAYS,
            n_jobs=-1
        )
        
        if multi_rule_aep_results is None:
            print("❌ Multi-Rule AEP simulation failed")
            return
        
        # Save comprehensive results (same format as Single Rule)
        multi_summary_path = os.path.join(results_dir, f'multi_rule_aep_summary_{timestamp}.csv')
        multi_curve_path = os.path.join(results_dir, f'multi_rule_aep_curve_{timestamp}.csv')
        
        # Save main summary
        summary_data = multi_rule_aep_results['standard_summary'].copy()
        summary_data.update({
            'rule_type': best_combination['type'],
            'rule_description': best_combination.get('description', 'Multi-condition rule'),
            'features': str(best_combination['features']),
            'thresholds': str(best_combination['thresholds']),
            'f1_score': best_combination['f1_score']
        })
        
        pd.DataFrame([summary_data]).to_csv(multi_summary_path, index=False)
        multi_rule_aep_results['standard_aep_curve'].to_csv(multi_curve_path, index=False)
        
        print(f"✅ Saved Multi-Rule AEP summary: {multi_summary_path}")
        print(f"✅ Saved Multi-Rule AEP curve: {multi_curve_path}")
        
        # Save observed yearly losses (same as Single Rule)
        if 'obs_yearly_losses' in multi_rule_aep_results:
            obs_losses_df = pd.DataFrame(
                list(multi_rule_aep_results['obs_yearly_losses'].items()),
                columns=['year', 'observed_loss']
            )
            obs_losses_path = os.path.join(results_dir, f'multi_rule_observed_yearly_losses_{timestamp}.csv')
            obs_losses_df.to_csv(obs_losses_path, index=False)
            print(f"✅ Saved Multi-Rule observed yearly losses: {obs_losses_path}")
        
        # Save confusion matrix results if available
        if multi_rule_aep_results.get('confusion_matrix_results'):
            cm_results = multi_rule_aep_results['confusion_matrix_results']
            
            # Save individual CM curves
            cm_results['fp_aep'].to_csv(
                os.path.join(results_dir, f'multi_rule_fp_aep_curve_{timestamp}.csv'), index=False
            )
            cm_results['fn_aep'].to_csv(
                os.path.join(results_dir, f'multi_rule_fn_aep_curve_{timestamp}.csv'), index=False
            )
            cm_results['tp_aep'].to_csv(
                os.path.join(results_dir, f'multi_rule_tp_aep_curve_{timestamp}.csv'), index=False
            )
            
            # Save CM summary
            cm_summary_df = pd.DataFrame([cm_results['summary']])
            cm_summary_path = os.path.join(results_dir, f'multi_rule_confusion_matrix_summary_{timestamp}.csv')
            cm_summary_df.to_csv(cm_summary_path, index=False)
            print(f"✅ Saved Multi-Rule confusion matrix analysis: {cm_summary_path}")
        
        # Print final summary
        mean_loss = multi_rule_aep_results['standard_summary']['mean_loss']
        max_loss = multi_rule_aep_results['standard_summary']['max_loss']
        zero_prob = multi_rule_aep_results['standard_summary']['zero_prob']
        
        print(f"\n📊 MULTI-RULE AEP RESULTS:")
        print(f"   Rule: {best_combination['type']} - {best_combination.get('description', 'Multi-condition')}")
        print(f"   Mean Annual Loss: ${mean_loss:,.0f}")
        print(f"   Max Annual Loss: ${max_loss:,.0f}")
        print(f"   Zero Loss Probability: {zero_prob:.1%}")
        print(f"   F1 Score: {best_combination['f1_score']:.3f}")
        
        if multi_rule_aep_results.get('confusion_matrix_results'):
            cm_summary = multi_rule_aep_results['confusion_matrix_results']['summary']
            print(f"   FP Cost: ${cm_summary['fp_costs']['mean']:,.0f}")
            print(f"   FN Cost: ${cm_summary['fn_costs']['mean']:,.0f}")
            print(f"   TP Cost: ${cm_summary['tp_costs']['mean']:,.0f}")
        
        return multi_rule_aep_results
        
    except Exception as e:
        print(f"❌ Error in Multi-Rule AEP analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

# ENHANCEMENT: Modify existing working pipeline to generate complete metrics
# This enhances calculate_fast_multi_rule_aep() and related functions

def calculate_enhanced_multi_rule_aep(df, rule_features, rule_operators, rule_logic_ops, 
                                      thresholds, N, W, min_days, n_simulations=4000):
    """Enhanced version of the existing fast multi-rule AEP with complete metrics"""
    print(f"  🚀 Enhanced multi-rule AEP: {len(rule_features)} features, {n_simulations} sims")
    
    # Prepare data (same as existing)
    df_clean = df.copy().sort_values('date').reset_index(drop=True)
    
    # Check if all features exist
    missing_features = [f for f in rule_features if f not in df_clean.columns]
    if missing_features:
        print(f"    ❌ Missing features: {missing_features}")
        return None
    
    # Get observed events
    if 'event_dummy_1' not in df_clean.columns:
        print("❌ No observed events column found")
        return None
        
    observed = df_clean['event_dummy_1'].astype(int)
    print(f"📊 Observed events: {observed.sum()} out of {len(observed)} days")
    
    # Apply the multi-rule to get predictions on the full dataset
    predictions_full = evaluate_multi_rule_fast(
        df_clean, rule_features, rule_operators, rule_logic_ops, thresholds
    )
    
    print(f"📊 Rule predictions: {predictions_full.sum()} out of {len(predictions_full)} days ({predictions_full.mean()*100:.1f}%)")
    
    if predictions_full.sum() == 0:
        print("⚠️ Warning: Rule produces 0 predictions - this will result in 0 losses")
    
    # Calculate observed confusion matrix
    obs_tp = int(np.sum((predictions_full == 1) & (observed == 1)))
    obs_fp = int(np.sum((predictions_full == 1) & (observed == 0)))
    obs_tn = int(np.sum((predictions_full == 0) & (observed == 0)))
    obs_fn = int(np.sum((predictions_full == 0) & (observed == 1)))
    
    print(f"📊 Observed confusion matrix: TP={obs_tp}, FP={obs_fp}, TN={obs_tn}, FN={obs_fn}")
    
    # Extract feature matrices for simulation (same as existing)
    feature_matrices = {}
    for feature in rule_features:
        feature_matrices[feature] = df_clean[feature].fillna(0).values.astype(np.float32)
    
    # Enhanced simulation with detailed tracking
    print(f"🔄 Running enhanced simulation with detailed tracking...")
    
    annual_losses = []
    annual_events = []
    annual_fp_counts = []
    annual_fn_counts = []
    annual_tp_counts = []
    annual_fp_costs = []
    annual_fn_costs = []
    annual_tp_costs = []
    
    # Use block bootstrap for proper temporal correlation
    from datetime import datetime, timedelta
    
    # Simple block bootstrap indices generation
    np.random.seed(42)  # For reproducibility
    for sim_idx in range(n_simulations):
        np.random.seed(sim_idx)  # Different seed per simulation
        
        try:
            # Create annual simulation (365 days) using block sampling
            sim_indices = []
            block_length = 7  # Weekly blocks
            
            while len(sim_indices) < 365:
                # Random starting position for block
                start_idx = np.random.randint(0, max(1, len(df_clean) - block_length))
                block_indices = list(range(start_idx, min(start_idx + block_length, len(df_clean))))
                sim_indices.extend(block_indices)
            
            # Trim to exactly 365 days
            sim_indices = sim_indices[:365]
            
            # Sample feature values for this simulation
            sampled_features = {}
            for feature in rule_features:
                sampled_features[feature] = feature_matrices[feature][sim_indices]
            
            # Create simulation DataFrame
            sim_df = pd.DataFrame(sampled_features)
            
            # Apply rule to simulation
            sim_predictions = evaluate_multi_rule_fast(
                sim_df, rule_features, rule_operators, rule_logic_ops, thresholds
            )
            
            # Sample corresponding observed events
            sim_observed = observed.iloc[sim_indices].values[:365]
            
            # Calculate annual loss
            total_loss = calculate_annual_loss_jit(sim_predictions, N, W, min_days)
            annual_losses.append(total_loss)
            
            # Count events
            annual_events.append(sim_predictions.sum())
            
            # Calculate confusion matrix for this simulation
            sim_tp = int(np.sum((sim_predictions == 1) & (sim_observed == 1)))
            sim_fp = int(np.sum((sim_predictions == 1) & (sim_observed == 0)))
            sim_fn = int(np.sum((sim_predictions == 0) & (sim_observed == 1)))
            
            annual_tp_counts.append(sim_tp)
            annual_fp_counts.append(sim_fp)
            annual_fn_counts.append(sim_fn)
            
            # Calculate costs for each confusion matrix component
            fp_cost, fn_cost, tp_cost = calculate_cm_costs_jit(
                sim_predictions, sim_observed, N, W, min_days
            )
            annual_fp_costs.append(fp_cost)
            annual_fn_costs.append(fn_cost)
            annual_tp_costs.append(tp_cost)
            
        except Exception as e:
            # Handle failures gracefully
            annual_losses.append(0)
            annual_events.append(0)
            annual_tp_counts.append(0)
            annual_fp_counts.append(0)
            annual_fn_counts.append(0)
            annual_fp_costs.append(0)
            annual_fn_costs.append(0)
            annual_tp_costs.append(0)
    
    # Convert to numpy arrays for analysis
    annual_losses = np.array(annual_losses)
    annual_events = np.array(annual_events)
    annual_fp_counts = np.array(annual_fp_counts)
    annual_fn_counts = np.array(annual_fn_counts)
    annual_tp_counts = np.array(annual_tp_counts)
    annual_fp_costs = np.array(annual_fp_costs)
    annual_fn_costs = np.array(annual_fn_costs)
    annual_tp_costs = np.array(annual_tp_costs)
    
    print(f"✅ Completed {len(annual_losses)} simulations")
    
    # Calculate AEP curve
    losses_sorted = np.sort(annual_losses)[::-1]
    exceedance_prob = np.arange(1, len(losses_sorted)+1) / (len(losses_sorted)+1)
    
    aep_curve = pd.DataFrame({
        'loss': losses_sorted,
        'probability': exceedance_prob
    })
    
    # Calculate comprehensive summary with all the missing metrics
    summary = {
        'mean_loss': float(np.mean(annual_losses)),
        'std_loss': float(np.std(annual_losses)),
        'max_loss': float(np.max(annual_losses)),
        'min_loss': float(np.min(annual_losses)),
        'zero_prob': float(np.mean(annual_losses == 0)),
        'p99_loss': float(np.percentile(annual_losses, 99)),
        
        # Event metrics
        'mean_events': float(np.mean(annual_events)),
        'std_events': float(np.std(annual_events)),
        'p99_events': float(np.percentile(annual_events, 99)),
        'max_events': float(np.max(annual_events)),
        
        # Confusion matrix event counts
        'mean_tp': float(np.mean(annual_tp_counts)),
        'mean_fp': float(np.mean(annual_fp_counts)),
        'mean_fn': float(np.mean(annual_fn_counts)),
        'p99_tp': float(np.percentile(annual_tp_counts, 99)),
        'p99_fp': float(np.percentile(annual_fp_counts, 99)),
        'p99_fn': float(np.percentile(annual_fn_counts, 99)),
        
        # Confusion matrix costs
        'mean_tp_cost': float(np.mean(annual_tp_costs)),
        'mean_fp_cost': float(np.mean(annual_fp_costs)),
        'mean_fn_cost': float(np.mean(annual_fn_costs)),
        'p99_tp_cost': float(np.percentile(annual_tp_costs, 99)),
        'p99_fp_cost': float(np.percentile(annual_fp_costs, 99)),
        'p99_fn_cost': float(np.percentile(annual_fn_costs, 99)),
        
        # Observed confusion matrix
        'obs_tp': obs_tp,
        'obs_fp': obs_fp,
        'obs_tn': obs_tn,
        'obs_fn': obs_fn,
        
        # Performance metrics
        'obs_precision': obs_tp / (obs_tp + obs_fp) if (obs_tp + obs_fp) > 0 else 0,
        'obs_recall': obs_tp / (obs_tp + obs_fn) if (obs_tp + obs_fn) > 0 else 0,
        'obs_accuracy': (obs_tp + obs_tn) / len(observed) if len(observed) > 0 else 0,
        'obs_f1': 2 * obs_tp / (2 * obs_tp + obs_fp + obs_fn) if (2 * obs_tp + obs_fp + obs_fn) > 0 else 0,
        
        # Metadata
        'method': 'enhanced_multi_rule_block_bootstrap',
        'rule_features': rule_features,
        'thresholds': thresholds,
        'min_days': min_days,
        'n_fishermen': N,
        'daily_wage': W,
        'n_simulations': len(annual_losses)
    }
    
    return {
        'summary': summary,
        'aep_curve': aep_curve,
        'annual_losses': annual_losses,
        'annual_events': annual_events,
        'annual_fp_costs': annual_fp_costs,
        'annual_fn_costs': annual_fn_costs,
        'annual_tp_costs': annual_tp_costs,
        'distributions': {
            'losses': annual_losses,
            'events': annual_events,
            'tp_counts': annual_tp_counts,
            'fp_counts': annual_fp_counts,
            'fn_counts': annual_fn_counts
        }
    }

# Enhanced version of the main multi-rule function
def enhanced_multi_rule_main_keeping_working_parts():
    """
    Enhanced version that keeps the working rule evaluation but adds complete simulation outputs
    """
    print("\n🚀 ENHANCED MULTI-RULE AEP ANALYSIS (Keeping Working Parts)")
    print(f"Run: {config.RUN_PATH}")
    print(f"Reference port: {config.reference_port}")

    # Load input data (same as working version)
    merged_path = os.path.join(config.run_output_dir, 'df_swan_waverys_merged.csv')
    if not os.path.exists(merged_path):
        print(f"❌ Input file not found: {merged_path}")
        return
    df = pd.read_csv(merged_path, parse_dates=['date'])

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = config.results_output_dir

    # Locate CV results (same as working version)
    cv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv') and 'cv_results' in f]
    if not cv_files:
        print(f"❌ No CV results found in {results_dir}")
        return
    cv_results_path = os.path.join(results_dir, sorted(cv_files)[-1])

    # Setup observed events (same as working version)
    observed_col = 'event_dummy_1' if 'event_dummy_1' in df.columns else None
    observed_events = df.set_index('date')[observed_col] if observed_col else None

    # Get top rules (same as working version)
    print("\n📋 Generating rule combinations...")
    top_rules = generate_top_single_rules(cv_results_path, top_k=6)
    if len(top_rules) == 0:
        print("❌ No single rules found")
        return
    
    print(f"  Top features: {[r['feature'] for r in top_rules[:3]]}")
    
    # Generate combinations (same as working version)
    double_combinations = generate_double_rule_combinations(top_rules, max_combinations=15)
    print(f"  Testing {len(double_combinations)} double rule combinations...")

    # Find best combination (same as working version)
    best_results = []
    for i, combination in enumerate(double_combinations):
        print(f"\n--- {i+1}/{len(double_combinations)}: {combination['type']} ---")
        
        try:
            # Optimize thresholds (same as working version)
            if observed_events is not None:
                best_thresholds, best_score = optimize_thresholds_fast(
                    df, combination, observed_events, threshold_grid_size=4
                )
                print(f"    F1: {best_score:.3f}")
            else:
                best_thresholds = []
                for feature in combination['features']:
                    if feature in df.columns:
                        best_thresholds.append(np.percentile(df[feature].dropna(), 75))
                    else:
                        best_thresholds.append(0.0)
                best_score = 0.0
            
            # NEW: Run enhanced AEP analysis instead of fast version
            aep_results = calculate_enhanced_multi_rule_aep(
                df, combination['features'], combination['operators'], 
                combination['logic_ops'], best_thresholds, N_PARAM, W_PARAM, 
                config.MIN_DAYS, n_simulations=config.N_SIMULATIONS
            )
            
            if aep_results is not None:
                result = {
                    'combination_id': i,
                    'type': combination['type'],
                    'description': combination['description'],
                    'features': combination['features'],
                    'thresholds': best_thresholds,
                    'f1_score': best_score,
                    **aep_results['summary']  # Include ALL the enhanced metrics
                }
                best_results.append(result)
                print(f"    ✅ Mean loss: ${result['mean_loss']:,.0f}")
            else:
                print(f"    ❌ Failed")
                
        except Exception as e:
            print(f"    ❌ Error: {e}")

    # Process and save results (enhanced version)
    if best_results:
        results_df = pd.DataFrame(best_results).sort_values('mean_loss')
        
        # Get the best result for detailed outputs
        best_result = results_df.iloc[0]
        
        # Save enhanced summary with ALL metrics
        summary_path = os.path.join(results_dir, f'enhanced_multi_rule_complete_summary_{timestamp}.csv')
        results_df.to_csv(summary_path, index=False)
        
        # NEW: Save individual outputs for the best rule (matching Single Rule format)
        best_combination_idx = best_result['combination_id']
        best_combination = double_combinations[best_combination_idx]
        
        # Re-run the best combination to get detailed outputs
        print(f"\n🎯 Generating detailed outputs for best combination...")
        best_thresholds = best_result['thresholds']
        
        detailed_aep_results = calculate_enhanced_multi_rule_aep(
            df, best_combination['features'], best_combination['operators'], 
            best_combination['logic_ops'], best_thresholds, N_PARAM, W_PARAM, 
            config.MIN_DAYS, n_simulations=config.N_SIMULATIONS
        )
        
        if detailed_aep_results:
            # Save AEP curve (matching Single Rule format)
            curve_path = os.path.join(results_dir, f'multi_rule_aep_curve_{timestamp}.csv')
            detailed_aep_results['aep_curve'].to_csv(curve_path, index=False)
            
            # Save detailed summary (matching Single Rule format)
            detailed_summary_path = os.path.join(results_dir, f'multi_rule_aep_summary_{timestamp}.csv')
            pd.DataFrame([detailed_aep_results['summary']]).to_csv(detailed_summary_path, index=False)
            
            # Save observed yearly losses if available
            if observed_events is not None:
                obs_yearly_losses = {}
                df_with_date = df.set_index('date')
                years = df_with_date.index.year
                observed_aligned = observed_events.reindex(df_with_date.index, fill_value=0)
                
                for year in np.unique(years):
                    mask = (years == year)
                    obs_loss = calculate_annual_loss_jit(
                        observed_aligned.values[mask], N_PARAM, W_PARAM, config.MIN_DAYS
                    )
                    obs_yearly_losses[int(year)] = float(obs_loss)
                
                obs_losses_df = pd.DataFrame(
                    list(obs_yearly_losses.items()), 
                    columns=['year', 'observed_loss']
                )
                obs_losses_path = os.path.join(results_dir, f'multi_rule_observed_yearly_losses_{timestamp}.csv')
                obs_losses_df.to_csv(obs_losses_path, index=False)
                print(f"✅ Saved observed yearly losses: {obs_losses_path}")
            
            print(f"✅ Saved enhanced AEP curve: {curve_path}")
            print(f"✅ Saved enhanced AEP summary: {detailed_summary_path}")
        
        # Print comprehensive results
        print(f"\n🏆 TOP 5 ENHANCED MULTI-RULE COMBINATIONS:")
        print("=" * 70)
        for i, (_, row) in enumerate(results_df.head(5).iterrows(), 1):
            print(f"{i}. {row['type']}: {row['description']}")
            print(f"   F1: {row['f1_score']:.3f} | Mean Loss: ${row['mean_loss']:,.0f}")
            print(f"   Events: {row['mean_events']:.1f} | Zero Prob: {row['zero_prob']:.1%}")
            print(f"   CM: TP={row['mean_tp']:.1f}, FP={row['mean_fp']:.1f}, FN={row['mean_fn']:.1f}")
            print()
        
        print(f"✅ Results saved: {summary_path}")
        return results_df
    else:
        print("❌ No successful analyses")
        return None

# USAGE: Replace the call in __main__ section:
# if __name__ == "__main__":
#     enhanced_multi_rule_main_keeping_working_parts()

# USAGE: Replace the call to enhanced_fast_multi_rule_main() in your main execution
# with enhanced_multi_rule_main_with_full_aep()
# if __name__ == "__main__":
#     main()

if __name__ == "__main__":
    # enhanced_fast_multi_rule_main()
    enhanced_multi_rule_main_keeping_working_parts()

