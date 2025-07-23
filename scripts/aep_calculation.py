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
            if has_observed:
                observed_vals = observed_matrix[sim_indices]
                fp_cost, fn_cost, tp_cost = calculate_cm_costs_jit(predicted_events, observed_vals, N, W, min_days)
                batch_fp_costs.append(fp_cost)
                batch_fn_costs.append(fn_cost)
                batch_tp_costs.append(tp_cost)
                fp_events = len(find_events_in_category_jit(predicted_events, observed_vals, 0))
                fn_events = len(find_events_in_category_jit(predicted_events, observed_vals, 1))
                tp_events = len(find_events_in_category_jit(predicted_events, observed_vals, 2))
                batch_event_counts.append({'fp_events': fp_events, 'fn_events': fn_events, 'tp_events': tp_events})
        except Exception as e:
            batch_losses.append(0)
            if has_observed:
                batch_fp_costs.append(0)
                batch_fn_costs.append(0)
                batch_tp_costs.append(0)
                batch_event_counts.append({'fp_events': 0, 'fn_events': 0, 'tp_events': 0})
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
            'raw_fp_losses': fp_annual_costs,
            'raw_fn_losses': fn_annual_costs,
            'raw_tp_losses': tp_annual_costs,
            'raw_insurance_losses': insurance_annual_costs
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

# --- Helper: Find best single rule and mean threshold ---
def load_best_single_rule_and_threshold(cv_results_path, folds_dir):
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

    # Now get the mean threshold for this rule from fold_thresholds file
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
        threshold = np.nanmean(folds[feature].values)
    else:
        raise ValueError(f"No threshold column for feature '{feature}' found in fold thresholds file. Columns are: {list(folds.columns)}")
    return feature, threshold

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

# --- MAIN EXECUTION ---
def main():
    print("\n🔧 AEP_CALCULATION.PY - Final AEP Analysis")
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

    # --- Select best rule and threshold ---
    feature, threshold = load_best_single_rule_and_threshold(cv_results_path, results_dir)
    print(f"✅ Best single rule: {feature} > {threshold:.3f}")

    # --- Optional: Plot rule condition for validation ---
    if ENABLE_PLOTTING:
        plot_path = os.path.join(results_dir, f'best_rule_plot_{config.RUN_PATH}.png')
        plot_rule_condition(df, feature, threshold, save_path=plot_path)

    # --- Run AEP simulation (speed-optimized, ported logic) ---
    print("\n🚀 Running speed-optimized AEP simulation...")
    # Prepare observed events column if available
    observed_col = 'event_dummy_1' if 'event_dummy_1' in df.columns else None
    observed_events = df.set_index('date')[observed_col] if observed_col else None
    swh_data = df.set_index('date')
    aep_results = calculate_unified_aep_analysis_fast(
        swh_data,
        trigger_feature=feature,
        trigger_threshold=threshold,
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
        print("❌ AEP simulation failed.")
    # --- Robust: Find latest ML probabilities and threshold files ---
    import glob
    results_root = os.path.join(os.getcwd(), 'results', 'cv_results')
    ml_probs_candidates = sorted(glob.glob(os.path.join(results_root, '**', 'ML_probs_2024.csv'), recursive=True), key=os.path.getmtime, reverse=True)
    if not ml_probs_candidates:
        print("❌ No ML_probs_2024.csv found in results/cv_results.")
        ml_results = None
    else:
        ml_probs_path = ml_probs_candidates[0]
        ml_probs_dir = os.path.dirname(ml_probs_path)
        ml_threshold_path = os.path.join(ml_probs_dir, 'ML_probs_2024_optimal_threshold.txt')
        print(f"✅ Using ML probabilities: {ml_probs_path}")
        print(f"✅ Using ML optimal threshold: {ml_threshold_path}")

        if not os.path.exists(ml_probs_path):
            print(f"❌ ML probabilities file not found: {ml_probs_path}")
            ml_results = None
        else:
            ml_probs = pd.read_csv(ml_probs_path, parse_dates=['date'])
            df_ml = df.set_index('date').join(ml_probs.set_index('date')[['calibrated_probability']], how='left')
            df_ml = df_ml.dropna(subset=['calibrated_probability'])

            # --- Load optimal ML threshold from file ---
            if not os.path.exists(ml_threshold_path):
                print(f"❌ ML optimal threshold file not found: {ml_threshold_path}")
                print("    Please run rule_evaluation.py to generate this file.")
                return
            with open(ml_threshold_path, 'r') as f:
                ml_trigger_threshold = float(f.read().strip())
            print(f"✅ Loaded ML optimal threshold: {ml_trigger_threshold}")

            ml_trigger_feature = 'calibrated_probability'

            print(f"\n🚀 Running ML-based AEP simulation (logistic regression probabilities)...")
            ml_results = calculate_unified_aep_analysis_fast(
                df_ml,
                trigger_feature=ml_trigger_feature,
                trigger_threshold=ml_trigger_threshold,
                N=N_PARAM,
                W=W_PARAM,
                min_days=config.MIN_DAYS,
                n_simulations=config.N_SIMULATIONS,
                observed_events=df_ml['event_dummy_1'] if 'event_dummy_1' in df_ml.columns else None,
                block_length=config.BLOCK_LENGTH,
                window_days=config.WINDOW_DAYS,
                n_jobs=-1
            )
            if ml_results is None:
                print("❌ ML-based AEP simulation failed.")
            else:
                # Save ML results
                ml_summary_path = os.path.join(results_dir, f'ml_aep_summary_{timestamp}.csv')
                ml_curve_path = os.path.join(results_dir, f'ml_aep_curve_{timestamp}.csv')
                pd.DataFrame([ml_results['standard_summary']]).to_csv(ml_summary_path, index=False)
                pd.DataFrame(ml_results['standard_aep_curve']).to_csv(ml_curve_path, index=False)
                print(f"✅ Saved ML AEP summary: {ml_summary_path}")
                print(f"✅ Saved ML AEP curve: {ml_curve_path}")

    # --- Print final comparison summary ---
    print("\n===== FINAL AEP SIMULATION SUMMARY =====")
    print(f"Rule-based method: mean annual loss = ${aep_results['standard_summary']['mean_loss']:,.0f}")
    if ml_results is not None:
        print(f"ML-based method: mean annual loss = ${ml_results['standard_summary']['mean_loss']:,.0f}")
    else:
        print("ML-based method: No results.")

    # --- Save results ---
    summary_path = os.path.join(results_dir, f'aep_summary_{timestamp}.csv')
    aep_curve_path = os.path.join(results_dir, f'aep_curve_{timestamp}.csv')
    # Save summary
    pd.DataFrame([aep_results['standard_summary']]).to_csv(summary_path, index=False)
    print(f"✅ Saved summary: {summary_path}")
    pd.DataFrame(aep_results['standard_aep_curve']).to_csv(aep_curve_path, index=False)
    print(f"✅ Saved AEP curve: {aep_curve_path}")

    # --- Save observed yearly losses if available ---
    obs_losses = aep_results.get('obs_yearly_losses', None)
    if obs_losses is not None:
        obs_losses_df = pd.DataFrame(list(obs_losses.items()), columns=['year', 'observed_loss'])
        obs_losses_path = os.path.join(results_dir, f'observed_yearly_losses_{timestamp}.csv')
        obs_losses_df.to_csv(obs_losses_path, index=False)
        print(f"✅ Saved observed yearly losses: {obs_losses_path}")

    # --- Optional: Plot AEP curve ---
    if ENABLE_PLOTTING:
        plt.figure(figsize=(10, 6))
        plt.plot(aep_results['standard_aep_curve']['loss'], aep_results['standard_aep_curve']['probability'], marker='o')
        plt.xlabel('Annual Loss ($)')
        plt.ylabel('Exceedance Probability')
        plt.title(f'AEP Curve: {feature} > {threshold:.2f}')
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(results_dir, f'aep_curve_plot_{timestamp}.png')
        plt.savefig(plot_path)
        print(f"✅ Saved AEP curve plot: {plot_path}")
        plt.show()

    print("\n🎉 AEP calculation completed!")

if __name__ == "__main__":
    main()
