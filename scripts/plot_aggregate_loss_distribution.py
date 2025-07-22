"""
Aggregate Loss Distribution Analysis
===================================

This script aggregates observed and simulated losses from all 10 regions
to create comprehensive aggregate plots and statistics.

Author: Wave Analysis Team
Date: 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def find_latest_aep_files(run_dir):
    """Find the most recent AEP curve CSV files for all three methods"""
    files = {}
    
    # Simple condition AEP
    simple_pattern = os.path.join(run_dir, 'aep_curve_*.csv')
    simple_files = sorted(glob.glob(simple_pattern))
    if simple_files:
        files['simple'] = simple_files[-1]
        print(f"📈 Found simple AEP: {os.path.basename(files['simple'])}")
    
    # ML AEP
    ml_pattern = os.path.join(run_dir, 'ml_aep_curve_*.csv')
    ml_files = sorted(glob.glob(ml_pattern))
    if ml_files:
        files['ml'] = ml_files[-1]
        print(f"🤖 Found ML AEP: {os.path.basename(files['ml'])}")
    
    # Multi-condition AEP
    multi_pattern = os.path.join(run_dir, 'multi_rule_aep_curve_*.csv')
    multi_files = sorted(glob.glob(multi_pattern))
    if multi_files:
        files['multi'] = multi_files[-1]
        print(f"🔗 Found multi-condition AEP: {os.path.basename(files['multi'])}")
    
    return files

def load_all_simulated_losses(file_dict):
    """Load and extract simulated losses from all AEP curve files"""
    simulated_data = {}
    
    for method, file_path in file_dict.items():
        try:
            df = pd.read_csv(file_path)
            
            # Validate required columns
            if 'loss' not in df.columns or 'probability' not in df.columns:
                print(f"❌ Invalid AEP curve format for {method}: missing required columns")
                continue
            
            # Extract the loss values (these are the simulated annual losses)
            simulated_losses = df['loss'].values
            simulated_data[method] = simulated_losses
            
            print(f"✅ Loaded {method.upper()} AEP: {len(simulated_losses)} simulations")
            print(f"   Loss range: ${simulated_losses.min():,.0f} - ${simulated_losses.max():,.0f}")
            print(f"   Mean: ${simulated_losses.mean():,.0f}")
            
        except Exception as e:
            print(f"❌ Error loading {method} AEP: {e}")
    
    return simulated_data

def get_port_name_from_run(run_path):
    """Get port name from run path"""
    port_mapping = {
        'run_g1': 'CALETA_GRAU',
        'run_g2': 'CALETA_LA_CRUZ',
        'run_g3': 'COLAN',
        'run_g4': 'BAYOVAR',
        'run_g5': 'CHICAMA',
        'run_g6': 'CHIMBOTE',
        'run_g7': 'CASMA',
        'run_g8': 'HUARMEY',
        'run_g9': 'SUPE',
        'run_g10': 'DPA_VILA_VILA'
    }
    return port_mapping.get(run_path, run_path)

def aggregate_all_regions_analysis():
    """Aggregate observed and simulated losses from all 10 regions"""
    print("\n🌍 AGGREGATE ANALYSIS: All 10 Regions")
    print("=" * 50)
    
    # Define all run paths
    run_paths = [f'run_g{i}' for i in range(1, 11)]
    
    # Storage for data by region and year
    region_observed_data = {}
    region_simulated_data = {'simple': {}, 'ml': {}, 'multi': {}}
    region_stats = {}
    
    # Collect data from each region
    for run_path in run_paths:
        print(f"\n📊 Processing {run_path}...")
        
        # Find results directory for this run
        results_dir = os.path.join('results', 'cv_results', run_path)
        
        if not os.path.exists(results_dir):
            print(f"   ❌ Results directory not found: {results_dir}")
            continue
        
        # Find observed losses file
        obs_pattern = os.path.join(results_dir, 'observed_yearly_losses_*.csv')
        obs_files = sorted(glob.glob(obs_pattern))
        
        if not obs_files:
            print(f"   ❌ No observed losses found for {run_path}")
            continue
            
        # Load observed losses
        obs_file = obs_files[-1]
        try:
            obs_df = pd.read_csv(obs_file)
            if 'observed_loss' in obs_df.columns:
                region_observed = obs_df['observed_loss'].values
                region_observed_data[run_path] = region_observed
                region_stats[run_path] = {
                    'n_years': len(region_observed),
                    'mean_loss': np.mean(region_observed),
                    'total_loss': np.sum(region_observed),
                    'port_name': get_port_name_from_run(run_path)
                }
                print(f"   ✅ Loaded {len(region_observed)} observed losses")
            else:
                print(f"   ❌ Invalid observed losses format for {run_path}")
                continue
        except Exception as e:
            print(f"   ❌ Error loading observed losses for {run_path}: {e}")
            continue
        
        # Find AEP files for this region
        aep_files = find_latest_aep_files(results_dir)
        
        if not aep_files:
            print(f"   ⚠️ No AEP files found for {run_path}")
            continue
        
        # Load simulated losses for each method
        simulated_data = load_all_simulated_losses(aep_files)
        
        for method, sim_losses in simulated_data.items():
            if sim_losses is not None:
                region_simulated_data[method][run_path] = sim_losses
                print(f"   ✅ Added {len(sim_losses)} {method.upper()} simulations")
    
    # Now aggregate by summing across regions for each year
    print(f"\n🔗 Aggregating losses across all regions...")
    
    # Get the number of years (should be the same for all regions)
    n_years = len(list(region_observed_data.values())[0]) if region_observed_data else 0
    print(f"   Number of years: {n_years}")
    
    # Aggregate observed losses by year (sum across all regions)
    aggregated_observed = np.zeros(n_years)
    for region_losses in region_observed_data.values():
        aggregated_observed += region_losses
    
    # Aggregate simulated losses by year (sum across all regions for each simulation)
    aggregated_simulated = {'simple': [], 'ml': [], 'multi': []}
    
    for method in ['simple', 'ml', 'multi']:
        if method in region_simulated_data and region_simulated_data[method]:
            # Get the number of simulations (should be the same for all regions)
            n_sims = len(list(region_simulated_data[method].values())[0])
            print(f"   {method.upper()} simulations per region: {n_sims}")
            
            # For each simulation, sum across all regions
            for sim_idx in range(n_sims):
                total_loss = 0
                for region_losses in region_simulated_data[method].values():
                    total_loss += region_losses[sim_idx]
                aggregated_simulated[method].append(total_loss)
    
    # Convert to numpy arrays
    aggregated_observed = np.array(aggregated_observed)
    for method in aggregated_simulated:
        aggregated_simulated[method] = np.array(aggregated_simulated[method])
    
    print(f"\n📈 AGGREGATE SUMMARY:")
    print(f"   Total observed years: {len(aggregated_observed)}")
    print(f"   Total observed loss: ${aggregated_observed.sum():,.0f}")
    print(f"   Mean observed loss: ${aggregated_observed.mean():,.0f}")
    
    for method, sim_losses in aggregated_simulated.items():
        if len(sim_losses) > 0:
            print(f"   {method.upper()} simulations: {len(sim_losses):,}")
            print(f"   {method.upper()} mean loss: ${sim_losses.mean():,.0f}")
    
    # Create aggregate plots
    create_aggregate_plots(aggregated_observed, aggregated_simulated, region_stats)

def create_aggregate_plots(observed_losses, simulated_losses, region_stats):
    """Create comprehensive aggregate plots for all regions"""
    
    # Filter out negative values
    observed_positive = observed_losses[observed_losses >= 0]
    simulated_positive = {}
    for method, losses in simulated_losses.items():
        simulated_positive[method] = losses[losses >= 0]
    
    # Debug: Check for any negative values
    if len(observed_losses) != len(observed_positive):
        print(f"   ⚠️ Filtered out {len(observed_losses) - len(observed_positive)} negative observed losses")
    
    for method, losses in simulated_losses.items():
        if len(losses) != len(simulated_positive[method]):
            print(f"   ⚠️ Filtered out {len(losses) - len(simulated_positive[method])} negative {method.upper()} losses")
    
    print(f"\n📊 Creating aggregate plots...")
    print(f"   Observed: {len(observed_losses)} total, {len(observed_positive)} non-negative")
    for method, losses in simulated_positive.items():
        print(f"   {method.upper()}: {len(simulated_losses[method])} total, {len(losses)} non-negative")
    
    # Additional check: ensure no negative values in filtered data
    if len(observed_positive) > 0 and np.any(observed_positive < 0):
        print(f"   ❌ ERROR: Found negative values in filtered observed data!")
        observed_positive = observed_positive[observed_positive >= 0]
    
    for method, losses in simulated_positive.items():
        if len(losses) > 0 and np.any(losses < 0):
            print(f"   ❌ ERROR: Found negative values in filtered {method.upper()} data!")
            simulated_positive[method] = losses[losses >= 0]
    
    # Set up plotting
    plt.style.use('default')
    sns.set_palette("husl")
    
    output_dir = os.path.join('results', 'cv_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Color scheme
    colors = {
        'simple': 'red',
        'ml': 'green', 
        'multi': 'orange'
    }
    
    # 1. Simple Observed Losses Density Plot
    plt.figure(figsize=(12, 8))
    
    if len(observed_positive) > 0:
        sns.kdeplot(observed_positive, fill=True, alpha=0.6, color='blue', linewidth=3, label='Observed Losses')
        
        # Add vertical lines for key statistics
        obs_mean = np.mean(observed_positive)
        obs_median = np.median(observed_positive)
        obs_p99 = np.percentile(observed_positive, 99)
        
        plt.axvline(obs_mean, color='red', linestyle='--', alpha=0.8, linewidth=2, 
                   label=f'Mean: ${obs_mean:,.0f}')
        plt.axvline(obs_median, color='green', linestyle='--', alpha=0.8, linewidth=2,
                   label=f'Median: ${obs_median:,.0f}')
        plt.axvline(obs_p99, color='purple', linestyle=':', alpha=0.8, linewidth=2,
                   label=f'P99: ${obs_p99:,.0f}')
    
    plt.xlabel('Annual Loss ($)', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title(f'Observed Loss Distribution: All 10 Regions Combined\n{len(observed_positive)} Years of Data', 
             fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    # Ensure x-axis starts from 0 (no negative losses)
    plt.xlim(left=0)
    plt.tight_layout()
    
    # Save simple observed density plot
    simple_density_path = os.path.join(output_dir, 'aggregate_observed_losses_density.png')
    plt.savefig(simple_density_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved simple observed density plot: {simple_density_path}")
    
    # 2. Yearly Loss Evolution Plot
    create_yearly_evolution_plot(region_stats, output_dir)
    
    # 3. Aggregate Density Plot (existing)
    plt.figure(figsize=(14, 10))
    
    # Plot observed losses density
    if len(observed_positive) > 0:
        sns.kdeplot(observed_positive, fill=True, alpha=0.3, color='blue', linewidth=3, label='Observed Losses (All Regions)')
    
    # Plot simulated losses density for each method
    for method, sim_losses in simulated_positive.items():
        if len(sim_losses) > 0:
            color = colors.get(method, 'gray')
            sns.kdeplot(sim_losses, fill=True, alpha=0.2, color=color, linewidth=2, 
                       label=f'{method.upper()} AEP (All Regions)')
    
    # Add vertical lines for observed statistics
    if len(observed_positive) > 0:
        obs_mean = np.mean(observed_positive)
        obs_median = np.median(observed_positive)
        obs_p99 = np.percentile(observed_positive, 99)
        
        plt.axvline(obs_mean, color='blue', linestyle='--', alpha=0.8, linewidth=2, 
                   label=f'Observed Mean: ${obs_mean:,.0f}')
        plt.axvline(obs_median, color='blue', linestyle='-.', alpha=0.8, linewidth=2,
                   label=f'Observed Median: ${obs_median:,.0f}')
        plt.axvline(obs_p99, color='blue', linestyle=':', alpha=0.8, linewidth=2,
                   label=f'Observed P99: ${obs_p99:,.0f}')
    
    # Add vertical lines for simulated statistics
    for method, sim_losses in simulated_positive.items():
        if len(sim_losses) > 0:
            color = colors.get(method, 'gray')
            sim_mean = np.mean(sim_losses)
            sim_median = np.median(sim_losses)
            sim_p99 = np.percentile(sim_losses, 99)
            
            plt.axvline(sim_mean, color=color, linestyle='--', alpha=0.6, linewidth=1.5,
                       label=f'{method.upper()} Mean: ${sim_mean:,.0f}')
            plt.axvline(sim_median, color=color, linestyle='-.', alpha=0.6, linewidth=1.5,
                       label=f'{method.upper()} Median: ${sim_median:,.0f}')
            plt.axvline(sim_p99, color=color, linestyle=':', alpha=0.6, linewidth=1.5,
                       label=f'{method.upper()} P99: ${sim_p99:,.0f}')
    
    plt.xlabel('Annual Loss ($)', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title(f'Aggregate Loss Distribution: All 10 Regions\nObserved vs All AEP Methods', 
             fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    # Ensure x-axis starts from 0 (no negative losses)
    plt.xlim(left=0)
    plt.tight_layout()
    
    # Save aggregate density plot
    aggregate_density_path = os.path.join(output_dir, 'aggregate_loss_distribution.png')
    plt.savefig(aggregate_density_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved aggregate density plot: {aggregate_density_path}")
    
    # 4. Aggregate Statistics Table (existing)
    plt.figure(figsize=(16, 12))
    plt.axis('tight')
    plt.axis('off')
    
    # Calculate aggregate statistics
    table_data = [['Method', 'Total Years/Sims', 'Mean', 'Median', 'Std Dev', 'P99', 'CV', 'Zero Prob']]
    
    # Add observed statistics
    if len(observed_positive) > 0:
        obs_cv = np.std(observed_positive) / np.mean(observed_positive) if np.mean(observed_positive) > 0 else 0
        obs_zero_prob = np.mean(observed_positive == 0)
        table_data.append([
            'OBSERVED (All Regions)',
            f"{len(observed_losses)}",
            f"${np.mean(observed_positive):,.0f}",
            f"${np.median(observed_positive):,.0f}",
            f"${np.std(observed_positive):,.0f}",
            f"${np.percentile(observed_positive, 99):,.0f}",
            f"{obs_cv:.2f}",
            f"{obs_zero_prob:.1%}"
        ])
    
    # Add simulated statistics for each method
    for method, sim_losses in simulated_positive.items():
        if len(sim_losses) > 0:
            method_cv = np.std(sim_losses) / np.mean(sim_losses) if np.mean(sim_losses) > 0 else 0
            method_zero_prob = np.mean(sim_losses == 0)
            table_data.append([
                f'{method.upper()} AEP (All Regions)',
                f"{len(simulated_losses[method]):,}",
                f"${np.mean(sim_losses):,.0f}",
                f"${np.median(sim_losses):,.0f}",
                f"${np.std(sim_losses):,.0f}",
                f"${np.percentile(sim_losses, 99):,.0f}",
                f"{method_cv:.2f}",
                f"{method_zero_prob:.1%}"
            ])
    
    table = plt.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style the table
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            if i == 0:  # Header row
                table[(i, j)].set_facecolor('#4CAF50')
                table[(i, j)].set_text_props(weight='bold', color='white')
            elif i == 1:  # Observed row
                table[(i, j)].set_facecolor('#2196F3')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:  # Simulated rows
                colors_list = ['#FF9800', '#4CAF50', '#9C27B0']  # Orange, Green, Purple
                color_idx = (i - 2) % len(colors_list)
                table[(i, j)].set_facecolor(colors_list[color_idx])
                table[(i, j)].set_text_props(weight='bold', color='white')
    
    plt.title(f'Aggregate AEP Methods Comparison\nAll 10 Regions Combined', 
             fontsize=16, fontweight='bold', pad=20)
    
    # Save aggregate comparison table
    aggregate_table_path = os.path.join(output_dir, 'aggregate_aep_methods_comparison.png')
    plt.savefig(aggregate_table_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved aggregate comparison table: {aggregate_table_path}")
    
    # 5. Region-by-Region Summary (existing)
    if region_stats:
        plt.figure(figsize=(14, 10))
        plt.axis('tight')
        plt.axis('off')
        
        region_table_data = [['Region', 'Years', 'Total Loss', 'Mean Loss', 'Port Name']]
        
        for run_path, stats in region_stats.items():
            region_table_data.append([
                run_path,
                f"{stats['n_years']}",
                f"${stats['total_loss']:,.0f}",
                f"${stats['mean_loss']:,.0f}",
                stats['port_name']
            ])
        
        region_table = plt.table(cellText=region_table_data, cellLoc='center', loc='center')
        region_table.auto_set_font_size(False)
        region_table.set_fontsize(11)
        region_table.scale(1, 2.5)
        
        # Style the region table
        for i in range(len(region_table_data)):
            for j in range(len(region_table_data[0])):
                if i == 0:  # Header row
                    region_table[(i, j)].set_facecolor('#4CAF50')
                    region_table[(i, j)].set_text_props(weight='bold', color='white')
                else:  # Data rows
                    region_table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        plt.title(f'Region-by-Region Summary\nAll 10 Regions', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Save region summary table
        region_summary_path = os.path.join(output_dir, 'region_by_region_summary.png')
        plt.savefig(region_summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved region summary table: {region_summary_path}")
    
    print(f"\n🎉 Aggregate analysis completed!")
    print(f"   Total observed years: {len(observed_losses)}")
    print(f"   Total observed loss: ${observed_losses.sum():,.0f}")
    print(f"   Files saved in: {output_dir}")

def create_yearly_evolution_plot(region_stats, output_dir):
    """Create a plot showing the evolution of yearly losses over time"""
    
    # Collect yearly data from each region
    yearly_data = []
    
    for run_path in [f'run_g{i}' for i in range(1, 11)]:
        if run_path not in region_stats:
            continue
            
        # Find results directory for this run
        results_dir = os.path.join('results', 'cv_results', run_path)
        
        # Find observed losses file
        obs_pattern = os.path.join(results_dir, 'observed_yearly_losses_*.csv')
        obs_files = sorted(glob.glob(obs_pattern))
        
        if not obs_files:
            continue
            
        # Load observed losses with years
        obs_file = obs_files[-1]
        try:
            obs_df = pd.read_csv(obs_file)
            if 'observed_loss' in obs_df.columns and 'year' in obs_df.columns:
                for _, row in obs_df.iterrows():
                    yearly_data.append({
                        'year': int(row['year']),
                        'loss': row['observed_loss'],
                        'region': run_path,
                        'port_name': region_stats[run_path]['port_name']
                    })
        except Exception as e:
            print(f"   ⚠️ Error loading yearly data for {run_path}: {e}")
            continue
    
    if not yearly_data:
        print("   ⚠️ No yearly data found for evolution plot")
        return
    
    # Convert to DataFrame
    yearly_df = pd.DataFrame(yearly_data)
    
    # Create evolution plot
    plt.figure(figsize=(16, 10))
    
    # Plot each region with different colors
    regions = yearly_df['region'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(regions)))
    
    for i, region in enumerate(regions):
        region_data = yearly_df[yearly_df['region'] == region]
        port_name = region_data['port_name'].iloc[0]
        
        plt.plot(region_data['year'], region_data['loss'], 
                marker='o', linewidth=2, markersize=6, 
                color=colors[i], label=f'{region} ({port_name})', alpha=0.8)
    
    # Add overall trend line
    yearly_df_sorted = yearly_df.sort_values('year')
    overall_trend = yearly_df_sorted.groupby('year')['loss'].sum().reset_index()
    plt.plot(overall_trend['year'], overall_trend['loss'], 
            color='black', linewidth=3, linestyle='--', 
            label='Total Loss (All Regions)', alpha=0.9)
    
    # Add mean line
    mean_loss = yearly_df.groupby('year')['loss'].mean()
    plt.plot(mean_loss.index, mean_loss.values, 
            color='red', linewidth=2, linestyle=':', 
            label='Mean Loss (All Regions)', alpha=0.8)
    
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Annual Loss ($)', fontsize=14)
    plt.title(f'Evolution of Yearly Losses: All 10 Regions\n{len(yearly_df)} Data Points', 
             fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save yearly evolution plot
    evolution_path = os.path.join(output_dir, 'yearly_losses_evolution.png')
    plt.savefig(evolution_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved yearly evolution plot: {evolution_path}")

if __name__ == "__main__":
    aggregate_all_regions_analysis() 