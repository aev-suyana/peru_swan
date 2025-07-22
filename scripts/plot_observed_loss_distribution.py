"""
Plot Observed Loss Distribution (Simplified)
===========================================

This script plots the observed loss distribution for the run specified in the config file.
Simplified version with only: losses by year, density plot, and summary table.

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

# Import centralized configuration
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from config import config
    print("✅ Using centralized configuration")
except ImportError as e:
    print(f"❌ Error: Cannot import config: {e}")
    exit(1)

def find_latest_observed_losses_file(run_dir):
    """Find the most recent observed yearly losses CSV file"""
    pattern = os.path.join(run_dir, '*observed_yearly_losses*.csv')
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"❌ No observed yearly losses files found in {run_dir}")
        return None
    
    # Return the most recent file
    latest_file = files[-1]
    print(f"📅 Found observed losses file: {os.path.basename(latest_file)}")
    return latest_file

def load_observed_losses(file_path):
    """Load and validate observed losses data"""
    try:
        df = pd.read_csv(file_path)
        
        # Validate required columns
        if 'year' not in df.columns or 'observed_loss' not in df.columns:
            print(f"❌ Invalid file format: missing required columns")
            print(f"   Available columns: {list(df.columns)}")
            return None
        
        # Convert year to int and observed_loss to float
        df['year'] = df['year'].astype(int)
        df['observed_loss'] = df['observed_loss'].astype(float)
        
        # Sort by year
        df = df.sort_values('year').reset_index(drop=True)
        
        print(f"✅ Loaded {len(df)} years of observed losses")
        print(f"   Year range: {df['year'].min()} - {df['year'].max()}")
        print(f"   Loss range: ${df['observed_loss'].min():,.0f} - ${df['observed_loss'].max():,.0f}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading observed losses: {e}")
        return None

def calculate_loss_statistics(df):
    """Calculate comprehensive loss statistics"""
    stats = {
        'n_years': len(df),
        'total_loss': df['observed_loss'].sum(),
        'mean_loss': df['observed_loss'].mean(),
        'median_loss': df['observed_loss'].median(),
        'std_loss': df['observed_loss'].std(),
        'min_loss': df['observed_loss'].min(),
        'max_loss': df['observed_loss'].max(),
        'p25_loss': df['observed_loss'].quantile(0.25),
        'p75_loss': df['observed_loss'].quantile(0.75),
        'p90_loss': df['observed_loss'].quantile(0.90),
        'p95_loss': df['observed_loss'].quantile(0.95),
        'p99_loss': df['observed_loss'].quantile(0.99),
        'zero_years': (df['observed_loss'] == 0).sum(),
        'high_loss_years': (df['observed_loss'] > df['observed_loss'].quantile(0.75)).sum(),
        'extreme_loss_years': (df['observed_loss'] > df['observed_loss'].quantile(0.90)).sum(),
    }
    
    # Calculate loss per year statistics
    stats['mean_loss_per_year'] = stats['total_loss'] / stats['n_years']
    
    # Calculate coefficient of variation (CV)
    if stats['mean_loss'] > 0:
        stats['cv_loss'] = stats['std_loss'] / stats['mean_loss']
    else:
        stats['cv_loss'] = np.nan
    
    return stats

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

def calculate_aep_statistics(simulated_losses, method_name):
    """Calculate comprehensive statistics for AEP method (non-negative only)"""
    if simulated_losses is None or len(simulated_losses) == 0:
        return None
    
    # Filter out negative values
    positive_losses = simulated_losses[simulated_losses >= 0]
    
    if len(positive_losses) == 0:
        print(f"⚠️ Warning: All {method_name} losses are negative or zero")
        return None
    
    stats = {
        'method': method_name,
        'n_simulations': len(simulated_losses),
        'n_positive': len(positive_losses),
        'mean_loss': float(np.mean(positive_losses)),
        'median_loss': float(np.median(positive_losses)),
        'std_loss': float(np.std(positive_losses)),
        'min_loss': float(np.min(positive_losses)),
        'max_loss': float(np.max(positive_losses)),
        'p25_loss': float(np.percentile(positive_losses, 25)),
        'p75_loss': float(np.percentile(positive_losses, 75)),
        'p90_loss': float(np.percentile(positive_losses, 90)),
        'p95_loss': float(np.percentile(positive_losses, 95)),
        'p99_loss': float(np.percentile(positive_losses, 99)),
        'zero_prob': float(np.mean(positive_losses == 0)),
        'cv_loss': float(np.std(positive_losses) / np.mean(positive_losses)) if np.mean(positive_losses) > 0 else 0
    }
    
    return stats

def plot_simplified_loss_analysis(df, stats, run_path, port_name):
    """Create individual plots: losses by year, density plot, histogram, summary table, and overlay plot"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    output_dir = os.path.join(config.results_output_dir)
    
    # Load simulated losses for overlay plot
    aep_file = find_latest_aep_files(output_dir)
    simulated_losses = None
    if aep_file:
        simulated_losses = load_all_simulated_losses(aep_file)
    
    # 1. Losses by year (bar chart) - Individual plot
    plt.figure(figsize=(10, 6))
    years = df['year'].astype(str)
    losses = df['observed_loss']
    
    bars = plt.bar(years, losses, color='skyblue', alpha=0.7, edgecolor='black')
    
    # Color bars based on loss level
    for i, (bar, loss) in enumerate(zip(bars, losses)):
        if loss <= stats['p25_loss']:
            bar.set_color('lightgreen')
        elif loss <= stats['p75_loss']:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.axhline(stats['mean_loss'], color='red', linestyle='--', alpha=0.8, label=f'Mean: ${stats["mean_loss"]:,.0f}')
    plt.axhline(stats['p99_loss'], color='purple', linestyle=':', alpha=0.8, label=f'P99: ${stats["p99_loss"]:,.0f}')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Observed Loss ($)', fontsize=12)
    plt.title(f'Observed Losses by Year\n{port_name} ({run_path})', fontsize=14, fontweight='bold')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    plt.tight_layout()
    
    # Save individual bar chart
    bar_plot_path = os.path.join(output_dir, 'observed_losses_by_year.png')
    plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved bar chart: {bar_plot_path}")
    
    # 2. Density plot - Individual plot
    plt.figure(figsize=(10, 6))
    # Filter out negative values for density plot
    observed_losses_positive = df['observed_loss'][df['observed_loss'] >= 0]
    if len(observed_losses_positive) > 0:
        sns.kdeplot(observed_losses_positive, fill=True, alpha=0.6, color='blue', linewidth=2)
    plt.axvline(stats['mean_loss'], color='red', linestyle='--', alpha=0.8, label=f'Mean: ${stats["mean_loss"]:,.0f}')
    plt.axvline(stats['median_loss'], color='green', linestyle='--', alpha=0.8, label=f'Median: ${stats["median_loss"]:,.0f}')
    plt.axvline(stats['p99_loss'], color='purple', linestyle=':', alpha=0.8, label=f'P99: ${stats["p99_loss"]:,.0f}')
    plt.xlabel('Observed Loss ($)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'Loss Distribution Density\n{port_name} ({run_path})', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    plt.tight_layout()
    
    # Save individual density plot
    density_plot_path = os.path.join(output_dir, 'observed_loss_density.png')
    plt.savefig(density_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved density plot: {density_plot_path}")
    
    # 3. Histogram - Individual plot
    plt.figure(figsize=(10, 6))
    # Filter out negative values for histogram
    observed_losses_positive = df['observed_loss'][df['observed_loss'] >= 0]
    if len(observed_losses_positive) > 0:
        plt.hist(observed_losses_positive, bins=min(7, len(observed_losses_positive)//2), alpha=0.7, color='skyblue', edgecolor='black', density=True)
    plt.axvline(stats['mean_loss'], color='red', linestyle='--', alpha=0.8, label=f'Mean: ${stats["mean_loss"]:,.0f}')
    plt.axvline(stats['median_loss'], color='green', linestyle='--', alpha=0.8, label=f'Median: ${stats["median_loss"]:,.0f}')
    plt.axvline(stats['p99_loss'], color='purple', linestyle=':', alpha=0.8, label=f'P99: ${stats["p99_loss"]:,.0f}')
    plt.xlabel('Observed Loss ($)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'Loss Distribution Histogram\n{port_name} ({run_path})', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    plt.tight_layout()
    
    # Save individual histogram
    hist_plot_path = os.path.join(output_dir, 'observed_loss_histogram.png')
    plt.savefig(hist_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved histogram: {hist_plot_path}")
    
    # 4. Summary statistics table - Individual plot
    plt.figure(figsize=(12, 8))
    plt.axis('tight')
    plt.axis('off')
    
    # Create summary table
    summary_data = [
        ['Statistic', 'Value'],
        ['Total Years', f"{stats['n_years']}"],
        ['Total Loss', f"${stats['total_loss']:,.0f}"],
        ['Mean Loss', f"${stats['mean_loss']:,.0f}"],
        ['Median Loss', f"${stats['median_loss']:,.0f}"],
        ['Std Dev', f"${stats['std_loss']:,.0f}"],
        ['Min Loss', f"${stats['min_loss']:,.0f}"],
        ['Max Loss', f"${stats['max_loss']:,.0f}"],
        ['P25', f"${stats['p25_loss']:,.0f}"],
        ['P75', f"${stats['p75_loss']:,.0f}"],
        ['P90', f"${stats['p90_loss']:,.0f}"],
        ['P95', f"${stats['p95_loss']:,.0f}"],
        ['P99', f"${stats['p99_loss']:,.0f}"],
        ['CV', f"{stats['cv_loss']:.2f}"],
        ['Zero Years', f"{stats['zero_years']}"],
        ['High Loss Years', f"{stats['high_loss_years']}"],
        ['Extreme Loss Years', f"{stats['extreme_loss_years']}"]
    ]
    
    table = plt.table(cellText=summary_data, cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style the table
    for i in range(len(summary_data)):
        for j in range(len(summary_data[0])):
            if i == 0:  # Header row
                table[(i, j)].set_facecolor('#4CAF50')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    plt.title(f'Summary Statistics\n{port_name} ({run_path})', fontsize=14, fontweight='bold', pad=20)
    
    # Save individual summary table
    table_plot_path = os.path.join(output_dir, 'observed_loss_summary_table.png')
    plt.savefig(table_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved summary table: {table_plot_path}")
    
    # 5. NEW: Comprehensive overlay plot of observed vs all simulated losses
    if simulated_losses:
        # Calculate statistics for all methods
        aep_stats = {}
        for method, sim_losses in simulated_losses.items():
            method_stats = calculate_aep_statistics(sim_losses, method)
            if method_stats:
                aep_stats[method] = method_stats
        
        # Create comprehensive overlay plot
        plt.figure(figsize=(14, 10))
        
        # Color scheme for different methods
        colors = {
            'simple': 'red',
            'ml': 'green', 
            'multi': 'orange'
        }
        
        # Filter out negative values for observed losses
        observed_losses_positive = df['observed_loss'][df['observed_loss'] >= 0]
        print(f"📊 Observed losses: {len(df['observed_loss'])} total, {len(observed_losses_positive)} non-negative")
        
        # Plot observed losses density (non-negative only)
        if len(observed_losses_positive) > 0:
            sns.kdeplot(observed_losses_positive, fill=True, alpha=0.3, color='blue', linewidth=3, label='Observed Losses')
        
        # Plot simulated losses density for each method (non-negative only)
        for method, sim_losses in simulated_losses.items():
            color = colors.get(method, 'gray')
            # Filter out negative values
            sim_losses_positive = sim_losses[sim_losses >= 0]
            print(f"📊 {method.upper()} AEP: {len(sim_losses)} total, {len(sim_losses_positive)} non-negative")
            
            if len(sim_losses_positive) > 0:
                sns.kdeplot(sim_losses_positive, fill=True, alpha=0.2, color=color, linewidth=2, 
                           label=f'{method.upper()} AEP')
        
        # Add vertical lines for observed statistics (non-negative only)
        obs_mean = np.mean(observed_losses_positive) if len(observed_losses_positive) > 0 else 0
        obs_median = np.median(observed_losses_positive) if len(observed_losses_positive) > 0 else 0
        obs_p99 = np.percentile(observed_losses_positive, 99) if len(observed_losses_positive) > 0 else 0
        
        plt.axvline(obs_mean, color='blue', linestyle='--', alpha=0.8, linewidth=2, 
                   label=f'Observed Mean: ${obs_mean:,.0f}')
        plt.axvline(obs_median, color='blue', linestyle='-.', alpha=0.8, linewidth=2,
                   label=f'Observed Median: ${obs_median:,.0f}')
        plt.axvline(obs_p99, color='blue', linestyle=':', alpha=0.8, linewidth=2,
                   label=f'Observed P99: ${obs_p99:,.0f}')
        
        # Add vertical lines for simulated statistics (non-negative only)
        for method, sim_losses in simulated_losses.items():
            color = colors.get(method, 'gray')
            sim_losses_positive = sim_losses[sim_losses >= 0]
            sim_mean = np.mean(sim_losses_positive) if len(sim_losses_positive) > 0 else 0
            sim_median = np.median(sim_losses_positive) if len(sim_losses_positive) > 0 else 0
            sim_p99 = np.percentile(sim_losses_positive, 99) if len(sim_losses_positive) > 0 else 0
            
            plt.axvline(sim_mean, color=color, linestyle='--', alpha=0.6, linewidth=1.5,
                       label=f'{method.upper()} Mean: ${sim_mean:,.0f}')
            plt.axvline(sim_median, color=color, linestyle='-.', alpha=0.6, linewidth=1.5,
                       label=f'{method.upper()} Median: ${sim_median:,.0f}')
            plt.axvline(sim_p99, color=color, linestyle=':', alpha=0.6, linewidth=1.5,
                       label=f'{method.upper()} P99: ${sim_p99:,.0f}')
        
        plt.xlabel('Annual Loss ($)', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.title(f'Observed vs All AEP Methods Loss Distribution\n{port_name} ({run_path})', 
                 fontsize=16, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        plt.tight_layout()
        
        # Save comprehensive overlay plot
        overlay_plot_path = os.path.join(output_dir, 'observed_vs_all_aep_methods.png')
        plt.savefig(overlay_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved comprehensive overlay plot: {overlay_plot_path}")
        
        # 6. NEW: AEP Methods Comparison Table
        if aep_stats:
            plt.figure(figsize=(16, 12))
            plt.axis('tight')
            plt.axis('off')
            
            # Create comparison table
            table_data = [['Method', 'Simulations', 'Mean', 'Median', 'Std Dev', 'P99', 'CV', 'Zero Prob']]
            
            # Add observed statistics
            obs_cv = stats['std_loss'] / stats['mean_loss'] if stats['mean_loss'] > 0 else 0
            table_data.append([
                'OBSERVED',
                f"{stats['n_years']}",
                f"${stats['mean_loss']:,.0f}",
                f"${stats['median_loss']:,.0f}",
                f"${stats['std_loss']:,.0f}",
                f"${stats['p99_loss']:,.0f}",
                f"{obs_cv:.2f}",
                f"{stats['zero_years']/stats['n_years']:.1%}"
            ])
            
            # Add simulated statistics for each method
            for method, method_stats in aep_stats.items():
                table_data.append([
                    method.upper(),
                    f"{method_stats['n_simulations']:,}",
                    f"${method_stats['mean_loss']:,.0f}",
                    f"${method_stats['median_loss']:,.0f}",
                    f"${method_stats['std_loss']:,.0f}",
                    f"${method_stats['p99_loss']:,.0f}",
                    f"{method_stats['cv_loss']:.2f}",
                    f"{method_stats['zero_prob']:.1%}"
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
            
            plt.title(f'AEP Methods Comparison\n{port_name} ({run_path})', 
                     fontsize=16, fontweight='bold', pad=20)
            
            # Save comparison table
            comparison_table_path = os.path.join(output_dir, 'aep_methods_comparison_table.png')
            plt.savefig(comparison_table_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved AEP methods comparison table: {comparison_table_path}")
            
            return {
                'bar_chart': bar_plot_path,
                'density_plot': density_plot_path,
                'histogram': hist_plot_path,
                'summary_table': table_plot_path,
                'overlay_plot': overlay_plot_path,
                'comparison_table': comparison_table_path
            }
        else:
            return {
                'bar_chart': bar_plot_path,
                'density_plot': density_plot_path,
                'histogram': hist_plot_path,
                'summary_table': table_plot_path,
                'overlay_plot': overlay_plot_path
            }
    else:
        print("⚠️ No simulated losses found - skipping overlay plots")
        return {
            'bar_chart': bar_plot_path,
            'density_plot': density_plot_path,
            'histogram': hist_plot_path,
            'summary_table': table_plot_path
        }



def print_summary_report(df, stats, port_name, run_path):
    """Print a comprehensive summary report"""
    
    print("\n" + "="*80)
    print(f"OBSERVED LOSS DISTRIBUTION ANALYSIS")
    print("="*80)
    print(f"Port: {port_name}")
    print(f"Run: {run_path}")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*80)
    
    print(f"\n📊 BASIC STATISTICS:")
    print(f"   Total Years Analyzed: {stats['n_years']}")
    print(f"   Total Economic Loss: ${stats['total_loss']:,.0f}")
    print(f"   Average Loss per Year: ${stats['mean_loss']:,.0f}")
    print(f"   Median Loss: ${stats['median_loss']:,.0f}")
    print(f"   Standard Deviation: ${stats['std_loss']:,.0f}")
    print(f"   Coefficient of Variation: {stats['cv_loss']:.2f}")
    
    print(f"\n📈 PERCENTILE ANALYSIS:")
    print(f"   Minimum Loss: ${stats['min_loss']:,.0f}")
    print(f"   25th Percentile: ${stats['p25_loss']:,.0f}")
    print(f"   75th Percentile: ${stats['p75_loss']:,.0f}")
    print(f"   90th Percentile: ${stats['p90_loss']:,.0f}")
    print(f"   95th Percentile: ${stats['p95_loss']:,.0f}")
    print(f"   99th Percentile: ${stats['p99_loss']:,.0f}")
    print(f"   Maximum Loss: ${stats['max_loss']:,.0f}")
    
    print(f"\n🎯 LOSS CATEGORIES:")
    print(f"   Zero Loss Years: {stats['zero_years']} ({stats['zero_years']/stats['n_years']*100:.1f}%)")
    print(f"   High Loss Years (>P75): {stats['high_loss_years']} ({stats['high_loss_years']/stats['n_years']*100:.1f}%)")
    print(f"   Extreme Loss Years (>P90): {stats['extreme_loss_years']} ({stats['extreme_loss_years']/stats['n_years']*100:.1f}%)")
    
    print(f"\n📅 YEARLY BREAKDOWN:")
    for _, row in df.iterrows():
        year = row['year']
        loss = row['observed_loss']
        category = "High" if loss > stats['p75_loss'] else "Medium" if loss > stats['p25_loss'] else "Low"
        print(f"   {year}: ${loss:,.0f} ({category})")
    
    print("\n" + "="*80)

def main():
    """Main function to run the observed loss distribution analysis"""
    
    print("🔍 OBSERVED LOSS DISTRIBUTION ANALYSIS (SIMPLIFIED)")
    print("="*60)
    
    # Get configuration
    run_path = config.RUN_PATH
    port_name = config.reference_port
    
    print(f"📍 Analyzing: {port_name} ({run_path})")
    
    # Find the run directory (results directory, not processed data directory)
    run_dir = config.results_output_dir
    if not os.path.exists(run_dir):
        print(f"❌ Results directory not found: {run_dir}")
        return
    
    print(f"📁 Results directory: {run_dir}")
    
    # Find and load observed losses data
    obs_file = find_latest_observed_losses_file(run_dir)
    if obs_file is None:
        return
    
    df = load_observed_losses(obs_file)
    if df is None:
        return
    
    # Calculate statistics
    stats = calculate_loss_statistics(df)
    
    # Print summary report
    print_summary_report(df, stats, port_name, run_path)
    
    # Create plots
    print("\n📊 Creating individual plots...")
    plot_paths = plot_simplified_loss_analysis(df, stats, run_path, port_name)
    
    # Save summary statistics as CSV
    print("\n💾 Saving summary statistics...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.results_output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    summary_filename = f'observed_loss_summary_{timestamp}.csv'
    summary_path = os.path.join(output_dir, summary_filename)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame([{
        'run_path': run_path,
        'port_name': port_name,
        'analysis_date': timestamp,
        **stats
    }])
    
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Saved summary statistics: {summary_path}")
    
    print("\n🎉 Analysis completed successfully!")
    print(f"📊 Individual plots created:")
    for plot_type, plot_path in plot_paths.items():
        print(f"   • {plot_type}: {os.path.basename(plot_path)}")
    print(f"📋 Summary: {os.path.basename(summary_path)}")



if __name__ == "__main__":
    main() 