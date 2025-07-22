#%%
"""
Interactive Observed Loss Distribution Analysis
==============================================

This script provides an interactive analysis of observed loss distributions.
It loads observed yearly losses and creates various plots and analyses.

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

#%%
# Helper functions
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

#%%
# Load observed yearly losses
print("📊 Loading observed yearly losses...")
obs_file = find_latest_observed_losses_file(config.results_output_dir)
if obs_file is None:
    print("❌ No observed losses file found")
    exit(1)

df_observed = load_observed_losses(obs_file)
if df_observed is None:
    print("❌ Failed to load observed losses")
    exit(1)

print(f"✅ Loaded observed losses: {df_observed.shape}")
print(f"   Date range: {df_observed['year'].min()} - {df_observed['year'].max()}")
print(f"   Loss range: ${df_observed['observed_loss'].min():,.0f} - ${df_observed['observed_loss'].max():,.0f}")

#%%
# Calculate basic statistics
print("\n📈 Calculating basic statistics...")
stats = calculate_loss_statistics(df_observed)

print(f"✅ Calculated statistics:")
print(f"   Total years: {stats['n_years']}")
print(f"   Total loss: ${stats['total_loss']:,.0f}")
print(f"   Mean loss: ${stats['mean_loss']:,.0f}")
print(f"   Median loss: ${stats['median_loss']:,.0f}")
print(f"   P99 loss: ${stats['p99_loss']:,.0f}")

#%%
# Create plots
print("\n📊 Creating plots...")

# Plot 1: Observed losses by year
plt.figure(figsize=(12, 6))
years = df_observed['year'].astype(str)
losses = df_observed['observed_loss']

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
plt.title(f'Observed Losses by Year\n{config.reference_port} ({config.RUN_PATH})', fontsize=14, fontweight='bold')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
plt.tight_layout()
plt.show()

# Plot 2: Density plot
plt.figure(figsize=(12, 6))
observed_losses_positive = df_observed['observed_loss'][df_observed['observed_loss'] >= 0]
if len(observed_losses_positive) > 0:
    sns.kdeplot(observed_losses_positive, fill=True, alpha=0.6, color='blue', linewidth=2)
plt.axvline(stats['mean_loss'], color='red', linestyle='--', alpha=0.8, label=f'Mean: ${stats["mean_loss"]:,.0f}')
plt.axvline(stats['median_loss'], color='green', linestyle='--', alpha=0.8, label=f'Median: ${stats["median_loss"]:,.0f}')
plt.axvline(stats['p99_loss'], color='purple', linestyle=':', alpha=0.8, label=f'P99: ${stats["p99_loss"]:,.0f}')
plt.xlabel('Observed Loss ($)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title(f'Loss Distribution Density\n{config.reference_port} ({config.RUN_PATH})', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
plt.tight_layout()
plt.show()

#%%
# Print summary statistics
print("\n📋 Summary Statistics:")
print("="*50)
print(f"Total Years: {stats['n_years']}")
print(f"Total Loss: ${stats['total_loss']:,.0f}")
print(f"Mean Loss: ${stats['mean_loss']:,.0f}")
print(f"Median Loss: ${stats['median_loss']:,.0f}")
print(f"Std Dev: ${stats['std_loss']:,.0f}")
print(f"Min Loss: ${stats['min_loss']:,.0f}")
print(f"Max Loss: ${stats['max_loss']:,.0f}")
print(f"P25: ${stats['p25_loss']:,.0f}")
print(f"P75: ${stats['p75_loss']:,.0f}")
print(f"P90: ${stats['p90_loss']:,.0f}")
print(f"P95: ${stats['p95_loss']:,.0f}")
print(f"P99: ${stats['p99_loss']:,.0f}")
print(f"CV: {stats['cv_loss']:.2f}")
print(f"Zero Years: {stats['zero_years']}")
print(f"High Loss Years: {stats['high_loss_years']}")
print(f"Extreme Loss Years: {stats['extreme_loss_years']}")

#%%
# Print yearly breakdown
print("\n📅 Yearly Breakdown:")
print("="*50)
for _, row in df_observed.iterrows():
    year = row['year']
    loss = row['observed_loss']
    category = "High" if loss > stats['p75_loss'] else "Medium" if loss > stats['p25_loss'] else "Low"
    print(f"{year}: ${loss:,.0f} ({category})")

#%%
# MULTIPLE THRESHOLDS ANALYSIS
print("\n🔍 MULTIPLE THRESHOLDS ANALYSIS")
print("="*50)

# Load REAL daily event data
print("📊 Loading REAL daily event data...")

# Try multiple possible locations for the merged data
possible_paths = [
    os.path.join(config.run_output_dir, 'df_swan_waverys_merged.csv'),
    os.path.join('wave_analysis_pipeline', 'data', 'processed', config.RUN_PATH, 'df_swan_waverys_merged.csv'),
    os.path.join('data', 'processed', config.RUN_PATH, 'df_swan_waverys_merged.csv')
]

merged_path = None
for path in possible_paths:
    if os.path.exists(path):
        merged_path = path
        break

if merged_path is None:
    print("❌ No merged data file found in any expected location")
    print("⚠️ Skipping multiple thresholds analysis - no daily event data available")
else:
    print(f"✅ Found merged data: {merged_path}")
    df_merged = pd.read_csv(merged_path, parse_dates=['date'])
    print(f"✅ Loaded merged data: {df_merged.shape}")
    
    # Check if event_dummy_1 exists
    if 'event_dummy_1' not in df_merged.columns:
        print("❌ No event_dummy_1 column found in merged data")
        print("⚠️ Skipping multiple thresholds analysis")
    else:
        print(f"✅ Found event_dummy_1 column with {df_merged['event_dummy_1'].sum()} event days")
        
        # Load wage data for economic parameters
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
            df_wages['daily_losses'] = df_wages['n_fishermen'] * df_wages['daily_wages']
            
            # Get economic parameters for the reference port
            port_name = config.reference_port.upper().replace(' ', '_')
            port_wage_data = df_wages[df_wages['port_name'] == port_name]
            
            if port_wage_data.empty:
                print(f"⚠️ No wage data found for {port_name}, using median values")
                avg_wages = df_wages['daily_wages'].median()
                avg_fishermen = df_wages['n_fishermen'].median()
                avg_losses = df_wages['daily_losses'].median()
                port_wage_data = pd.DataFrame([{
                    'port_name': port_name,
                    'daily_wages': avg_wages,
                    'n_fishermen': avg_fishermen,
                    'daily_losses': avg_losses
                }])
            
            print(f"✅ Economic parameters for {port_name}:")
            print(f"   Daily wages: ${port_wage_data['daily_wages'].iloc[0]:.2f}")
            print(f"   Number of fishermen: {port_wage_data['n_fishermen'].iloc[0]:.0f}")
            print(f"   Daily losses: ${port_wage_data['daily_losses'].iloc[0]:.2f}")
            
        except Exception as e:
            print(f"⚠️ Could not load wage data: {e}")
            print("Using default economic parameters")
            port_wage_data = pd.DataFrame([{
                'port_name': config.reference_port,
                'daily_wages': 12.0,
                'n_fishermen': 1000,
                'daily_losses': 12000.0
            }])

        # Create economic DataFrame with REAL event data
        print("💰 Creating economic DataFrame with REAL event data...")
        df_economic = df_merged[['date', 'event_dummy_1']].copy()
        df_economic['N_fishermen'] = port_wage_data['n_fishermen'].iloc[0]
        df_economic['wages'] = port_wage_data['daily_wages'].iloc[0]
        df_economic['daily_losses'] = port_wage_data['daily_losses'].iloc[0]
        df_economic['event_dummy_1'] = df_economic['event_dummy_1'].fillna(0).astype(int)
        df_economic['daily_losses'] = np.where(df_economic['event_dummy_1'] == 1, df_economic['daily_losses'], 0)

        print(f"✅ Created economic DataFrame: {df_economic.shape}")
        print(f"   Total days: {len(df_economic)}")
        print(f"   Event days: {df_economic['event_dummy_1'].sum()} ({df_economic['event_dummy_1'].mean()*100:.1f}%)")

        # Create event_duration column (max duration for each consecutive event period)
        df_economic['event_duration'] = df_economic['event_dummy_1'].groupby((df_economic['event_dummy_1'] != df_economic['event_dummy_1'].shift()).cumsum()).cumsum().groupby((df_economic['event_dummy_1'] != df_economic['event_dummy_1'].shift()).cumsum()).transform('max')

        # Create event_dummy_2 through event_dummy_7 (events with duration >= respective number)
        for i in range(2, 8):
            df_economic[f'event_dummy_{i}'] = np.where((df_economic['event_dummy_1'] == 1) & (df_economic['event_duration'] >= i), 1, 0)

        # Create event_count columns (mark only first day of each event period)
        for i in range(2, 8):
            df_economic[f'event_count_{i}'] = np.where(
                (df_economic[f'event_dummy_{i}'] == 1) &
                (df_economic[f'event_dummy_{i}'].shift(1) == 0), 1, 0
            )

        # Create daily_losses columns for each event duration type
        for i in range(2, 8):
            df_economic[f'daily_losses_{i}'] = np.where(df_economic[f'event_dummy_{i}'] == 1, df_economic['daily_losses'], 0)

        # Calculate yearly losses by threshold
        print("📊 Calculating yearly losses by threshold...")
        yearly_losses_by_duration = {}
        df_economic['year'] = df_economic['date'].dt.year

        for i in range(1, 8):
            column_name = 'daily_losses' if i == 1 else f'daily_losses_{i}'
            yearly_losses = df_economic.groupby('year')[column_name].sum().reset_index()
            yearly_losses.rename(columns={column_name: 'yearly_loss'}, inplace=True)
            yearly_losses_by_duration[f'{i}day'] = yearly_losses['yearly_loss'].values

        # Create summary DataFrame for threshold comparison
        print("📋 Creating summary DataFrame for threshold comparison...")
        summary_data = []
        for i, duration in enumerate(yearly_losses_by_duration.keys()):
            losses = yearly_losses_by_duration[duration]
            non_zero_losses = losses[losses > 0]
            summary_data.append({
                'threshold': duration,
                'total_loss': np.sum(losses),
                'mean_loss': np.mean(non_zero_losses) if len(non_zero_losses) > 0 else 0,
                'median_loss': np.median(non_zero_losses) if len(non_zero_losses) > 0 else 0,
                'max_loss': np.max(losses),
                'years_with_events': len(non_zero_losses),
                'total_years': len(losses),
                'event_rate': len(non_zero_losses) / len(losses) if len(losses) > 0 else 0
            })

        df_summary = pd.DataFrame(summary_data)

        # Create threshold comparison plots
        print("📊 Creating threshold comparison plots...")

        # Plot 3: Total loss by threshold
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df_summary['threshold'], df_summary['total_loss'], color='orange', alpha=0.7, edgecolor='black')

        # Add value labels on bars
        for bar, value in zip(bars, df_summary['total_loss']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(df_summary['total_loss'])*0.01, 
                    f'${value:,.0f}', ha='center', va='bottom', fontsize=10)

        plt.xlabel('Event Duration Threshold', fontsize=12)
        plt.ylabel('Total Loss ($)', fontsize=12)
        plt.title(f'Total Loss by Event Duration Threshold\n{config.reference_port} ({config.RUN_PATH})', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        plt.tight_layout()
        plt.show()

        # Plot 4: Density plot overlay for different thresholds
        plt.figure(figsize=(12, 6))
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']

        for i, (duration, color) in enumerate(zip(yearly_losses_by_duration.keys(), colors)):
            losses = yearly_losses_by_duration[duration]
            non_zero_losses = losses[losses > 0]
            if len(non_zero_losses) > 0:
                sns.kdeplot(non_zero_losses, fill=True, alpha=0.3, color=color, linewidth=2, label=duration)
                
                # Add P99 line
                p99 = np.percentile(non_zero_losses, 99)
                plt.axvline(p99, color=color, linestyle=':', alpha=0.8, linewidth=1.5)
                plt.text(p99, plt.gca().get_ylim()[1]*0.9, f'P99: ${p99:,.0f}', 
                        rotation=90, va='top', ha='right', fontsize=8, color=color)

        plt.xlabel('Yearly Loss ($)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title(f'Loss Distribution by Event Duration Threshold\n{config.reference_port} ({config.RUN_PATH})', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        plt.tight_layout()
        plt.show()

        # Plot 5: P99/Mean ratio by threshold
        print("📊 Creating P99/Mean ratio plot...")
        ratios = []
        thresholds = []
        for duration in yearly_losses_by_duration.keys():
            losses = yearly_losses_by_duration[duration]
            non_zero_losses = losses[losses > 0]
            if len(non_zero_losses) > 0:
                mean_loss = np.mean(non_zero_losses)
                p99_loss = np.percentile(non_zero_losses, 99)
                ratio = p99_loss / mean_loss if mean_loss > 0 else 0
                ratios.append(ratio)
                thresholds.append(duration)

        plt.figure(figsize=(10, 6))
        bars = plt.bar(thresholds, ratios, color='orange', alpha=0.7, edgecolor='black')

        # Add value labels on bars
        for bar, ratio in zip(bars, ratios):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ratios)*0.01, 
                    f'{ratio:.1f}x', ha='center', va='bottom', fontsize=10)

        plt.xlabel('Event Duration Threshold', fontsize=12)
        plt.ylabel('P99/Mean Ratio', fontsize=12)
        plt.title(f'P99/Mean Ratio by Event Duration Threshold\n{config.reference_port} ({config.RUN_PATH})', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Print detailed summary
        print("\n📋 THRESHOLD ANALYSIS SUMMARY:")
        print("="*60)
        print("Threshold\tTotal Loss\tMean Loss\tEvent Rate\tP99/Mean")
        print("-"*60)
        for _, row in df_summary.iterrows():
            threshold = row['threshold']
            total_loss = row['total_loss']
            mean_loss = row['mean_loss']
            event_rate = row['event_rate']
            
            # Calculate P99/Mean ratio
            losses = yearly_losses_by_duration[threshold]
            non_zero_losses = losses[losses > 0]
            if len(non_zero_losses) > 0:
                p99 = np.percentile(non_zero_losses, 99)
                ratio = p99 / mean_loss if mean_loss > 0 else 0
            else:
                ratio = 0
            
            print(f"{threshold}\t${total_loss:,.0f}\t${mean_loss:,.0f}\t{event_rate:.1%}\t{ratio:.1f}x")

        print("\n" + "="*60)

        # Create yearly summary DataFrame by event threshold
        print("\n📊 Creating yearly summary DataFrame by event threshold...")
        
        # Initialize the yearly summary DataFrame
        years = sorted(df_economic['year'].unique())
        yearly_summary_data = []
        
        for year in years:
            year_data = df_economic[df_economic['year'] == year]
            
            row_data = {'year': year}
            
            # Calculate metrics for each threshold
            for i in range(1, 8):
                threshold = f'{i}day'
                
                # For 1-day threshold, use event_dummy_1
                if i == 1:
                    event_dummy_col = 'event_dummy_1'
                    event_count_col = 'event_dummy_1'  # Count all event days as "events"
                else:
                    event_dummy_col = f'event_dummy_{i}'
                    event_count_col = f'event_count_{i}'  # Count only first day of each event period
                
                # Total losses for this threshold
                loss_col = 'daily_losses' if i == 1 else f'daily_losses_{i}'
                total_loss = year_data[loss_col].sum()
                
                # Number of days closed (event days)
                days_closed = year_data[event_dummy_col].sum()
                
                # Number of events (count only first day of each event period)
                if i == 1:
                    # For 1-day threshold, count consecutive event periods
                    event_starts = ((year_data['event_dummy_1'] == 1) & 
                                   (year_data['event_dummy_1'].shift(1) == 0)).sum()
                    num_events = event_starts
                else:
                    num_events = year_data[event_count_col].sum()
                
                # Add to row data
                row_data[f'{threshold}_total_loss'] = total_loss
                row_data[f'{threshold}_num_events'] = num_events
                row_data[f'{threshold}_days_closed'] = days_closed
            
            yearly_summary_data.append(row_data)
        
        # Create the summary DataFrame
        df_yearly_summary = pd.DataFrame(yearly_summary_data)
        
        # Display the summary
        print(f"✅ Created yearly summary DataFrame: {df_yearly_summary.shape}")
        print(f"   Years: {df_yearly_summary['year'].min()} - {df_yearly_summary['year'].max()}")
        print(f"   Thresholds: 1day through 7day")
        
        # Print a sample of the data
        print("\n📋 YEARLY SUMMARY BY EVENT THRESHOLD (Sample):")
        print("="*100)
        
        # Create a formatted display
        display_columns = ['year']
        for i in range(1, 8):
            threshold = f'{i}day'
            display_columns.extend([
                f'{threshold}_total_loss',
                f'{threshold}_num_events', 
                f'{threshold}_days_closed'
            ])
        
        # Print header
        header = "Year"
        for i in range(1, 8):
            threshold = f'{i}day'
            header += f"\t{threshold}_Loss\t{threshold}_Events\t{threshold}_Days"
        print(header)
        print("-" * len(header))
        
        # Print first 5 rows
        for _, row in df_yearly_summary.head().iterrows():
            line = f"{row['year']}"
            for i in range(1, 8):
                threshold = f'{i}day'
                total_loss = row[f'{threshold}_total_loss']
                num_events = row[f'{threshold}_num_events']
                days_closed = row[f'{threshold}_days_closed']
                line += f"\t${total_loss:,.0f}\t{num_events}\t{days_closed}"
            print(line)
        
        print("...")
        
        # Print last 5 rows
        for _, row in df_yearly_summary.tail().iterrows():
            line = f"{row['year']}"
            for i in range(1, 8):
                threshold = f'{i}day'
                total_loss = row[f'{threshold}_total_loss']
                num_events = row[f'{threshold}_num_events']
                days_closed = row[f'{threshold}_days_closed']
                line += f"\t${total_loss:,.0f}\t{num_events}\t{days_closed}"
            print(line)
        
        # Calculate and display summary statistics
        print("\n📊 SUMMARY STATISTICS BY THRESHOLD:")
        print("="*80)
        
        summary_stats = []
        for i in range(1, 8):
            threshold = f'{i}day'
            
            total_losses = df_yearly_summary[f'{threshold}_total_loss'].sum()
            avg_loss_per_year = df_yearly_summary[f'{threshold}_total_loss'].mean()
            total_events = df_yearly_summary[f'{threshold}_num_events'].sum()
            total_days_closed = df_yearly_summary[f'{threshold}_days_closed'].sum()
            years_with_events = (df_yearly_summary[f'{threshold}_total_loss'] > 0).sum()
            avg_events_per_year = df_yearly_summary[f'{threshold}_num_events'].mean()
            avg_days_closed_per_year = df_yearly_summary[f'{threshold}_days_closed'].mean()
            
            summary_stats.append({
                'threshold': threshold,
                'total_loss': total_losses,
                'avg_loss_per_year': avg_loss_per_year,
                'total_events': total_events,
                'total_days_closed': total_days_closed,
                'years_with_events': years_with_events,
                'avg_events_per_year': avg_events_per_year,
                'avg_days_closed_per_year': avg_days_closed_per_year
            })
        
        df_threshold_stats = pd.DataFrame(summary_stats)
        
        # Print summary table
        print("Threshold\tTotal Loss\tAvg Loss/Year\tTotal Events\tTotal Days\tYears w/Events\tAvg Events/Year\tAvg Days/Year")
        print("-" * 120)
        for _, row in df_threshold_stats.iterrows():
            print(f"{row['threshold']}\t${row['total_loss']:,.0f}\t${row['avg_loss_per_year']:,.0f}\t{row['total_events']}\t{row['total_days_closed']}\t{row['years_with_events']}\t{row['avg_events_per_year']:.1f}\t{row['avg_days_closed_per_year']:.1f}")
        
        # Save the yearly summary DataFrame
        output_file = os.path.join(config.results_output_dir, f'yearly_summary_by_threshold_{config.RUN_PATH}.csv')
        df_yearly_summary.to_csv(output_file, index=False)
        print(f"\n💾 Saved yearly summary to: {output_file}")
        
        # Also save the threshold statistics
        stats_file = os.path.join(config.results_output_dir, f'threshold_statistics_{config.RUN_PATH}.csv')
        df_threshold_stats.to_csv(stats_file, index=False)
        print(f"💾 Saved threshold statistics to: {stats_file}")

#%%
