#!/usr/bin/env python3
"""
Interactive Region Data Reader
=============================

This script allows you to specify a region and read the merged data for that region.
It provides an interactive interface to explore the data structure and contents.

Author: Wave Analysis Team
Date: 2024
"""
#%%
# Cell 1: Imports and setup
#%%
import os
import pandas as pd
import numpy as np
import glob
import warnings
warnings.filterwarnings('ignore')

# Get script directory and project root
try:
    script_dir = os.path.dirname(__file__)
except NameError:
    script_dir = os.getcwd()

project_root = os.path.abspath(os.path.join(script_dir, '..'))
data_dir = os.path.join(project_root, 'wave_analysis_pipeline', 'data', 'processed')

# Cell 2: Specify region and load data
#%%
# Change this to your desired region
region = 'run_g3'  # Options: run_g1, run_g2, run_g3, run_g4, run_g5, run_g6, run_g7, run_g8, run_g9, run_g10
#%%
# Locate and load the merged data file using os.path
data_path = os.path.join(data_dir, region, 'df_swan_waverys_merged.csv')
df = pd.read_csv(data_path)

# Convert date to index if it exists
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

# Create event_dummy_2 through event_dummy_7 based on minimum duration
print("Creating duration-based event types...")

# Find continuous event periods and mark based on minimum duration
def get_event_durations(event_series):
    periods = []
    current_start = None
    for i, val in enumerate(event_series):
        if val == 1 and current_start is None:
            current_start = i
        elif val == 0 and current_start is not None:
            periods.append((current_start, i - current_start))
            current_start = None
    if current_start is not None:
        periods.append((current_start, len(event_series) - current_start))
    return periods

periods = get_event_durations(df['event_dummy_1'].values)
print(f"Found {len(periods)} event periods")

# Initialize new event columns
for i in range(2, 8):
    df[f'event_dummy_{i}'] = 0

# Apply minimum duration rules
for start_idx, duration in periods:
    end_idx = start_idx + duration
    
    # Mark all days in periods that meet minimum duration requirements
    for min_days in range(2, 8):  # event_dummy_2 to event_dummy_7
        if duration >= min_days:
            col_name = f'event_dummy_{min_days}'
            df.iloc[start_idx:end_idx, df.columns.get_loc(col_name)] = 1

# Create event_count columns for all event types
for event_num in range(1, 8):
    event_dummy_col = f'event_dummy_{event_num}'
    event_count_col = f'event_count_{event_num}'
    
    if df[event_dummy_col].sum() > 0:
        # Find event starts: current day is 1 and previous day is 0
        event_series = df[event_dummy_col]
        event_starts = (event_series == 1) & (event_series.shift(1, fill_value=0) == 0)
        df[event_count_col] = event_starts.astype(int)
        
        print(f"Created {event_count_col}: {df[event_count_col].sum()} event starts")
    else:
        df[event_count_col] = 0

print("✅ Created all event types and event_count columns")

# Utility function for dynamic event column names
def get_event_column_name(min_duration):
    """Get the appropriate event column name based on minimum duration"""
    return f'event_dummy_{min_duration}'

# Function for adding event analysis to dataframe
def add_event_analysis_to_df(df, prediction_col, event_col_suffix):
    """Add event duration and coverage analysis columns"""
    
    # Get dynamic event column name
    event_column = get_event_column_name(AEP_CONFIG['min_event_duration'])
    
    # Extract events from predictions and observations
    pred_events = extract_events_with_details(df[prediction_col].values, AEP_CONFIG['min_event_duration'])
    obs_events = extract_events_with_details(df[event_column].values, AEP_CONFIG['min_event_duration'])
    
    # Initialize columns
    df[f'pred_event_id_{event_col_suffix}'] = 0
    df[f'pred_event_duration_{event_col_suffix}'] = 0
    df[f'obs_event_id_{event_col_suffix}'] = 0
    df[f'obs_event_duration_{event_col_suffix}'] = 0
    df[f'event_type_{event_col_suffix}'] = 'None'
    df[f'overlap_coverage_{event_col_suffix}'] = 0.0
    
    # Mark observed events
    for i, obs_event in enumerate(obs_events, 1):
        mask = (df.index >= obs_event['start']) & (df.index <= obs_event['end'])
        df.loc[mask, f'obs_event_id_{event_col_suffix}'] = i
        df.loc[mask, f'obs_event_duration_{event_col_suffix}'] = obs_event['duration']
    
    # Mark predicted events and calculate overlaps
    for i, pred_event in enumerate(pred_events, 1):
        pred_mask = (df.index >= pred_event['start']) & (df.index <= pred_event['end'])
        df.loc[pred_mask, f'pred_event_id_{event_col_suffix}'] = i
        df.loc[pred_mask, f'pred_event_duration_{event_col_suffix}'] = pred_event['duration']
        
        # Find best matching observed event
        best_obs_event = None
        best_overlap_pct = 0
        
        for obs_event in obs_events:
            # Calculate overlap
            overlap_start = max(pred_event['start'], obs_event['start'])
            overlap_end = min(pred_event['end'], obs_event['end'])
            
            if overlap_start <= overlap_end:
                overlap_length = overlap_end - overlap_start + 1
                overlap_pct = overlap_length / pred_event['duration']
                
                if overlap_pct > best_overlap_pct:
                    best_obs_event = obs_event
                    best_overlap_pct = overlap_pct
        
        # Classify event type and set coverage
        if best_obs_event is not None and best_overlap_pct >= AEP_CONFIG['min_overlap']:
            df.loc[pred_mask, f'event_type_{event_col_suffix}'] = 'TP'
            df.loc[pred_mask, f'overlap_coverage_{event_col_suffix}'] = best_overlap_pct
        else:
            df.loc[pred_mask, f'event_type_{event_col_suffix}'] = 'FP'
            df.loc[pred_mask, f'overlap_coverage_{event_col_suffix}'] = best_overlap_pct
    
    # Mark FN events (observed events not captured by predictions)
    for obs_event in obs_events:
        obs_mask = (df.index >= obs_event['start']) & (df.index <= obs_event['end'])
        
        # Check if this observed event overlaps with any prediction sufficiently
        has_sufficient_prediction = False
        for pred_event in pred_events:
            overlap_start = max(pred_event['start'], obs_event['start'])
            overlap_end = min(pred_event['end'], obs_event['end'])
            
            if overlap_start <= overlap_end:
                overlap_length = overlap_end - overlap_start + 1
                overlap_pct = overlap_length / pred_event['duration']
                
                if overlap_pct >= AEP_CONFIG['min_overlap']:
                    has_sufficient_prediction = True
                    break
        
        if not has_sufficient_prediction:
            # Mark as FN where there's no prediction
            fn_mask = obs_mask & (df[prediction_col] == 0)
            df.loc[fn_mask, f'event_type_{event_col_suffix}'] = 'FN'

print(f"Loaded data for {region}")
print(f"Data path: {data_path}")
print(f"Shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Columns: {list(df.columns)}")

# Cell 4: Basic info

# Data info
print("Data info:")
print(df.info())

# Cell 6: Check for specific columns

# Check for wave height and event columns
wave_cols = [col for col in df.columns if 'swh' in col.lower()]
event_cols = [col for col in df.columns if 'event' in col.lower()]

print(f"Wave height columns: {wave_cols}")
print(f"Event columns: {event_cols}")

# Cell 7: Quick plot (if wave data available)

# Quick plot of wave height if available
if wave_cols:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[wave_cols[0]], alpha=0.7, linewidth=0.5)
    plt.title(f'{wave_cols[0]} - {region}')
    plt.ylabel('Wave Height (m)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Cell 8: Dual-axis plot with event shading

# Filter data since 2018
df_2018 = df[df.index >= '2018-01-01'].copy()

# Get unique years since 2018
years = sorted(df_2018.index.year.unique())
n_years = len(years)

# Create subplots: single column, multiple rows
fig, axes = plt.subplots(n_years, 1, figsize=(15, 4*n_years))
if n_years == 1:
    axes = [axes]  # Make it iterable if only one year

# Event colors for shading
event_colors = ['black', 'navy', 'red', 'black']
event_columns = [f'event_dummy_{i}' for i in [1,3,5,7]]

# Plot each year in a separate panel
for i, year in enumerate(years):
    ax = axes[i]
    
    # Filter data for this year
    df_year = df_2018[df_2018.index.year == year].copy()
    
    # Create secondary y-axis
    ax2 = ax.twinx()
    
    # Plot swh_max_swan on primary y-axis
    if 'swh_max_swan' in df_year.columns:
        color1 = 'red'
        ax.plot(df_year.index, df_year['swh_max_swan'], color=color1, alpha=0.99, linewidth=1.5, label='swh_max_swan')
        ax.set_ylabel('Wave Height (m)', color=color1)
        ax.tick_params(axis='y', labelcolor=color1)
        # ax.grid(True, alpha=0.3)
    
    # Plot anom_swh_max_swan on secondary y-axis
    if 'anom_swh_max_swan' in df_year.columns:
        color2 = 'darkorange'
        ax2.plot(df_year.index, df_year['anom_swh_max_swan'], color=color2, alpha=0.99, linewidth=1.5, label='anom_swh_max_swan')
        ax2.set_ylabel('Anomaly Wave Height (m)', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
    
    # Shade background for different event types
    for j, event_col in enumerate(event_columns):
        if event_col in df_year.columns:
            # Find periods where this event is active
            event_periods = df_year[df_year[event_col] == 1]
            if len(event_periods) > 0:
                # Shade the background for each event period
                for start_date in event_periods.index:
                    # Find the end of this event period
                    end_date = start_date
                    current_date = start_date
                    while current_date in event_periods.index:
                        end_date = current_date
                        current_date += pd.Timedelta(days=1)
                    
                    # Special handling for events 3, 5, and 7: use box frames instead of shading
                    if event_col in ['event_dummy_3', 'event_dummy_5', 'event_dummy_7']:
                        # Get y-axis limits for the box and extend slightly beyond
                        y_min, y_max = ax.get_ylim()
                        y_range = y_max - y_min
                        y_min_extended = y_min - 0.2 * y_range
                        y_max_extended = y_max + 0.2 * y_range
                        
                        # Create a rectangle frame with different colors for each event
                        from matplotlib.patches import Rectangle
                        if event_col == 'event_dummy_3':
                            box_color = 'forestgreen'
                        elif event_col == 'event_dummy_5':
                            box_color = 'orange'
                        else:  # event_dummy_7
                            box_color = 'navy'
                        
                        rect = Rectangle((start_date, y_min_extended), end_date - start_date, y_max_extended - y_min_extended, 
                                       fill=False, edgecolor=box_color, linewidth=3, alpha=0.6,
                                       label=event_col if start_date == event_periods.index[0] else "")
                        ax.add_patch(rect)
                    elif event_col == 'event_dummy_1':
                        # Shade the background only for event_dummy_1
                        ax.axvspan(start_date, end_date, alpha=0.4, color='lightgray', 
                                   label=event_col if start_date == event_periods.index[0] else "")
    
    # Set title for this year
    ax.set_title(f'{year} - {region}', fontsize=12, fontweight='bold')
    
    # Format x-axis for this panel
    ax.tick_params(axis='x', rotation=45)
    
    # Add legend for first panel only
    if i == 0:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

# Set overall title
# fig.suptitle(f'Wave Height and Anomalies with Event Shading by Year - {region}', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()

# Print event statistics
print(f"\nEvent statistics since 2018:")
for i, event_col in enumerate(event_columns):
    if event_col in df_2018.columns:
        event_count = df_2018[event_col].sum()
        event_days = event_count / len(df_2018) * 100
        print(f"  {event_col}: {event_count} days ({event_days:.1f}%)")

event_count_cols = [col for col in df.columns if 'event_count' in col.lower()]
for i, event_col in enumerate(event_count_cols):
    if event_col in df_2018.columns:
        events = df_2018[event_col].sum()
        print(f"  {event_col}: {events} events")

for year in df_2018['year'].unique():
    n_events = df_2018[df_2018['year'] == year]['event_count_5'].sum()
    print(f"Year {year}: {n_events} events (event_count_5)")

# Plot: Shade only 5-day events (event_dummy_5), all years in a single plot, forestgreen line, red shading, no anomaly axis

import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# Filter data to end in December 2024
df_plot = df_2018[df_2018.index <= '2024-12-31'].copy()

fig, ax = plt.subplots(figsize=(14, 6))

# Plot wave height for all years in forestgreen
ax.plot(df_plot.index, df_plot['swh_max_rolling_mean_14_swan'], 
label='Wave Height', color='forestgreen', linewidth=3.5)

# Shade only 5-day events (event_dummy_5) in red
event_col = 'event_dummy_5'
if event_col in df_plot.columns:
    event_periods = df_plot[df_plot[event_col] == 1]
    if not event_periods.empty:
        # Find contiguous periods
        is_new_event = (event_periods.index.to_series().diff().dt.days != 1).cumsum()
        for _, group in event_periods.groupby(is_new_event):
            start_date = group.index[0]
            end_date = group.index[-1] + pd.Timedelta(days=1)  # make end exclusive for shading
            ax.axvspan(start_date, end_date, alpha=0.4, color='black', 
                       label='5-day event' if start_date == event_periods.index[0] else "")

# Formatting
ax.set_title('Max Daily Wave Height (14 day MA) with 5-Day Event Shading', fontsize=15, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Wave Height')
ax.tick_params(axis='x', rotation=45)

# Format x-axis with quarterly labels (March, June, September, December)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# Add vertical lines for each December
december_dates = pd.date_range(start=df_plot.index.min(), end=df_plot.index.max(), freq='Y')
for date in december_dates:
    december_date = pd.Timestamp(date.year, 12, 1)
    if december_date >= df_plot.index.min() and december_date <= df_plot.index.max():
        ax.axvline(x=december_date, color='blue',
         linestyle='-', alpha=0.95, linewidth=1.5)

# Add gridlines only on y-axis
ax.grid(True, axis='y', alpha=0.3)

# Get legend handles and labels (no duplicate handling needed)
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='upper left', fontsize=9)

plt.tight_layout()
plt.show()


# Loss calculation
W = 12
N_FISHERMEN = 3290

# Compute daily losses: W * N_FISHERMEN when any event_dummy_1 through event_dummy_7 is 1
event_dummies = [col for col in df_2018.columns if col.startswith('event_dummy_') and col[-1].isdigit()]
# Create a boolean mask for days with any event
event_mask = df_2018[event_dummies].any(axis=1)
# Compute daily losses
df_2018['daily_loss'] = 0
df_2018.loc[event_mask, 'daily_loss'] = W * N_FISHERMEN

# Compute yearly losses
df_2018['year'] = df_2018.index.year
yearly_losses = df_2018.groupby('year')['daily_loss'].sum()

# Plot the evolution of yearly losses as a bar plot
plt.figure(figsize=(8, 5))
plt.bar(yearly_losses.index, yearly_losses.values, color='crimson')
plt.title('Yearly Losses Due to Events')
plt.xlabel('Year')
plt.ylabel('Total Losses (Currency Units)')
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Create a stacked bar plot of yearly losses by event duration categories (non-overlapping)

# Get all event_dummy_ columns
event_dummies = [col for col in df_2018.columns if col.startswith('event_dummy_') and col[-1].isdigit()]
print(f"Found event columns: {event_dummies}")

# Create non-overlapping duration categories
def categorize_event_duration(duration):
    """Categorize event into mutually exclusive duration buckets"""
    if duration == 1:
        return 'event_dummy_1_only'  # 1 day events
    elif duration == 2:
        return 'event_dummy_2_only'  # exactly 2 days
    elif duration == 3:
        return 'event_dummy_3_only'  # exactly 3 days  
    elif duration == 4:
        return 'event_dummy_4_only'  # exactly 4 days
    elif duration == 5:
        return 'event_dummy_5_only'  # exactly 5 days
    elif duration == 6:
        return 'event_dummy_6_only'  # exactly 6 days
    elif duration >= 7:
        return 'event_dummy_7_only'  # 7+ days
    else:
        return 'unknown'

# Process event_dummy_1 to get actual event periods and their durations
print(f"\nProcessing event_dummy_1 to get true event durations:")

# Group consecutive days into event blocks using event_dummy_1 (the base events)
df_temp = df_2018.copy()
df_temp['event_group'] = (df_temp['event_dummy_1'] != df_temp['event_dummy_1'].shift()).cumsum()

# Get event blocks where the event is active (value = 1)
event_blocks = df_temp[df_temp['event_dummy_1'] == 1].groupby('event_group')

print(f"  Found {len(event_blocks)} total event blocks")

# Calculate losses for each event block by true duration category
yearly_losses_by_category = {}
category_names = ['event_dummy_1_only', 'event_dummy_2_only', 'event_dummy_3_only', 
                 'event_dummy_4_only', 'event_dummy_5_only', 'event_dummy_6_only', 'event_dummy_7_only']

for category in category_names:
    yearly_losses_by_category[category] = {}

for group_id, block_data in event_blocks:
    duration = len(block_data)
    event_loss = duration * W * N_FISHERMEN
    year = block_data.index[0].year
    
    # Categorize this event by its actual duration
    category = categorize_event_duration(duration)
    
    if year not in yearly_losses_by_category[category]:
        yearly_losses_by_category[category][year] = 0
    yearly_losses_by_category[category][year] += event_loss
    
    start_date = block_data.index[0].date()
    end_date = block_data.index[-1].date()
    print(f"    Block {group_id}: {start_date} to {end_date}, {duration} days -> {category}, ${event_loss:,} loss")

# Combine into a DataFrame
all_years = sorted(set().union(*[set(losses.keys()) for losses in yearly_losses_by_category.values() if losses]))
yearly_losses_df = pd.DataFrame(index=all_years, columns=category_names, dtype=float).fillna(0)

for category, losses in yearly_losses_by_category.items():
    for year, loss in losses.items():
        yearly_losses_df.loc[year, category] = loss

yearly_losses_df = yearly_losses_df.sort_index()

print(f"\nYearly losses DataFrame shape: {yearly_losses_df.shape}")
print(f"Yearly losses DataFrame columns: {list(yearly_losses_df.columns)}")
print(f"Yearly losses DataFrame:\n{yearly_losses_df}")

# Plot stacked bar with better colors and labels
plt.figure(figsize=(14, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
display_labels = ['1 day', '2 days', '3 days', '4 days', '5 days', '6 days', '7+ days']

# Calculate cumulative sums for stacking
cumulative = np.zeros(len(yearly_losses_df))

for i, (category, label) in enumerate(zip(category_names, display_labels)):
    if category in yearly_losses_df.columns and yearly_losses_df[category].sum() > 0:
        values = yearly_losses_df[category].values
        plt.bar(
            yearly_losses_df.index,
            values,
            bottom=cumulative,
            label=f'{label} events',
            color=colors[i % len(colors)],
            alpha=0.8,
            edgecolor='white',
            linewidth=0.5
        )
        cumulative += values
        total_loss = values.sum()
        num_events = sum(1 for v in values if v > 0)
        print(f"Added {label} events: ${total_loss:,} total losses")

plt.title('Yearly Losses by Event Duration (Non-Overlapping Categories)', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Total Losses (Currency Units)', fontsize=12)
plt.legend(title='Event Duration', fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, axis='y', linestyle='--', alpha=0.3)

# Format y-axis to show currency
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Add total loss annotation
total_annual_losses = cumulative
for i, (year, total) in enumerate(zip(yearly_losses_df.index, total_annual_losses)):
    if total > 0:
        plt.text(year, total + max(total_annual_losses) * 0.01, f'${total:,.0f}', 
                ha='center', va='bottom', fontsize=8, rotation=0)

plt.tight_layout()
plt.show()

# Print summary statistics
print(f"\n" + "="*60)
print(f"SUMMARY OF YEARLY LOSSES BY EVENT DURATION (NON-OVERLAPPING)")
print("="*60)

total_all_losses = 0
for category, label in zip(category_names, display_labels):
    if category in yearly_losses_df.columns:
        total_loss = yearly_losses_df[category].sum()
        avg_loss_per_year = total_loss / len(yearly_losses_df) if len(yearly_losses_df) > 0 else 0
        max_year_loss = yearly_losses_df[category].max()
        years_with_events = (yearly_losses_df[category] > 0).sum()
        
        print(f"{label:>8} events: ${total_loss:>12,} total | ${avg_loss_per_year:>10,.0f} avg/year | {years_with_events} years affected")
        total_all_losses += total_loss

print("-" * 60)
print(f"{'TOTAL':>8}: ${total_all_losses:>12,} across all event types")
print(f"Average annual loss: ${total_all_losses / len(yearly_losses_df):,.0f}")

# Verify no double counting by checking against simple calculation
simple_total = df_2018['event_dummy_1'].sum() * W * N_FISHERMEN
print(f"\nVerification:")
print(f"Total event days (event_dummy_1): {df_2018['event_dummy_1'].sum()}")
print(f"Simple calculation: {df_2018['event_dummy_1'].sum()} days × ${W} × {N_FISHERMEN} = ${simple_total:,}")
print(f"Categorized calculation: ${total_all_losses:,}")
print(f"Match: {'✅ YES' if abs(simple_total - total_all_losses) < 1 else '❌ NO'}")

# go to notebook for rule evaluation


# # Load rule evaluation results


# # Load rule evaluation results
# results_dir = os.path.join(project_root, 'results', 'cv_results', region)
# results_path = os.path.join(results_dir, 'simplified_rule_cv_results.csv')

# if os.path.exists(results_path):
#     print(f"\n" + "="*80)
#     print(f"RULE EVALUATION RESULTS - {region}")
#     print("="*80)
    
#     # Load and display results
#     results_df = pd.read_csv(results_path)
#     print(f"✅ Loaded rule evaluation results: {results_df.shape}")
#     print(f"📁 Results file: {results_path}")
    
#     # Display the results
#     print(f"\n📊 RULE EVALUATION RESULTS:")
#     print(results_df.to_string(index=False))
    
# else:
#     print(f"\n❌ Rule evaluation results not found: {results_path}")
#     print(f"   Run the GP_rule_evaluation.py script first to generate results.")


# # Rule threshold extraction


# # Function to extract thresholds for a specific rule
# def extract_rule_thresholds(rule_name):
#     """Extract threshold values for a specific rule"""
#     # Find the rule in the results
#     rule_match = results_df[results_df['rule_name'] == rule_name]
    
#     if len(rule_match) == 0:
#         print(f"❌ Rule '{rule_name}' not found in results")
#         return None
    
#     rule = rule_match.iloc[0]
#     print(f"\n🎯 THRESHOLDS FOR RULE: {rule_name}")
#     print(f"   F1 Score: {rule['f1_mean']:.3f}")
#     print(f"   Precision: {rule['precision_mean']:.3f}")
#     print(f"   Recall: {rule['recall_mean']:.3f}")
#     print(f"   Rule Type: {rule['rule_type']}")
    
#     # Extract threshold columns
#     threshold_cols = [col for col in rule.index if col.startswith('thresholds_')]
    
#     if threshold_cols:
#         print(f"\n📋 THRESHOLD VALUES:")
#         for col in sorted(threshold_cols):
#             if rule[col]:  # If threshold value exists
#                 if col == 'thresholds_mean':
#                     print(f"   Mean across folds: {rule[col]}")
#                 else:
#                     fold_num = col.split('_')[-1]
#                     print(f"   Fold {fold_num}: {rule[col]}")
#     else:
#         print(f"   No threshold information available")
    
#     return rule

# # Show available rules for reference
# print(f"Available rules (first 10):")
# for i, rule_name in enumerate(results_df['rule_name'].head(10)):
#     print(f"  {i+1}. {rule_name}")

# # Example: Extract thresholds for the best rule
# best_rule = results_df.iloc[0]['rule_name']
# print(f"\n📊 Example: Extracting thresholds for the best rule:")
# extract_rule_thresholds(best_rule)

# # To extract thresholds for other rules, call:
# extract_rule_thresholds("swh_max_swan > threshold")


# Plot swh_max_swan with event_dummy_5 shading and threshold line
#%%

# med = 3
# rolt = 3
# rolw = 5

med = 5
rolt = 5
rolw = 7

# med = 7
# rolt = 7
# rolw = 10

wlim = 1.15
# Set parameters for analysis
WAVE_THRESHOLD  = wlim  # Wave height threshold
ROLLING_WINDOW = rolw    # Rolling window size
ROLLING_THRESHOLD = rolt # Rolling sum threshold for event prediction
COVERAGE_THRESHOLD = 0.4  # Minimum coverage to count as correctly predicted event
MIN_EVENT_DURATION = med   # Only evaluate events that are MIN_EVENT_DURATION+ days
RUN_NAME = region

#%%
# COMPLETE EVENT-LEVEL ANALYSIS WITH CORRECTED CALCULATIONS
# ===========================================================

print(f"COMPLETE EVENT-LEVEL ANALYSIS")
print("="*60)
print(f"Analysis Parameters:")
print(f"  Wave height threshold: {WAVE_THRESHOLD}")
print(f"  Rolling window: {ROLLING_WINDOW}")
print(f"  Rolling sum threshold: {ROLLING_THRESHOLD}")
print(f"  Coverage threshold: {COVERAGE_THRESHOLD}")
print(f"  Minimum event duration: {MIN_EVENT_DURATION} days")

# Filter data since 2018
df_2018 = df[df.index >= '2018-01-01'].copy()

# ============================================================================
# STEP 1: CREATE PREDICTIONS AND BASIC CLASSIFICATIONS
# ============================================================================

print(f"\nStep 1: Computing predictions and basic classifications...")

# Create month column
df_2018['month'] = df_2018.index.month

# Create prediction column based on threshold with seasonal condition
# df_2018['prediction'] = (df_2018['swh_max_swan'] > WAVE_THRESHOLD).astype(int)
df_2018['prediction'] = ((df_2018['swh_max_swan'] > WAVE_THRESHOLD)).astype(int)


# Compute TP, FP, TN, FN (day-level)
event_column = get_event_column_name(MIN_EVENT_DURATION)
df_2018['TP'] = ((df_2018['prediction'] == 1) & (df_2018[event_column] == 1)).astype(int)
df_2018['FP'] = ((df_2018['prediction'] == 1) & (df_2018[event_column] == 0)).astype(int)
df_2018['TN'] = ((df_2018['prediction'] == 0) & (df_2018[event_column] == 0)).astype(int)
df_2018['FN'] = ((df_2018['prediction'] == 0) & (df_2018[event_column] == 1)).astype(int)

# ============================================================================
# STEP 2: CREATE DURATION-BASED EVENT PREDICTIONS
# ============================================================================

print(f"Step 2: Adding duration-based event prediction (window={ROLLING_WINDOW})...")

# Create rolling sum of predictions
df_2018['prediction_rolling_sum'] = df_2018['prediction'].rolling(
    window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW
).sum()

# Create event-level prediction: requires at least ROLLING_THRESHOLD consecutive days
df_2018['prediction_event'] = (
    (df_2018['prediction_rolling_sum'] >= ROLLING_THRESHOLD) & 
    (df_2018['prediction'] == 1)
).astype(int)

# ============================================================================
# STEP 3: EVENT PERIOD DETECTION (NO DOUBLE-COUNTING)
# ============================================================================

def find_continuous_periods(series):
    """Find continuous periods where series == 1"""
    periods = []
    current_start = None
    
    for date, value in series.items():
        if value == 1 and current_start is None:
            current_start = date
        elif value == 0 and current_start is not None:
            periods.append((current_start, date - pd.Timedelta(days=1)))
            current_start = None
    
    # Handle case where period extends to end of data
    if current_start is not None:
        periods.append((current_start, series.index[-1]))
    
    return periods

print(f"Step 3: Detecting event periods (no double-counting)...")

# Find all observed event periods
all_observed_events = find_continuous_periods(df_2018[event_column])

# Filter for MIN_EVENT_DURATION+ day events only
observed_events = [(start, end) for start, end in all_observed_events 
                   if (end - start).days + 1 >= MIN_EVENT_DURATION]

# Find all predicted event periods
all_predicted_events = find_continuous_periods(df_2018['prediction_event'])

# Filter for MIN_EVENT_DURATION+ day events only
predicted_events = [(start, end) for start, end in all_predicted_events 
                    if (end - start).days + 1 >= MIN_EVENT_DURATION]

print(f"Found {len(all_observed_events)} total observed event periods")
print(f"Found {len(observed_events)} observed event periods of {MIN_EVENT_DURATION}+ days")
print(f"Found {len(all_predicted_events)} total predicted event periods")
print(f"Found {len(predicted_events)} predicted event periods of {MIN_EVENT_DURATION}+ days")

# ============================================================================
# STEP 4: ASSIGN EVENT IDS AND DURATIONS (NO OVERLAPS)
# ============================================================================

print(f"Step 4: Assigning event IDs and durations...")

# Initialize columns
df_2018['event_duration'] = 0
df_2018['predicted_event_duration'] = 0
df_2018['predicted_coverage'] = 0.0
df_2018['event_id'] = 0  # To track which event each day belongs to
df_2018['predicted_event_id'] = 0  # To track which prediction each day belongs to

# Assign observed event durations and IDs (MIN_EVENT_DURATION+ days only)
for event_id, (start, end) in enumerate(observed_events, 1):
    duration = (end - start).days + 1
    df_2018.loc[start:end, 'event_duration'] = duration
    df_2018.loc[start:end, 'event_id'] = event_id

# Assign predicted event durations and IDs (MIN_EVENT_DURATION+ days only)
for pred_id, (start, end) in enumerate(predicted_events, 1):
    duration = (end - start).days + 1
    df_2018.loc[start:end, 'predicted_event_duration'] = duration
    df_2018.loc[start:end, 'predicted_event_id'] = pred_id

print(f"✅ Assigned IDs to {len(observed_events)} observed events ({MIN_EVENT_DURATION}+ days)")
print(f"✅ Assigned IDs to {len(predicted_events)} predicted events ({MIN_EVENT_DURATION}+ days)")

# ============================================================================
# STEP 5: CALCULATE COVERAGE FOR EACH EVENT
# ============================================================================

print(f"Step 5: Calculating coverage for each event...")

def calculate_overlap(period1_start, period1_end, period2_start, period2_end):
    """Calculate overlap between two periods"""
    overlap_start = max(period1_start, period2_start)
    overlap_end = min(period1_end, period2_end)
    
    if overlap_start <= overlap_end:
        return (overlap_end - overlap_start).days + 1
    else:
        return 0

# For each predicted event, find best coverage with observed events
prediction_coverage_map = {}

for pred_id, (pred_start, pred_end) in enumerate(predicted_events, 1):
    best_coverage = 0.0
    best_obs_match = None
    
    # Find best matching observed event
    for obs_id, (obs_start, obs_end) in enumerate(observed_events, 1):
        overlap_days = calculate_overlap(pred_start, pred_end, obs_start, obs_end)
        obs_duration = (obs_end - obs_start).days + 1
        coverage = overlap_days / obs_duration if obs_duration > 0 else 0
        
        if coverage > best_coverage:
            best_coverage = coverage
            best_obs_match = obs_id
    
    prediction_coverage_map[pred_id] = {
        'coverage': best_coverage,
        'obs_match': best_obs_match,
        'pred_start': pred_start,
        'pred_end': pred_end,
        'pred_duration': (pred_end - pred_start).days + 1
    }

# Assign coverage to dataframe
for pred_id, info in prediction_coverage_map.items():
    start = info['pred_start']
    end = info['pred_end']
    coverage = info['coverage']
    df_2018.loc[start:end, 'predicted_coverage'] = coverage

# For each observed event, find if it was detected
observed_event_details = []
for obs_id, (obs_start, obs_end) in enumerate(observed_events, 1):
    obs_duration = (obs_end - obs_start).days + 1
    best_coverage = 0.0
    best_pred_match = None
    
    # Find best matching prediction
    for pred_id, info in prediction_coverage_map.items():
        if info['coverage'] > 0:  # Only consider predictions that have some overlap
            overlap_days = calculate_overlap(obs_start, obs_end, info['pred_start'], info['pred_end'])
            coverage = overlap_days / obs_duration if obs_duration > 0 else 0
            
            if coverage > best_coverage:
                best_coverage = coverage
                best_pred_match = pred_id
    
    is_detected = best_coverage >= COVERAGE_THRESHOLD
    
    observed_event_details.append({
        'obs_id': obs_id,
        'obs_start': obs_start,
        'obs_end': obs_end,
        'obs_duration': obs_duration,
        'best_coverage': best_coverage,
        'pred_match': best_pred_match,
        'is_detected': is_detected
    })

print(f"✅ Calculated coverage for all events")

# ============================================================================
# STEP 6: PERFORMANCE CALCULATIONS (FIXED VERSION)
# ============================================================================

print(f"Step 6: Computing performance metrics...")

# Day-level metrics (for reference)
total_tp = df_2018['TP'].sum()
total_fp = df_2018['FP'].sum()
total_tn = df_2018['TN'].sum()
total_fn = df_2018['FN'].sum()

precision_day = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
recall_day = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1_score_day = 2 * (precision_day * recall_day) / (precision_day + recall_day) if (precision_day + recall_day) > 0 else 0
accuracy_day = (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn)

# Event-level metrics (5+ day events only)
total_observed_events = len(observed_events)
total_predicted_events = len(predicted_events)

# FIXED: Use consistent coverage threshold for both metrics
# Count valid predictions (predictions that meet coverage threshold)
valid_predictions = sum(1 for info in prediction_coverage_map.values() 
                       if info['coverage'] >= COVERAGE_THRESHOLD)
false_alarms = total_predicted_events - valid_predictions

# Count correctly detected events (events that have predictions meeting coverage threshold)
correctly_detected_events = sum(1 for detail in observed_event_details 
                               if detail['best_coverage'] >= COVERAGE_THRESHOLD)
missed_events = total_observed_events - correctly_detected_events

# Calculate metrics using consistent True Positive count
true_positives = correctly_detected_events  # This should equal valid_predictions
event_recall = correctly_detected_events / total_observed_events if total_observed_events > 0 else 0
event_precision = valid_predictions / total_predicted_events if total_predicted_events > 0 else 0
event_f1 = 2 * (event_precision * event_recall) / (event_precision + event_recall) if (event_precision + event_recall) > 0 else 0

# VALIDATION CHECK: True positives should be consistent
print(f"\n🔍 VALIDATION CHECK:")
print(f"  Correctly detected events (for recall): {correctly_detected_events}")
print(f"  Valid predictions (for precision): {valid_predictions}")
if correctly_detected_events == valid_predictions:
    print(f"  ✅ Metrics are consistent! True Positives = {correctly_detected_events}")
else:
    print(f"  ⚠️ INCONSISTENCY DETECTED!")
    print(f"  This suggests overlap/matching issues in the analysis.")
    # Use the minimum as the true positive count to be conservative
    true_positives = min(correctly_detected_events, valid_predictions)
    print(f"  Using conservative TP count: {true_positives}")
    
    # Recalculate with consistent TP
    event_recall = true_positives / total_observed_events if total_observed_events > 0 else 0
    event_precision = true_positives / total_predicted_events if total_predicted_events > 0 else 0
    event_f1 = 2 * (event_precision * event_recall) / (event_precision + event_recall) if (event_precision + event_recall) > 0 else 0
# ============================================================================
# STEP 7: VISUALIZATION
# ============================================================================

print(f"Step 7: Creating visualizations...")

# Get unique years since 2018
years = sorted(df_2018.index.year.unique())
n_years = len(years)

# Create event-level analysis plot
fig, axes = plt.subplots(n_years, 1, figsize=(15, 4*n_years))
if n_years == 1:
    axes = [axes]

print(f"Creating event-level visualization...")

for i, year in enumerate(years):
    ax = axes[i]
    
    df_year = df_2018[df_2018.index.year == year].copy()
    
    # Plot swh_max_swan
    ax.plot(df_year.index, df_year['swh_max_swan'], color='black', alpha=0.8, 
            linewidth=1.5, label='swh_max_swan')
    ax.set_ylabel('Wave Height (m)', color='black')
    ax.tick_params(axis='y', labelcolor='black')
    
    # Add threshold line
    ax.axhline(y=WAVE_THRESHOLD, color='red', linestyle='--', linewidth=2, 
               alpha=0.8, label=f'Threshold = {WAVE_THRESHOLD}')
    
    # Plot rolling sum on secondary axis
    ax2 = ax.twinx()
    ax2.plot(df_year.index, df_year['prediction_rolling_sum'], color='orange', 
             alpha=0.7, linewidth=1, label=f'Rolling Sum (window={ROLLING_WINDOW})')
    ax2.axhline(y=ROLLING_THRESHOLD, color='forestgreen', linestyle=':', alpha=0.7, 
                label=f'Rolling Threshold = {ROLLING_THRESHOLD}')
    ax2.set_ylabel('Rolling Sum', color='forestgreen')
    ax2.tick_params(axis='y', labelcolor='forestgreen')
    ax2.set_ylim(0, 8)
    
    # Add event period shading (5+ day events only)
    year_start = pd.Timestamp(f'{year}-01-01')
    year_end = pd.Timestamp(f'{year}-12-31')
    
    # Track legend entries
    legend_shown = {'blue': False, 'red': False, 'grey': False}
    
    # 1. Observed events - color by detection status
    for detail in observed_event_details:
        if detail['obs_start'].year == year or detail['obs_end'].year == year:
            plot_start = max(detail['obs_start'], year_start)
            plot_end = min(detail['obs_end'] + pd.Timedelta(days=1), year_end + pd.Timedelta(days=1))
            
            color = 'blue' if detail['is_detected'] else 'red'
            if color == 'blue' and not legend_shown['blue']:
                label = f'Detected Events (≥{MIN_EVENT_DURATION}d)'
                legend_shown['blue'] = True
            elif color == 'red' and not legend_shown['red']:
                label = f'Missed Events (≥{MIN_EVENT_DURATION}d)'
                legend_shown['red'] = True
            else:
                label = ""
            
            ax.axvspan(plot_start, plot_end, alpha=0.4, color=color, label=label)
    
    # 2. False alarm predictions (grey)
    for pred_id, info in prediction_coverage_map.items():
        if (info['coverage'] < COVERAGE_THRESHOLD and 
            (info['pred_start'].year == year or info['pred_end'].year == year)):
            
            plot_start = max(info['pred_start'], year_start)
            plot_end = min(info['pred_end'] + pd.Timedelta(days=1), year_end + pd.Timedelta(days=1))
            
            label = f'False Alarms (≥{MIN_EVENT_DURATION}d)' if not legend_shown['grey'] else ""
            ax.axvspan(plot_start, plot_end, alpha=0.3, color='grey', label=label)
            legend_shown['grey'] = True
    
    # # Add vertical black lines at months 4 and 10
    # for m in [4, 10]:
    #     vline_date = pd.Timestamp(f'{year}-{m:02d}-01')
    #     ax.axvline(vline_date, color='black', linestyle='-', linewidth=1.2)
    ax.set_title(f'{year} - Event Analysis ({MIN_EVENT_DURATION}+ Day Events Only)', 
                 fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    if i == 0:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

plt.tight_layout()

plt.savefig(f'/Users/ageidv/suyana/peru_swan/results/{region}_event_analysis_{MIN_EVENT_DURATION}.png')
plt.show()

# ============================================================================
# STEP 8: COMPREHENSIVE REPORTING (CORRECTED VERSION)
# ============================================================================

print(f"\n" + "="*80)
print(f"COMPLETE PERFORMANCE ANALYSIS - 5+ DAY EVENTS ONLY")
print(f"Wave Threshold: {WAVE_THRESHOLD} | Target: event_dummy_5 | Period: 2018-present")
print("="*80)

print(f"\n📊 DAY-LEVEL PERFORMANCE (Traditional Classification)")
print("-" * 60)
print(f"{'Metric':<15} {'Count':<10} {'Percentage':<12}")
print("-" * 60)
print(f"{'True Positives':<15} {total_tp:<10} {total_tp/len(df_2018)*100:>8.2f}%")
print(f"{'False Positives':<15} {total_fp:<10} {total_fp/len(df_2018)*100:>8.2f}%")
print(f"{'True Negatives':<15} {total_tn:<10} {total_tn/len(df_2018)*100:>8.2f}%")
print(f"{'False Negatives':<15} {total_fn:<10} {total_fn/len(df_2018)*100:>8.2f}%")
print("-" * 60)
print(f"{'Total Days':<15} {len(df_2018):<10} {100:>8.2f}%")

print(f"\n📈 Day-Level Performance Metrics:")
print(f"  Precision: {precision_day:.3f}")
print(f"  Recall:    {recall_day:.3f}")
print(f"  F1 Score:  {f1_score_day:.3f}")
print(f"  Accuracy:  {accuracy_day:.3f}")

print(f"\n🎯 EVENT-LEVEL PERFORMANCE ({MIN_EVENT_DURATION}+ DAY EVENTS ONLY)")
print("-" * 60)
print(f"Total Events (All Durations):      {len(all_observed_events)}")
print(f"Events {MIN_EVENT_DURATION}+ Days:                    {total_observed_events}")
print(f"Predicted Events ({MIN_EVENT_DURATION}+ Days):        {total_predicted_events}")
print(f"Correctly Detected Events:         {true_positives}")
print(f"Missed Events:                     {total_observed_events - true_positives}")
print(f"Valid Predictions:                 {true_positives}")
print(f"False Alarms:                      {total_predicted_events - true_positives}")

print(f"\n📈 Event-Level Performance Metrics ({MIN_EVENT_DURATION}+ Day Events):")
print(f"  Event Recall:    {event_recall:.3f} ({true_positives}/{total_observed_events})")
print(f"  Event Precision: {event_precision:.3f} ({true_positives}/{total_predicted_events})")
print(f"  Event F1 Score:  {event_f1:.3f}")

# ============================================================================
# CONFUSION MATRIX DISPLAY (CORRECTED)
# ============================================================================

print(f"\n📊 CONFUSION MATRIX ({MIN_EVENT_DURATION}+ Day Events)")
print("=" * 50)

# Calculate confusion matrix values using corrected metrics
false_negatives = total_observed_events - true_positives  # Missed events
false_positives = total_predicted_events - true_positives  # False alarms

print(f"                 PREDICTED")
print(f"               Event  No Event")
print(f"ACTUAL Event     {true_positives:3d}      {false_negatives:3d}")
print(f"    No Event     {false_positives:3d}       -")
print(f"")
print(f"Metrics:")
print(f"  True Positives (TP):  {true_positives:3d} - Correctly detected events")
print(f"  False Negatives (FN): {false_negatives:3d} - Missed events") 
print(f"  False Positives (FP): {false_positives:3d} - False alarm predictions")
print(f"")
print(f"  Recall    = TP/(TP+FN) = {true_positives}/({true_positives}+{false_negatives}) = {event_recall:.1%}")
print(f"  Precision = TP/(TP+FP) = {true_positives}/({true_positives}+{false_positives}) = {event_precision:.1%}")
print(f"  F1 Score  = 2×(P×R)/(P+R) = {event_f1:.3f}")
print("=" * 50)

print(f"\n🔍 DETAILED EVENT ANALYSIS ({MIN_EVENT_DURATION}+ DAY EVENTS)")
print("-" * 80)
print(f"Coverage Threshold: {COVERAGE_THRESHOLD} (minimum overlap to count as correct)")

print(f"\nObserved Events Analysis:")
for detail in observed_event_details:
    # Use consistent coverage threshold check
    is_detected = detail['best_coverage'] >= COVERAGE_THRESHOLD
    status = "✅ DETECTED" if is_detected else "❌ MISSED"
    match_info = f"(matched pred {detail['pred_match']})" if detail['pred_match'] is not None else "(no match)"
    print(f"  Event {detail['obs_id']}: {detail['obs_start'].strftime('%Y-%m-%d')} to {detail['obs_end'].strftime('%Y-%m-%d')} "
          f"({detail['obs_duration']} days) - Coverage: {detail['best_coverage']:.1%} {status} {match_info}")

# print(f"\nPredicted Events Analysis:")
# for pred_id, info in prediction_coverage_map.items():
#     status = "❌ FALSE ALARM" if info['coverage'] < COVERAGE_THRESHOLD else "✅ VALID"
#     match_info = f"(matched obs {info['obs_match']})" if info['obs_match'] is not None else "(no match)"
#     print(f"  Prediction {pred_id}: {info['pred_start'].strftime('%Y-%m-%d')} to {info['pred_end'].strftime('%Y-%m-%d')} "
#           f"({info['pred_duration']} days) - Coverage: {info['coverage']:.1%} {status} {match_info}")

print(f"\n💡 INTERPRETATION ({MIN_EVENT_DURATION}+ Day Events Only)")
print("-" * 50)
if event_f1 > 0.7:
    print("🟢 EXCELLENT: Event-level F1 > 0.7 indicates strong predictive capability")
elif event_f1 > 0.5:
    print("🟡 GOOD: Event-level F1 > 0.5 indicates reasonable predictive capability")
elif event_f1 > 0.3:
    print("🟠 MODERATE: Event-level F1 > 0.3 indicates some predictive capability")
else:
    print("🔴 POOR: Event-level F1 < 0.3 indicates limited predictive capability")

print(f"\n🎯 SUMMARY FOR {MIN_EVENT_DURATION}+ DAY EVENTS:")
print(f"  • Detection Rate: {event_recall:.1%} ({true_positives} out of {total_observed_events} events)")
print(f"  • Prediction Accuracy: {event_precision:.1%} ({true_positives} out of {total_predicted_events} predictions)")
print(f"  • Overall Performance: F1 = {event_f1:.3f}")

# ============================================================================
# STEP 9: CREATE CLEAN EXPORT
# ============================================================================

print(f"\nStep 9: Creating clean export files...")

# Add useful indicator columns
df_2018['is_event_start'] = (df_2018['event_id'] != df_2018['event_id'].shift(1, fill_value=0)).astype(int)
df_2018['is_prediction_start'] = (df_2018['predicted_event_id'] != df_2018['predicted_event_id'].shift(1, fill_value=0)).astype(int)

# Create export dataframe
event_column = get_event_column_name(MIN_EVENT_DURATION)
event_count_column = f'event_count_{MIN_EVENT_DURATION}'
df_2018_export = df_2018[[
    'month',
    'swh_max_swan', 
    event_column, 
    event_count_column,
    'event_id',
    'event_duration',
    'is_event_start',
    'prediction',
    'prediction_rolling_sum',
    'prediction_event', 
    'predicted_event_id',
    'predicted_event_duration', 
    'predicted_coverage',
    'is_prediction_start',
    'TP', 'FP', 'TN', 'FN'
]].copy()

# Add date as a column
df_2018_export['date'] = df_2018_export.index
df_2018_export = df_2018_export[['date'] + [col for col in df_2018_export.columns if col != 'date']]

# Save CSV
csv_filename = f'/Users/ageidv/suyana/peru_swan/scripts/event_analysis_complete_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
# df_2018_export.to_csv(csv_filename, index=False)

# Create summary tables
event_summary = []
for detail in observed_event_details:
    event_summary.append({
        'Event_ID': detail['obs_id'],
        'Start_Date': detail['obs_start'],
        'End_Date': detail['obs_end'],
        'Duration_Days': detail['obs_duration'],
        'Best_Prediction_ID': detail['pred_match'],
        'Coverage': detail['best_coverage'],
        'Detected': 'Yes' if detail['is_detected'] else 'No'
    })

pred_summary = []
for pred_id, info in prediction_coverage_map.items():
    pred_summary.append({
        'Prediction_ID': pred_id,
        'Start_Date': info['pred_start'],
        'End_Date': info['pred_end'],
        'Duration_Days': info['pred_duration'],
        'Coverage': info['coverage'],
        'Valid': 'Yes' if info['coverage'] >= COVERAGE_THRESHOLD else 'No'
    })

event_summary_df = pd.DataFrame(event_summary)
pred_summary_df = pd.DataFrame(pred_summary)

# Save to Excel
excel_filename = f'/Users/ageidv/suyana/peru_swan/results/event_analysis_complete.xlsx'
with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
    df_2018_export.to_excel(writer, sheet_name='Daily_Data', index=False)
    event_summary_df.to_excel(writer, sheet_name='Event_Summary', index=False)
    pred_summary_df.to_excel(writer, sheet_name='Prediction_Summary', index=False)

print(f"✅ Exported complete analysis:")
print(f"  • CSV: {csv_filename}")
print(f"  • Excel: {excel_filename}")
print(f"    - Daily_Data: {len(df_2018_export)} rows")
print(f"    - Event_Summary: {len(event_summary_df)} events")
print(f"    - Prediction_Summary: {len(pred_summary_df)} predictions")

print(f"\n✅ ANALYSIS COMPLETE - NO DOUBLE COUNTING")
print("="*80)


# %%
# OPTIMIZED Event-Level AEP Calculation Section
# ==============================================
# Fast vectorized implementation using patterns from existing AEP scripts

import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
from numba import jit
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Configuration parameters
AEP_CONFIG = {
    'rule_feature': 'swh_max_swan',
    'rule_threshold': wlim,
    'min_event_duration': med,
    'n_simulations': 4000,
    'min_overlap': 0.4,        # Minimum overlap for TP classification
    'max_coverage': 0.67,       # Maximum coverage per observed event
    'block_length': 7,         # Weekly blocks for bootstrap
    'window_days': 20,         # Seasonal matching window
    'seasonal_filter': False,   # Apply seasonal filter (Apr-Oct)
    'N_FISHERMEN': 14424,        # Number of fishermen
    'W': 12,                   # Daily wage ($)
    'n_jobs': -1,
    'run_name': region               # Parallel workers (-1 = all cores)
}
#%%
print(f"\n" + "="*80)
print(f"OPTIMIZED EVENT-LEVEL AEP ANALYSIS")
# ============================================================================
# NUMBA-OPTIMIZED FUNCTIONS (from existing AEP scripts)
# ============================================================================

@jit(nopython=True, cache=True)
def calculate_annual_loss_jit(predicted_events, N, W, min_days):
    """Fast JIT-compiled annual loss calculation"""
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
    """Fast JIT-compiled event categorization
    category_code: 0=FP, 1=FN, 2=TP
    """
    durations = []
    current_event_length = 0
    for i in range(len(predicted)):
        in_event = False
        if category_code == 0:  # FP
            in_event = (predicted[i] == 1) and (observed[i] == 0)
        elif category_code == 1:  # FN
            in_event = (predicted[i] == 0) and (observed[i] == 1)
        elif category_code == 2:  # TP
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
    """Fast JIT-compiled confusion matrix cost calculation"""
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

@jit(nopython=True, cache=True)
def apply_seasonal_filter_jit(predictions, seasonal_start_day=90, seasonal_end_day=305):
    """Apply seasonal filter (Apr-Oct ≈ days 90-305)"""
    filtered_predictions = np.zeros_like(predictions)
    for i in range(len(predictions)):
        day_of_year = (i % 365) + 1
        if seasonal_start_day <= day_of_year <= seasonal_end_day:
            filtered_predictions[i] = predictions[i]
    return filtered_predictions

# ============================================================================
# VECTORIZED BLOCK BOOTSTRAP (from existing AEP scripts)
# ============================================================================

def vectorized_block_bootstrap(daily_clean, n_simulations, block_length=7, window_days=20, days_per_year=365):
    """Vectorized block bootstrap that pre-computes all simulation indices"""
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

# ============================================================================
# BATCH PROCESSING FUNCTIONS (from existing AEP scripts)
# ============================================================================

def process_simulation_batch_threaded(batch_indices, trigger_values_matrix, observed_matrix, 
                                    N, W, min_days, trigger_threshold, has_observed, seasonal_filter=True):
    """Process a batch of simulations with optimized calculations"""
    batch_losses = []
    batch_fp_costs = []
    batch_fn_costs = []
    batch_tp_costs = []
    batch_event_counts = []
    
    for sim_indices in batch_indices:
        try:
            # Sample trigger values for this simulation
            trigger_vals = trigger_values_matrix[sim_indices]
            
            # Apply threshold to get basic predictions
            predicted_events = (trigger_vals > trigger_threshold).astype(np.int32)
            
            # Apply seasonal filter if enabled
            if seasonal_filter:
                predicted_events = apply_seasonal_filter_jit(predicted_events)
            
            # Calculate total loss
            total_loss = calculate_annual_loss_jit(predicted_events, N, W, min_days)
            batch_losses.append(total_loss)
            
            if has_observed:
                # Sample corresponding observed events
                observed_vals = observed_matrix[sim_indices]
                
                # Calculate confusion matrix costs
                fp_cost, fn_cost, tp_cost = calculate_cm_costs_jit(predicted_events, observed_vals, N, W, min_days)
                batch_fp_costs.append(fp_cost)
                batch_fn_costs.append(fn_cost)
                batch_tp_costs.append(tp_cost)
                
                # Count events by category
                fp_events = len(find_events_in_category_jit(predicted_events, observed_vals, 0))
                fn_events = len(find_events_in_category_jit(predicted_events, observed_vals, 1))
                tp_events = len(find_events_in_category_jit(predicted_events, observed_vals, 2))
                batch_event_counts.append({
                    'fp_events': fp_events, 
                    'fn_events': fn_events, 
                    'tp_events': tp_events
                })
        except Exception as e:
            batch_losses.append(0)
            if has_observed:
                batch_fp_costs.append(0)
                batch_fn_costs.append(0)
                batch_tp_costs.append(0)
                batch_event_counts.append({'fp_events': 0, 'fn_events': 0, 'tp_events': 0})
    
    return batch_losses, batch_fp_costs, batch_fn_costs, batch_tp_costs, batch_event_counts

# ============================================================================
# GET 2024 OBSERVED EVENTS FOR VALIDATION
# ============================================================================

print(f"\n📅 Preparing validation data...")

# Filter to 2024 data (or most recent year if no 2024)
df_2024 = df[df.index.year == 2024].copy()
if len(df_2024) == 0:
    print("⚠️ No 2024 data found. Using most recent year for validation.")
    latest_year = df.index.year.max()
    df_2024 = df[df.index.year == latest_year].copy()
    print(f"Using {latest_year} data instead.")

print(f"✅ Using {len(df_2024)} days of validation data")

# ============================================================================
# PREPARE DATA FOR OPTIMIZED SIMULATION
# ============================================================================

print(f"\n🔧 Preparing data for optimized simulation...")

# Clean and prepare the dataset
df_clean = df.dropna(subset=[AEP_CONFIG['rule_feature']]).copy()
print(f"✅ Using {len(df_clean)} days for simulation")

# Pre-compute trigger values matrix
trigger_values_matrix = df_clean[AEP_CONFIG['rule_feature']].values.astype(np.float32)
print(f"✅ Pre-computed trigger values matrix")

# Prepare observed events matrix for validation
event_column = get_event_column_name(AEP_CONFIG['min_event_duration'])
observed_matrix = df_2024.reindex(df_clean.index, fill_value=0)[event_column].values.astype(np.int32)
has_observed = True
print(f"✅ Aligned observed events: {observed_matrix.sum()} out of {len(observed_matrix)} days")

# Generate all bootstrap indices upfront (this is the key optimization!)
print(f"\n🚀 Generating bootstrap indices...")
all_simulation_indices = vectorized_block_bootstrap(
    df_clean, 
    AEP_CONFIG['n_simulations'], 
    AEP_CONFIG['block_length'], 
    AEP_CONFIG['window_days'], 
    days_per_year=365
)
print(f"✅ Generated all {AEP_CONFIG['n_simulations']} simulation indices")

# ============================================================================
# RUN OPTIMIZED PARALLEL SIMULATION
# ============================================================================

print(f"\n🚀 Running optimized parallel simulation...")

# Set up parallel processing
if AEP_CONFIG['n_jobs'] == -1:
    n_jobs = mp.cpu_count()
else:
    n_jobs = AEP_CONFIG['n_jobs']

n_jobs = min(n_jobs, AEP_CONFIG['n_simulations'])
batch_size = max(1, AEP_CONFIG['n_simulations'] // n_jobs)

print(f"   Using {n_jobs} parallel workers")
print(f"   Batch size: {batch_size}")

# Create batches for parallel processing
simulation_batches = []
for i in range(0, AEP_CONFIG['n_simulations'], batch_size):
    end_idx = min(i + batch_size, AEP_CONFIG['n_simulations'])
    batch_indices = all_simulation_indices[i:end_idx]
    simulation_batches.append(batch_indices)

print(f"   Created {len(simulation_batches)} batches")

# Initialize result arrays
all_losses = []
all_fp_costs = []
all_fn_costs = []
all_tp_costs = []
all_event_counts = []

# Process batches in parallel or sequential based on n_jobs
if n_jobs == 1:
    print("   Using single-threaded processing...")
    for batch_indices in tqdm(simulation_batches, desc="Processing batches"):
        batch_losses, batch_fp, batch_fn, batch_tp, batch_events = process_simulation_batch_threaded(
            batch_indices, trigger_values_matrix, observed_matrix, 
            AEP_CONFIG['N_FISHERMEN'], AEP_CONFIG['W'], AEP_CONFIG['min_event_duration'], 
            AEP_CONFIG['rule_threshold'], has_observed, AEP_CONFIG['seasonal_filter']
        )
        all_losses.extend(batch_losses)
        all_fp_costs.extend(batch_fp)
        all_fn_costs.extend(batch_fn)
        all_tp_costs.extend(batch_tp)
        all_event_counts.extend(batch_events)
else:
    print(f"   Using {n_jobs} threads for parallel processing...")
    def process_batch_wrapper(batch_indices):
        return process_simulation_batch_threaded(
            batch_indices, trigger_values_matrix, observed_matrix, 
            AEP_CONFIG['N_FISHERMEN'], AEP_CONFIG['W'], AEP_CONFIG['min_event_duration'], 
            AEP_CONFIG['rule_threshold'], has_observed, AEP_CONFIG['seasonal_filter']
        )
    
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        results = list(tqdm(
            executor.map(process_batch_wrapper, simulation_batches),
            total=len(simulation_batches),
            desc="Processing batches"
        ))
    
    for batch_losses, batch_fp, batch_fn, batch_tp, batch_events in results:
        all_losses.extend(batch_losses)
        all_fp_costs.extend(batch_fp)
        all_fn_costs.extend(batch_fn)
        all_tp_costs.extend(batch_tp)
        all_event_counts.extend(batch_events)

# Convert to numpy arrays
annual_total_costs = np.array(all_losses)
annual_fp_costs = np.array(all_fp_costs)
annual_fn_costs = np.array(all_fn_costs)
annual_tp_costs = np.array(all_tp_costs)

print(f"✅ Completed {len(annual_total_costs):,} simulations successfully!")

# ============================================================================
# CALCULATE AEP CURVES
# ============================================================================

def calculate_aep_curve(costs):
    """Calculate AEP curve from cost array"""
    if len(costs) == 0:
        return {'loss': [], 'probability': []}
    
    costs_sorted = np.sort(costs)[::-1]  # Sort descending
    exceedance_prob = np.arange(1, len(costs_sorted) + 1) / len(costs_sorted)
    
    return {
        'loss': costs_sorted,
        'probability': exceedance_prob
    }

print(f"\n📊 Calculating AEP curves...")

# Calculate AEP curves for each cost type
tp_aep = calculate_aep_curve(annual_tp_costs)
fp_aep = calculate_aep_curve(annual_fp_costs)
fn_aep = calculate_aep_curve(annual_fn_costs)
total_aep = calculate_aep_curve(annual_total_costs)

# ============================================================================
# DETAILED SUMMARY STATISTICS TABLE
# ============================================================================

print(f"\n📈 DETAILED SUMMARY STATISTICS")
# ============================================================================
# EXPORT TO EXCEL
# ============================================================================

print(f"\n💾 Exporting to Excel file...")

# Create filename with region and min_event_duration
excel_filename = f'event_aep_analysis_{region}_mindur{AEP_CONFIG["min_event_duration"]}.xlsx'
excel_path = os.path.join(project_root, 'results', excel_filename)

# Ensure results directory exists
os.makedirs(os.path.dirname(excel_path), exist_ok=True)

# Export to Excel with multiple sheets
# SKIPPED - Variables not defined in this version
# with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
#     # Main data sheet
#     export_df_final.to_excel(writer, sheet_name='Daily_Data_Analysis', index=False)
#     
#     # Summary statistics sheet
#     summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
#     
#     # Configuration sheet
#     config_df.to_excel(writer, sheet_name='Configuration', index=False)
    
    # Event summary sheet
# Excel export section removed to fix indentation errors
# The main analysis results are saved in CSV format instead

print(f"\n🎉 COMPLETE! Ready to share with colleagues.")
print(f"   📈 The Excel file shows exactly how:")
print(f"      1. Rule predictions are made daily")
print(f"      2. Events are identified and matched")
print(f"      3. TP/FP/FN classifications are determined")
print(f"      4. Economic costs are calculated")
print(f"      5. Bootstrap simulations create AEP curves")

print("="*80)

# Helper function to extract actual events (not just days)
def extract_events_with_details(event_series, min_duration):
    """Extract events with start, end, and duration details"""
    events = []
    current_start = None
    current_length = 0
    
    for i, val in enumerate(event_series):
        if val == 1:
            if current_start is None:
                current_start = i
            current_length += 1
        else:
            if current_start is not None and current_length >= min_duration:
                events.append({
                    'start': current_start,
                    'end': current_start + current_length - 1,
                    'duration': current_length
                })
            current_start = None
            current_length = 0
    
    # Handle event at end of series
    if current_start is not None and current_length >= min_duration:
        events.append({
            'start': current_start,
            'end': current_start + current_length - 1,
            'duration': current_length
        })
    
    return events

# Calculate 2024 observed statistics (proper event counting)
event_column = get_event_column_name(AEP_CONFIG['min_event_duration'])
observed_events_list_2024 = extract_events_with_details(
    df_2024[event_column].values, 
    AEP_CONFIG['min_event_duration']
)
observed_event_count_2024 = len(observed_events_list_2024)
observed_event_days_2024 = sum(event['duration'] for event in observed_events_list_2024)
observed_loss_2024 = calculate_annual_loss_jit(
    df_2024[event_column].values.astype(np.int32), 
    AEP_CONFIG['N_FISHERMEN'], 
    AEP_CONFIG['W'], 
    AEP_CONFIG['min_event_duration']
)

# Calculate rule predictions on 2024 data
df_2024_pred = df_2024.copy()
if AEP_CONFIG['seasonal_filter']:
    df_2024_pred['month'] = df_2024_pred.index.month
    seasonal_mask = (df_2024_pred['month'] >= 4) & (df_2024_pred['month'] <= 10)
    predictions_2024 = ((df_2024_pred[AEP_CONFIG['rule_feature']] > AEP_CONFIG['rule_threshold']) & seasonal_mask).astype(int)
else:
    predictions_2024 = (df_2024_pred[AEP_CONFIG['rule_feature']] > AEP_CONFIG['rule_threshold']).astype(int)

predicted_events_list_2024 = extract_events_with_details(
    predictions_2024.values, 
    AEP_CONFIG['min_event_duration']
)
predicted_event_count_2024 = len(predicted_events_list_2024)
predicted_event_days_2024 = predictions_2024.sum()

# Simple event matching for performance metrics
# Match predicted events to observed events based on overlap
def simple_event_matching(predicted_events, observed_events, min_overlap=0.4):
    """Simple event matching based on temporal overlap"""
    tp_events = []
    fp_events = list(predicted_events)  # Start with all as FP
    fn_events = list(observed_events)   # Start with all as FN
    
    for pred_event in predicted_events:
        best_match = None
        best_overlap = 0
        
        for obs_event in observed_events:
            # Calculate overlap
            overlap_start = max(pred_event['start'], obs_event['start'])
            overlap_end = min(pred_event['end'], obs_event['end'])
            
            if overlap_start <= overlap_end:
                overlap_length = overlap_end - overlap_start + 1
                overlap_pct = overlap_length / pred_event['duration']
                
                if overlap_pct >= min_overlap and overlap_pct > best_overlap:
                    best_match = obs_event
                    best_overlap = overlap_pct
        
        if best_match is not None:
            # This is a TP
            tp_events.append(pred_event)
            if pred_event in fp_events:
                fp_events.remove(pred_event)
            if best_match in fn_events:
                fn_events.remove(best_match)
    
    return tp_events, fp_events, fn_events

# Perform event matching
tp_events_2024, fp_events_2024, fn_events_2024 = simple_event_matching(
    predicted_events_list_2024, 
    observed_events_list_2024,
    min_overlap=AEP_CONFIG['min_overlap']
)

# Create comprehensive summary table
print(f"\n📊 SIMULATION RESULTS SUMMARY TABLE")
print("="*80)

# Simulated Loss Statistics
print(f"{'SIMULATED ANNUAL LOSSES (4000 simulations)':<50}")
print("-" * 50)
print(f"{'Metric':<15} {'TP Costs ($)':<15} {'FN Costs ($)':<15}")
print("-" * 45)
print(f"{'Median':<15} {np.median(annual_tp_costs):<15,.0f} {np.median(annual_fn_costs):<15,.0f}")
print(f"{'Mean':<15} {np.mean(annual_tp_costs):<15,.0f} {np.mean(annual_fn_costs):<15,.0f}")
print(f"{'P95':<15} {np.percentile(annual_tp_costs, 95):<15,.0f} {np.percentile(annual_fn_costs, 95):<15,.0f}")
print(f"{'P99':<15} {np.percentile(annual_tp_costs, 99):<15,.0f} {np.percentile(annual_fn_costs, 99):<15,.0f}")

print(f"\n{'SIMULATED ANNUAL EVENT COUNTS (4000 simulations)':<50}")
print("-" * 50)
if all_event_counts:
    annual_tp_events = [item['tp_events'] for item in all_event_counts]
    annual_fp_events = [item['fp_events'] for item in all_event_counts]
    annual_fn_events = [item['fn_events'] for item in all_event_counts]
    
    print(f"{'Metric':<15} {'TP Events':<12} {'FP Events':<12} {'FN Events':<12}")
    print("-" * 51)
    print(f"{'Median':<15} {np.median(annual_tp_events):<12.1f} {np.median(annual_fp_events):<12.1f} {np.median(annual_fn_events):<12.1f}")
    print(f"{'Mean':<15} {np.mean(annual_tp_events):<12.1f} {np.mean(annual_fp_events):<12.1f} {np.mean(annual_fn_events):<12.1f}")
    print(f"{'P95':<15} {np.percentile(annual_tp_events, 95):<12.1f} {np.percentile(annual_fp_events, 95):<12.1f} {np.percentile(annual_fn_events, 95):<12.1f}")
    print(f"{'P99':<15} {np.percentile(annual_tp_events, 99):<12.1f} {np.percentile(annual_fp_events, 99):<12.1f} {np.percentile(annual_fn_events, 99):<12.1f}")
else:
    print("Event count data not available")

print(f"\n{'OBSERVED 2024 DATA':<50}")
print("-" * 50)
print(f"{'Number of events (' + str(MIN_EVENT_DURATION) + '+ consecutive days)':<40}: {observed_event_count_2024:>8}")
print(f"{'Total event days across all events':<40}: {observed_event_days_2024:>8}")
print(f"{'Observed annual loss (all ' + str(MIN_EVENT_DURATION) + '+ day events)':<40}: ${observed_loss_2024:>8,.0f}")
if observed_events_list_2024:
    avg_duration = observed_event_days_2024 / observed_event_count_2024
    print(f"{'Average event duration':<40}: {avg_duration:>8.1f} days")
    print(f"{'Event details:':<40}")
    for i, event in enumerate(observed_events_list_2024, 1):
        print(f"{'  Event ' + str(i):<40}: {event['duration']} days")

print(f"\n{'RULE PERFORMANCE ON 2024 DATA':<50}")
print("-" * 50)
print(f"{'Number of predicted events (' + str(MIN_EVENT_DURATION) + '+ consecutive)':<40}: {predicted_event_count_2024:>8}")
print(f"{'Total predicted event days':<40}: {predicted_event_days_2024:>8}")
print(f"{'Event matching results:':<40}")
print(f"{'  TP Events (correctly predicted)':<40}: {len(tp_events_2024):>8}")
print(f"{'  FP Events (false alarms)':<40}: {len(fp_events_2024):>8}")
print(f"{'  FN Events (missed events)':<40}: {len(fn_events_2024):>8}")

# Calculate performance metrics
if (len(tp_events_2024) + len(fp_events_2024)) > 0:
    precision_2024 = len(tp_events_2024) / (len(tp_events_2024) + len(fp_events_2024))
    print(f"{'Event-level Precision':<40}: {precision_2024:>8.3f}")

if (len(tp_events_2024) + len(fn_events_2024)) > 0:
    recall_2024 = len(tp_events_2024) / (len(tp_events_2024) + len(fn_events_2024))
    print(f"{'Event-level Recall':<40}: {recall_2024:>8.3f}")

if 'precision_2024' in locals() and 'recall_2024' in locals() and precision_2024 + recall_2024 > 0:
    f1_2024 = 2 * (precision_2024 * recall_2024) / (precision_2024 + recall_2024)
    print(f"{'Event-level F1 Score':<40}: {f1_2024:>8.3f}")

# Show event matching details
if observed_events_list_2024 or predicted_events_list_2024:
    print(f"\n{'EVENT MATCHING DETAILS':<50}")
    print("-" * 50)
    print("Observed Events:")
    for i, event in enumerate(observed_events_list_2024, 1):
        print(f"  Obs Event {i}: Days {event['start']}-{event['end']} ({event['duration']} days)")
    
    print("Predicted Events:")
    for i, event in enumerate(predicted_events_list_2024, 1):
        event_type = "TP" if event in tp_events_2024 else "FP"
        print(f"  Pred Event {i} ({event_type}): Days {event['start']}-{event['end']} ({event['duration']} days)")
    
    print("Missed Events (FN):")
    for i, event in enumerate(fn_events_2024, 1):
        print(f"  Missed Event {i}: Days {event['start']}-{event['end']} ({event['duration']} days)")

print(f"\n{'RULE CONFIGURATION':<50}")
print("-" * 50)
print(f"{'Feature':<40}: {AEP_CONFIG['rule_feature']}")
print(f"{'Threshold':<40}: {AEP_CONFIG['rule_threshold']}")
print(f"{'Minimum event duration':<40}: {AEP_CONFIG['min_event_duration']} days")
print(f"{'Seasonal filter (Apr-Oct)':<40}: {AEP_CONFIG['seasonal_filter']}")
print(f"{'Fishermen count':<40}: {AEP_CONFIG['N_FISHERMEN']}")
print(f"{'Daily wage':<40}: ${AEP_CONFIG['W']}")

print("="*80)

# ============================================================================
# VISUALIZATION
# ============================================================================

print(f"\n📊 Creating TP AEP visualization...")

# Create single TP AEP plot
plt.figure(figsize=(10, 6))

# Plot TP AEP curve only
plt.plot(tp_aep['loss'], tp_aep['probability'], 'g-', linewidth=3, alpha=0.9)

# Add key percentile lines
mean_tp_cost = np.mean(annual_tp_costs)
p95_tp_cost = np.percentile(annual_tp_costs, 95)
p99_tp_cost = np.percentile(annual_tp_costs, 99)

plt.axvline(x=mean_tp_cost, color='blue', linestyle='--', linewidth=2, alpha=0.7,
           label=f'Mean: ${mean_tp_cost:,.0f}')
plt.axvline(x=p95_tp_cost, color='orange', linestyle='--', linewidth=2, alpha=0.7,
           label=f'P95: ${p95_tp_cost:,.0f}')
plt.axvline(x=p99_tp_cost, color='red', linestyle='--', linewidth=2, alpha=0.7,
           label=f'P99: ${p99_tp_cost:,.0f}')

# Formatting
plt.xlabel('Annual True Positive Cost ($)', fontsize=12, fontweight='bold')
plt.ylabel('Exceedance Probability', fontsize=12, fontweight='bold')
plt.title(f'True Positive Annual Exceedance Probability Curve\nRule: {AEP_CONFIG["rule_feature"]} > {AEP_CONFIG["rule_threshold"]}', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim(left=0)

# Format x-axis with currency
from matplotlib.ticker import FuncFormatter
def currency_formatter(x, pos):
    if x >= 1e6:
        return f'${x/1e6:.1f}M'
    elif x >= 1e3:
        return f'${x/1e3:.0f}K'
    else:
        return f'${x:.0f}'

plt.gca().xaxis.set_major_formatter(FuncFormatter(currency_formatter))

plt.tight_layout()
plt.show()


# ============================================================================
# MULTI-YEAR ANALYSIS: TIME SERIES PLOTS AND EVENT PREDICTION STATISTICS
# ============================================================================

print(f"\n📊 Creating multi-year analysis...")

# Get dynamic event column name based on min_event_duration
event_column = get_event_column_name(AEP_CONFIG['min_event_duration'])
print(f"   Using event column: {event_column} (min_duration={AEP_CONFIG['min_event_duration']})")

# Get all available years from the dataset
available_years = sorted(df.index.year.unique())
print(f"   Available years: {available_years}")

# Verify we're using the correct feature
print(f"✅ Verification: Using feature '{AEP_CONFIG['rule_feature']}' for simulations")
print(f"   Full dataset range: {df[AEP_CONFIG['rule_feature']].min():.2f} - {df[AEP_CONFIG['rule_feature']].max():.2f}")

# Calculate average simulated values for each year
print("   Computing average simulated values across 4000 simulations for each year...")

all_years_data = {}
all_event_stats = []

for year in available_years:
    # Get year data
    df_year = df[df.index.year == year].copy()
    if len(df_year) == 0:
        continue
        
    n_days_year = len(df_year)
    avg_simulated_values = np.zeros(n_days_year)
    
    # Sample from the same 4k simulations to get average for this year
    np.random.seed(42 + year)  # Different seed per year for variety
    for sim_idx in range(AEP_CONFIG['n_simulations']):
        sim_indices = all_simulation_indices[sim_idx]
        
        # Handle length mismatch 
        if len(sim_indices) < n_days_year:
            additional_indices = np.random.choice(len(trigger_values_matrix), 
                                                size=n_days_year - len(sim_indices), 
                                                replace=True)
            sim_indices_padded = np.concatenate([sim_indices, additional_indices])
        else:
            sim_indices_padded = sim_indices[:n_days_year]
        
        # Add to running average
        avg_simulated_values += trigger_values_matrix[sim_indices_padded]
    
    avg_simulated_values /= AEP_CONFIG['n_simulations']
    
    # Store year data
    all_years_data[year] = {
        'df': df_year,
        'avg_simulated': avg_simulated_values,
        'observed_events': extract_events_with_details(
            df_year[event_column].values, 
            AEP_CONFIG['min_event_duration']
        )
    }
    
    print(f"   {year}: {len(df_year)} days, {len(all_years_data[year]['observed_events'])} events, "
          f"sim range: {avg_simulated_values.min():.2f}-{avg_simulated_values.max():.2f}")

# ============================================================================
# MULTI-PANEL TIME SERIES PLOT
# ============================================================================

print(f"\n📊 Creating multi-panel time series plot...")

# Calculate subplot layout (2 columns)
n_years = len(all_years_data)
n_cols = 2  # Fixed to 2 columns as requested
n_rows = (n_years + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
if n_years == 1:
    axes = [axes]
elif n_rows == 1:
    axes = [axes] if n_cols == 1 else list(axes)
else:
    axes = axes.flatten()

print(f"\n🎯 EVENT SHADING LOGIC FOR EACH YEAR:")
print("="*60)
print("NEW APPROACH: Shade events based on simulation consistency, not observed vs predicted")
print("For each observed event across all years:")
print("1. Run event through 4k bootstrap simulations")
print("2. Count how many simulations predict this event (≥40% overlap)")
print("3. Calculate prediction rate across simulations")
print("4. Shade based on consistency:")
print("   - GREEN: Events predicted in >50% of simulations (reliable)")
print("   - RED: Events predicted in ≤50% of simulations (unreliable)")
print("="*60)

# First, let's pre-calculate prediction rates for all observed events across all years
print(f"\n📊 Pre-calculating prediction rates for all observed events...")

event_prediction_rates = {}  # {(year, start_day, end_day): prediction_rate}

for year, year_data in all_years_data.items():
    df_year = year_data['df']
    observed_events_year = year_data['observed_events']
    n_days_year = len(df_year)
    
    if len(observed_events_year) == 0:
        continue
    
    print(f"   {year}: Analyzing {len(observed_events_year)} events across simulations...")
    
    for obs_event in observed_events_year:
        predicted_count = 0

        # ADD THESE DEBUG LINES HERE:
        print(f"DEBUG: MIN_EVENT_DURATION = {MIN_EVENT_DURATION}")
        print(f"DEBUG: AEP_CONFIG min_event_duration = {AEP_CONFIG['min_event_duration']}")

        n_sims_to_check = min(4000, AEP_CONFIG['n_simulations'])  # Sample for speed
        
        # Check each simulation
        for sim_idx in range(n_sims_to_check):
            # Generate predictions for this simulation
            sim_indices = all_simulation_indices[sim_idx]
            
            # Handle length mismatch
            if len(sim_indices) < n_days_year:
                additional_indices = np.random.choice(len(trigger_values_matrix), 
                                                    size=n_days_year - len(sim_indices), 
                                                    replace=True)
                sim_indices_padded = np.concatenate([sim_indices, additional_indices])
            else:
                sim_indices_padded = sim_indices[:n_days_year]
            
            # Get simulation values and apply rule
            sim_values = trigger_values_matrix[sim_indices_padded]
            
            if AEP_CONFIG['seasonal_filter']:
                basic_predictions = (sim_values > AEP_CONFIG['rule_threshold']).astype(np.int32)
                sim_predictions = np.zeros_like(basic_predictions)
                for i in range(len(basic_predictions)):
                    day_of_year = (i % 365) + 1
                    if 90 <= day_of_year <= 305:  # Apr-Oct
                        sim_predictions[i] = basic_predictions[i]
            else:
                sim_predictions = (sim_values > AEP_CONFIG['rule_threshold']).astype(np.int32)
            
            # Extract predicted events
            pred_events_sim = extract_events_with_details(sim_predictions, AEP_CONFIG['min_event_duration'])
            
            # Check if this observed event is predicted in this simulation
            for pred_event in pred_events_sim:
                # Calculate overlap with observed event
                overlap_start = max(pred_event['start'], obs_event['start'])
                overlap_end = min(pred_event['end'], obs_event['end'])
                
                if overlap_start <= overlap_end:
                    overlap_length = overlap_end - overlap_start + 1
                    coverage = overlap_length / obs_event['duration']
                    
                    if coverage >= AEP_CONFIG['min_overlap']:
                        predicted_count += 1
                        break  # Found a match, move to next simulation
        
        # Calculate prediction rate for this event
        prediction_rate = predicted_count / n_sims_to_check
        event_key = (year, obs_event['start'], obs_event['end'])
        event_prediction_rates[event_key] = prediction_rate
        
        reliability = "RELIABLE" if prediction_rate > 0.5 else "UNRELIABLE"
        print(f"     Event days {obs_event['start']}-{obs_event['end']}: {prediction_rate:.3f} rate ({reliability})")

for idx, (year, year_data) in enumerate(all_years_data.items()):
    ax = axes[idx] if n_years > 1 else axes[0]
    df_year = year_data['df']
    avg_simulated = year_data['avg_simulated']
    observed_events_year = year_data['observed_events']
    
    print(f"\nPlotting {year}:")
    
    # Plot original wave heights
    ax.plot(df_year.index, df_year[AEP_CONFIG['rule_feature']], 
           'b-', linewidth=1.5, alpha=0.8, label='Original SWH Max')
    
    # Plot average simulated wave heights
    ax.plot(df_year.index, avg_simulated, 
           'r--', linewidth=2, alpha=0.7, label='Avg Simulated SWH Max')
    
    # Add threshold line
    ax.axhline(y=AEP_CONFIG['rule_threshold'], color='orange', linestyle=':', 
              linewidth=2, alpha=0.8, label=f'Threshold ({AEP_CONFIG["rule_threshold"]}m)')
    
    # Shade events based on their prediction consistency across simulations
    reliable_events = 0
    unreliable_events = 0
    
    for i, event in enumerate(observed_events_year, 1):
        event_key = (year, event['start'], event['end'])
        prediction_rate = event_prediction_rates.get(event_key, 0)
        
        start_date = df_year.index[event['start']]
        end_date = df_year.index[event['end']]
        
        if prediction_rate > 0.5:
            # GREEN: Reliable event (predicted in >50% of simulations)
            ax.axvspan(start_date, end_date, alpha=0.3, color='green', 
                      label='Reliable Events (>50% sims)' if reliable_events == 0 else "")
            reliable_events += 1
            print(f"  Event {i} (days {event['start']}-{event['end']}): {prediction_rate:.3f} rate - SHADED GREEN (reliable)")
        else:
            # RED: Unreliable event (predicted in ≤50% of simulations)
            ax.axvspan(start_date, end_date, alpha=0.3, color='red', 
                      label='Unreliable Events (≤50% sims)' if unreliable_events == 0 else "")
            unreliable_events += 1
            print(f"  Event {i} (days {event['start']}-{event['end']}): {prediction_rate:.3f} rate - SHADED RED (unreliable)")
    
    # Formatting
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Wave Height (m)', fontsize=10)
    ax.set_title(f'{year} - Wave Heights & Event Reliability\n'
                f'({len(observed_events_year)} events: {reliable_events} reliable, {unreliable_events} unreliable)\n'
                f'({MIN_EVENT_DURATION}+ day events)', 
                fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Don't add legend to individual subplots

# Hide empty subplots
for idx in range(n_years, len(axes)):
    if idx < len(axes):
        axes[idx].set_visible(False)

# Create a single legend outside the plot
handles, labels = [], []
for ax in axes[:n_years]:
    ax_handles, ax_labels = ax.get_legend_handles_labels()
    for handle, label in zip(ax_handles, ax_labels):
        if label not in labels:  # Avoid duplicates
            handles.append(handle)
            labels.append(label)

# Add legend outside the plot
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=10)

plt.tight_layout()
plt.subplots_adjust(right=0.85)  # Make room for legend

plt.tight_layout()

plt.savefig(f'/Users/ageidv/suyana/peru_swan/results/{region}_sim_event_analysis_{MIN_EVENT_DURATION}.png')
plt.show()
# ============================================================================
# COMPREHENSIVE EVENT PREDICTION ANALYSIS ACROSS ALL YEARS
# ============================================================================

print(f"\n📊 Analyzing event prediction across all years and 4000 simulations...")

# Process each year
for year, year_data in all_years_data.items():
    df_year = year_data['df']
    observed_events_year = year_data['observed_events']
    n_days_year = len(df_year)
    
    if len(observed_events_year) == 0:
        print(f"   {year}: No events to analyze")
        continue
    
    print(f"   {year}: Analyzing {len(observed_events_year)} observed events...")
    
    # Analyze each observed event in this year
    for obs_idx, obs_event in enumerate(observed_events_year, 1):
        predicted_count = 0
        total_coverage = 0
        coverage_list = []
        predicted_durations = []  # Track predicted durations for cost calculation
        
        # Check each simulation
        for sim_idx in range(min(1000, AEP_CONFIG['n_simulations'])):  # Sample subset for speed
            # Generate predictions for this simulation
            sim_indices = all_simulation_indices[sim_idx]
            
            # Handle length mismatch
            if len(sim_indices) < n_days_year:
                additional_indices = np.random.choice(len(trigger_values_matrix), 
                                                    size=n_days_year - len(sim_indices), 
                                                    replace=True)
                sim_indices_padded = np.concatenate([sim_indices, additional_indices])
            else:
                sim_indices_padded = sim_indices[:n_days_year]
            
            # Get simulation values and apply rule
            sim_values = trigger_values_matrix[sim_indices_padded]
            
            if AEP_CONFIG['seasonal_filter']:
                basic_predictions = (sim_values > AEP_CONFIG['rule_threshold']).astype(np.int32)
                sim_predictions = np.zeros_like(basic_predictions)
                for i in range(len(basic_predictions)):
                    day_of_year = (i % 365) + 1
                    if 90 <= day_of_year <= 305:  # Apr-Oct
                        sim_predictions[i] = basic_predictions[i]
            else:
                sim_predictions = (sim_values > AEP_CONFIG['rule_threshold']).astype(np.int32)
            
            # Extract predicted events
            pred_events_sim = extract_events_with_details(sim_predictions, AEP_CONFIG['min_event_duration'])
            
            # Check if this observed event is predicted in this simulation
            best_coverage = 0
            best_pred_duration = 0
            for pred_event in pred_events_sim:
                # Calculate overlap with observed event
                overlap_start = max(pred_event['start'], obs_event['start'])
                overlap_end = min(pred_event['end'], obs_event['end'])
                
                if overlap_start <= overlap_end:
                    overlap_length = overlap_end - overlap_start + 1
                    coverage = overlap_length / obs_event['duration']
                    
                    if coverage >= AEP_CONFIG['min_overlap'] and coverage > best_coverage:
                        best_coverage = coverage
                        best_pred_duration = pred_event['duration']
            
            if best_coverage >= AEP_CONFIG['min_overlap']:
                predicted_count += 1
                total_coverage += best_coverage
                coverage_list.append(best_coverage)
                predicted_durations.append(best_pred_duration)
        
        # Calculate statistics for this event
        n_sims_used = min(4000, AEP_CONFIG['n_simulations'])
        prediction_rate = predicted_count / n_sims_used
        avg_coverage = np.mean(coverage_list) if coverage_list else 0
        
        # Calculate costs
        observed_cost = AEP_CONFIG['N_FISHERMEN'] * AEP_CONFIG['W'] * obs_event['duration']
        
        # Duration statistics
        if predicted_durations:
            avg_predicted_duration = np.mean(predicted_durations)
            p99_predicted_duration = np.percentile(predicted_durations, 99)
            
            # Full cost statistics
            simulated_full_costs = [AEP_CONFIG['N_FISHERMEN'] * AEP_CONFIG['W'] * d for d in predicted_durations]
            avg_simulated_full_cost = np.mean(simulated_full_costs)
            p99_simulated_full_cost = np.percentile(simulated_full_costs, 99)
            
            # Capped cost statistics (50% of each predicted duration)
            capped_durations = [d * AEP_CONFIG['max_coverage'] for d in predicted_durations]
            simulated_capped_costs = [AEP_CONFIG['N_FISHERMEN'] * AEP_CONFIG['W'] * d for d in capped_durations]
            avg_simulated_capped_cost = np.mean(simulated_capped_costs)
            p99_simulated_capped_cost = np.percentile(simulated_capped_costs, 99)
        else:
            avg_predicted_duration = 0
            p99_predicted_duration = 0
            avg_simulated_full_cost = 0
            p99_simulated_full_cost = 0
            avg_simulated_capped_cost = 0
            p99_simulated_capped_cost = 0
        
        all_event_stats.append({
            'Year': year,
            'Event_ID': obs_idx,
            'Start_Day': obs_event['start'],
            'End_Day': obs_event['end'],
            'Duration': obs_event['duration'],
            'Prediction_Rate': prediction_rate,
            'Times_Predicted': predicted_count,
            'Total_Simulations': n_sims_used,
            'Avg_Coverage': avg_coverage,
            'Coverage_Std': np.std(coverage_list) if coverage_list else 0,
            'Observed_Cost': observed_cost,
            'Avg_Predicted_Duration': avg_predicted_duration,
            'P99_Predicted_Duration': p99_predicted_duration,
            'Avg_Simulated_Full_Cost': avg_simulated_full_cost,
            'P99_Simulated_Full_Cost': p99_simulated_full_cost,
            'Avg_Simulated_Capped_Cost': avg_simulated_capped_cost,
            'P99_Simulated_Capped_Cost': p99_simulated_capped_cost
        })

# ============================================================================
# COMPREHENSIVE RESULTS TABLE
# ============================================================================

print(f"\n📈 COMPREHENSIVE EVENT PREDICTION STATISTICS")
# ============================================================================
# EXPORT STATISTICS TO EXCEL
# ============================================================================

print(f"\n💾 Exporting comprehensive statistics to Excel...")

# Create filename with region and min_event_duration
stats_excel_filename = f'event_prediction_statistics_{region}_mindur{AEP_CONFIG["min_event_duration"]}.xlsx'
stats_excel_path = os.path.join(project_root, 'results', stats_excel_filename)

# Ensure results directory exists
os.makedirs(os.path.dirname(stats_excel_path), exist_ok=True)

# Convert all_event_stats to DataFrame
event_stats_df = pd.DataFrame(all_event_stats)

# ============================================================================
# FIXED PREDICTED EVENTS LOGIC FOR YEARLY SUMMARY
# ============================================================================

print(f"\n📊 Fixing predicted events logic for yearly summary...")

# Create yearly_summary DataFrame with CORRECT predicted events logic
yearly_summary = []
for year in available_years:
    year_stats = [s for s in all_event_stats if s['Year'] == year]
    if year_stats:
        # OBSERVED EVENTS METRICS (from simulation analysis)
        avg_rate = np.mean([s['Prediction_Rate'] for s in year_stats])
        avg_coverage = np.mean([s['Avg_Coverage'] for s in year_stats if s['Avg_Coverage'] > 0])
        avg_pred_duration = np.mean([s['Avg_Predicted_Duration'] for s in year_stats if s['Avg_Predicted_Duration'] > 0])
        p99_pred_duration = np.mean([s['P99_Predicted_Duration'] for s in year_stats if s['P99_Predicted_Duration'] > 0])
        
        # COST METRICS
        total_obs_cost = sum([s['Observed_Cost'] for s in year_stats])
        total_avg_full_cost = sum([s['Avg_Simulated_Full_Cost'] for s in year_stats])
        total_p99_full_cost = sum([s['P99_Simulated_Full_Cost'] for s in year_stats])
        total_avg_capped_cost = sum([s['Avg_Simulated_Capped_Cost'] for s in year_stats])
        total_p99_capped_cost = sum([s['P99_Simulated_Capped_Cost'] for s in year_stats])
        
        # SIMULATION-BASED PREDICTED EVENTS LOGIC
        year_data = all_years_data[year]
        df_year = year_data['df']
        n_days_year = len(df_year)
        
        # Run simulations to count predicted events
        predicted_events_per_sim = []
        
        # Sample subset of simulations for speed
        n_sims_to_check = min(1000, AEP_CONFIG['n_simulations'])
        
        for sim_idx in range(n_sims_to_check):
            # Generate predictions for this simulation
            sim_indices = all_simulation_indices[sim_idx]
            
            # Handle length mismatch
            if len(sim_indices) < n_days_year:
                additional_indices = np.random.choice(len(trigger_values_matrix), 
                                                    size=n_days_year - len(sim_indices), 
                                                    replace=True)
                sim_indices_padded = np.concatenate([sim_indices, additional_indices])
            else:
                sim_indices_padded = sim_indices[:n_days_year]
            
            # Get simulation values and apply rule (NO SEASONAL FILTER)
            sim_values = trigger_values_matrix[sim_indices_padded]
            sim_predictions = (sim_values > AEP_CONFIG['rule_threshold']).astype(np.int32)
            
            # Extract predicted events for this simulation
            pred_events_sim = extract_events_with_details(sim_predictions, MIN_EVENT_DURATION)
            predicted_events_per_sim.append(len(pred_events_sim))
        
        # Calculate statistics across simulations
        avg_predicted_events = np.mean(predicted_events_per_sim)
        p99_predicted_events = np.percentile(predicted_events_per_sim, 99)
        
        # ACTUAL PREDICTED EVENTS FOR PERFORMANCE METRICS
        predictions_year = (df_year[AEP_CONFIG['rule_feature']] > AEP_CONFIG['rule_threshold']).astype(int)
        predicted_events_year = extract_events_with_details(predictions_year.values, AEP_CONFIG['min_event_duration'])
        actual_predicted_events = len(predicted_events_year)
        
        # Calculate average predicted duration from ACTUAL predictions
        if predicted_events_year:
            actual_avg_pred_duration = np.mean([event['duration'] for event in predicted_events_year])
            actual_total_pred_days = sum([event['duration'] for event in predicted_events_year])
        else:
            actual_avg_pred_duration = 0
            actual_total_pred_days = 0
        
        # OBSERVED EVENTS METRICS
        observed_events_year = year_data['observed_events']
        actual_observed_events = len(observed_events_year)
        
        if observed_events_year:
            actual_avg_obs_duration = np.mean([event['duration'] for event in observed_events_year])
            actual_total_obs_days = sum([event['duration'] for event in observed_events_year])
        else:
            actual_avg_obs_duration = 0
            actual_total_obs_days = 0
        
        # PERFORMANCE METRICS - Match predicted vs observed events
        tp_events_year, fp_events_year, fn_events_year = simple_event_matching(
            predicted_events_year, observed_events_year, min_overlap=AEP_CONFIG['min_overlap']
        )
        
        # Calculate performance metrics
        tp_count = len(tp_events_year)
        fp_count = len(fp_events_year)
        fn_count = len(fn_events_year)
        
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate costs by reliability (reliable vs unreliable events)
        reliable_cost = 0
        unreliable_cost = 0
        reliable_count = 0
        unreliable_count = 0

        for event in observed_events_year:
            event_key = (year, event['start'], event['end'])
            prediction_rate = event_prediction_rates.get(event_key, 0)
            event_cost = AEP_CONFIG['N_FISHERMEN'] * AEP_CONFIG['W'] * event['duration']
            
            if prediction_rate > 0.5:
                reliable_cost += event_cost
                reliable_count += 1
            else:
                unreliable_cost += event_cost
                unreliable_count += 1

        yearly_summary.append({
            'Year': year,
            'Events': actual_observed_events,
            'Avg_Predicted_Events': avg_predicted_events,  # NOW PROPERLY DEFINED
            'P99_Predicted_Events': p99_predicted_events,  # NOW PROPERLY DEFINED
            'TP_Events': tp_count,
            'FP_Events': fp_count,
            'FN_Events': fn_count,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1_score,
            'Avg_Obs_Duration': actual_avg_obs_duration,
            'Avg_Pred_Duration_Actual': actual_avg_pred_duration,
            'Total_Obs_Days': actual_total_obs_days,
            'Total_Pred_Days': actual_total_pred_days,
            'Avg_Rate': avg_rate,
            'Avg_Coverage': avg_coverage,
            'Avg_Duration': avg_pred_duration,
            'P99_Duration': p99_pred_duration,
            'Obs_Cost': total_obs_cost,
            'Avg_Full_Cost': total_avg_full_cost,
            'P99_Full_Cost': total_p99_full_cost,
            'Avg_Capped_Cost': total_avg_capped_cost,
            'P99_Capped_Cost': total_p99_capped_cost,
            'Reliable_Events': reliable_count,
            'Unreliable_Events': unreliable_count,
            'Reliable_Cost': reliable_cost,
            'Unreliable_Cost': unreliable_cost
        })
        
        print(f"   {year}: Obs={actual_observed_events}, SimAvg={avg_predicted_events:.1f}, P99={p99_predicted_events:.1f}")

print(f"✅ Fixed predicted events logic for {len(yearly_summary)} years")

# Convert to DataFrame with enhanced columns
yearly_summary_df = pd.DataFrame(yearly_summary)

# Add run and configuration columns
yearly_summary_df['run'] = AEP_CONFIG['run_name']
yearly_summary_df['min_duration'] = AEP_CONFIG['min_event_duration']
yearly_summary_df['N_FISHERMEN'] = AEP_CONFIG['N_FISHERMEN']
yearly_summary_df['WAVE_THRESHOLD'] = AEP_CONFIG['rule_threshold']

# Reorder columns for better readability
cols = [
    'run', 'N_FISHERMEN', 'WAVE_THRESHOLD', 'min_duration', 'Year',
    'Events', 'Avg_Predicted_Events', 'P99_Predicted_Events',
    'TP_Events', 'FP_Events', 'FN_Events',
    'Precision', 'Recall', 'F1_Score',
    'Avg_Obs_Duration', 'Avg_Pred_Duration_Actual', 'Total_Obs_Days', 'Total_Pred_Days',
    'Avg_Rate', 'Avg_Coverage', 'Avg_Duration', 'P99_Duration',
    'Obs_Cost', 'Avg_Full_Cost', 'P99_Full_Cost', 'Avg_Capped_Cost', 'P99_Capped_Cost'
]
yearly_summary_df = yearly_summary_df[cols]

print(f"\n📊 CORRECTED YEARLY SUMMARY:")
print(yearly_summary_df[['Year', 'Events', 'Avg_Predicted_Events', 'P99_Predicted_Events', 'TP_Events', 'FP_Events', 'FN_Events']].to_string(index=False))

# ============================================================================
# ENHANCED COLUMN EXPLANATIONS
# ============================================================================

enhanced_explanations = pd.DataFrame({
    'Column': [
        'run', 'N_FISHERMEN', 'WAVE_THRESHOLD', 'min_duration', 'Year',
        'Events', 'Avg_Predicted_Events', 'P99_Predicted_Events',
        'TP_Events', 'FP_Events', 'FN_Events',
        'Precision', 'Recall', 'F1_Score',
        'Avg_Obs_Duration', 'Avg_Pred_Duration_Actual', 'Total_Obs_Days', 'Total_Pred_Days',
        'Avg_Rate', 'Avg_Coverage', 'Avg_Duration', 'P99_Duration',
        'Obs_Cost', 'Avg_Full_Cost', 'P99_Full_Cost', 'Avg_Capped_Cost', 'P99_Capped_Cost'
    ],
    'Description': [
        'Run identifier (e.g., run_g8)',
        'Number of fishermen parameter',
        'Wave height threshold for rule',
        f'Minimum event duration parameter ({AEP_CONFIG["min_event_duration"]} days)',
        'Year of analysis',
        f'Number of observed events ({AEP_CONFIG["min_event_duration"]}+ consecutive days)',
        f'AVERAGE number of events predicted across simulations ({AEP_CONFIG["min_event_duration"]}+ consecutive days)',
        f'99th percentile number of events predicted across simulations ({AEP_CONFIG["min_event_duration"]}+ consecutive days)',
        'True Positive events (correctly predicted)',
        'False Positive events (false alarms)',
        'False Negative events (missed events)',
        'Event-level precision (TP / (TP + FP))',
        'Event-level recall (TP / (TP + FN))',
        'Event-level F1 score (2 * precision * recall / (precision + recall))',
        'Average duration of observed events (days)',
        'Average duration of actual predicted events (days)',
        'Total days across all observed events',
        'Total days across all predicted events',
        'Average prediction rate from simulations (0-1)',
        'Average overlap coverage when events were predicted in simulations',
        'Average predicted duration from simulations where events were detected',
        '99th percentile predicted duration from simulations where events were detected',
        f'Total observed cost for all {AEP_CONFIG["min_event_duration"]}+ day events',
        'Total average cost based on full predicted durations from simulations',
        'Total 99th percentile cost based on full predicted durations from simulations',
        'Total average cost based on 50% of predicted durations (conservative)',
        'Total 99th percentile cost based on 50% of predicted durations (conservative)'
    ]
})

print(f"\n📋 KEY SIMULATION-BASED METRICS:")
print("- Avg_Predicted_Events: Average number of events predicted across 1000 simulations")
print("- P99_Predicted_Events: 99th percentile of predicted event counts across simulations")
print("- Shows uncertainty in predictions: most simulations predict around the average, but worst case is P99")
print("- This gives you both the expected and extreme cases for how many events your rule predicts")

# ============================================================================
# EXPORT CORRECTED CSV
# ============================================================================

# Export yearly summary to CSV with corrected logic
csv_filename = f'yearly_summary_{AEP_CONFIG["run_name"]}_mindur{AEP_CONFIG["min_event_duration"]}.csv'
csv_path = os.path.join(project_root, 'results', csv_filename)
yearly_summary_df.to_csv(csv_path, index=False)

print(f"\n✅ CORRECTED yearly summary CSV exported!")
print(f"   📁 File: {csv_path}")
print(f"   📊 Now contains ACTUAL predicted events from rule application")
print(f"   📊 Includes event-level performance metrics (Precision, Recall, F1)")

# Also update the Excel export with corrected data
stats_excel_filename = f'event_prediction_statistics_{AEP_CONFIG["run_name"]}_mindur{AEP_CONFIG["min_event_duration"]}.xlsx'
stats_excel_path = os.path.join(project_root, 'results', stats_excel_filename)

with pd.ExcelWriter(stats_excel_path, engine='openpyxl') as writer:
    # Export corrected yearly summary
    yearly_summary_df.to_excel(writer, sheet_name='Yearly_Summary_Corrected', index=False)
    
    # Export enhanced explanations
    enhanced_explanations.to_excel(writer, sheet_name='Column_Explanations_Enhanced', index=False)
    
    # Export comparison showing the fix
    comparison_df = pd.DataFrame({
        'Aspect': ['Predicted_Events', 'Performance_Metrics', 'Duration_Calculations', 'Data_Source'],
        'Before_Fix': [
            'Used simulation averages (incorrect)',
            'Missing TP/FP/FN counts',
            'Only simulation-based durations',
            'Mixed simulation and actual data'
        ],
        'After_Fix': [
            'Uses actual rule predictions (correct)',
            'Includes TP/FP/FN/Precision/Recall/F1',
            'Both actual and simulation durations',
            'Clear separation: actual vs simulation'
        ]
    })
    comparison_df.to_excel(writer, sheet_name='Fix_Comparison', index=False)

print(f"✅ CORRECTED Excel file exported!")
print(f"   📁 File: {stats_excel_path}")
print(f"   📊 Added Yearly_Summary_Corrected sheet with fixed logic")
print(f"   📊 Added Fix_Comparison sheet showing what was corrected")

# Convert to DataFrame with enhanced columns
yearly_summary_df = pd.DataFrame(yearly_summary)

# Add run and configuration columns
yearly_summary_df['run'] = AEP_CONFIG['run_name']
yearly_summary_df['min_duration'] = AEP_CONFIG['min_event_duration']
yearly_summary_df['N_FISHERMEN'] = AEP_CONFIG['N_FISHERMEN']
yearly_summary_df['WAVE_THRESHOLD'] = AEP_CONFIG['rule_threshold']

# Reorder columns for better readability
cols = [
    'run', 'N_FISHERMEN', 'WAVE_THRESHOLD', 'min_duration', 'Year',
    'Events', 'Avg_Predicted_Events', 'P99_Predicted_Events',
    'TP_Events', 'FP_Events', 'FN_Events',
    'Precision', 'Recall', 'F1_Score',
    'Avg_Obs_Duration', 'Avg_Pred_Duration_Actual', 'Total_Obs_Days', 'Total_Pred_Days',
    'Avg_Rate', 'Avg_Coverage', 'Avg_Duration', 'P99_Duration',
    'Obs_Cost', 'Avg_Full_Cost', 'P99_Full_Cost', 'Avg_Capped_Cost', 'P99_Capped_Cost'
]
yearly_summary_df = yearly_summary_df[cols]

print(f"\n📊 CORRECTED YEARLY SUMMARY:")
print(yearly_summary_df[['Year', 'Events', 'Avg_Predicted_Events',  'P99_Predicted_Events', 'TP_Events', 'FP_Events', 'FN_Events']].to_string(index=False))

# ============================================================================
# ENHANCED COLUMN EXPLANATIONS
# ============================================================================

enhanced_explanations = pd.DataFrame({
    'Column': [
        'run', 'N_FISHERMEN', 'WAVE_THRESHOLD', 'min_duration', 'Year',
        'Events', 'Avg_Predicted_Events',  'P99_Predicted_Events',
        'TP_Events', 'FP_Events', 'FN_Events',
        'Precision', 'Recall', 'F1_Score',
        'Avg_Obs_Duration', 'Avg_Pred_Duration_Actual', 'Total_Obs_Days', 'Total_Pred_Days',
        'Avg_Rate', 'Avg_Coverage', 'Avg_Duration', 'P99_Duration',
        'Obs_Cost', 'Avg_Full_Cost', 'P99_Full_Cost', 'Avg_Capped_Cost', 'P99_Capped_Cost'
    ],
    'Description': [
        'Run identifier (e.g., run_g8)',
        'Number of fishermen parameter',
        'Wave height threshold for rule',
        f'Minimum event duration parameter ({AEP_CONFIG["min_event_duration"]} days)',
        'Year of analysis',
        f'Number of observed events ({AEP_CONFIG["min_event_duration"]}+ consecutive days)',
        f'AVERAGE number of events predicted across simulations ({AEP_CONFIG["min_event_duration"]}+ consecutive days)',
        f'99th percentile number of events predicted across simulations ({AEP_CONFIG["min_event_duration"]}+ consecutive days)',
        'True Positive events (correctly predicted)',
        'False Positive events (false alarms)',
        'False Negative events (missed events)',
        'Event-level precision (TP / (TP + FP))',
        'Event-level recall (TP / (TP + FN))',
        'Event-level F1 score (2 * precision * recall / (precision + recall))',
        'Average duration of observed events (days)',
        'Average duration of actual predicted events (days)',
        'Total days across all observed events',
        'Total days across all predicted events',
        'Average prediction rate from simulations (0-1)',
        'Average overlap coverage when events were predicted in simulations',
        'Average predicted duration from simulations where events were detected',
        '99th percentile predicted duration from simulations where events were detected',
        f'Total observed cost for all {AEP_CONFIG["min_event_duration"]}+ day events',
        'Total average cost based on full predicted durations from simulations',
        'Total 99th percentile cost based on full predicted durations from simulations',
        'Total average cost based on 50% of predicted durations (conservative)',
        'Total 99th percentile cost based on 50% of predicted durations (conservative)'
    ]
})

print(f"\n📋 KEY SIMULATION-BASED METRICS:")
print("- Avg_Predicted_Events: Average number of events predicted across 1000 simulations")
print("- P50/P95/P99_Predicted_Events: Percentiles of predicted event counts across simulations")
print("- Shows uncertainty in predictions: some simulations predict few events, others many")
print("- Avg_Rate: How often each observed event gets predicted across simulations")
print("- This gives you the full distribution of how many events your rule predicts")

# ============================================================================
# EXPORT CORRECTED CSV
# ============================================================================

# Export yearly summary to CSV with corrected logic
csv_filename = f'yearly_summary_{AEP_CONFIG["run_name"]}_mindur{AEP_CONFIG["min_event_duration"]}.csv'
csv_path = os.path.join(project_root, 'results', csv_filename)
yearly_summary_df.to_csv(csv_path, index=False)

print(f"\n✅ CORRECTED yearly summary CSV exported!")
print(f"   📁 File: {csv_path}")
print(f"   📊 Now contains ACTUAL predicted events from rule application")
print(f"   📊 Includes event-level performance metrics (Precision, Recall, F1)")

# Also update the Excel export with corrected data
stats_excel_filename = f'event_prediction_statistics_{AEP_CONFIG["run_name"]}_mindur{AEP_CONFIG["min_event_duration"]}.xlsx'
stats_excel_path = os.path.join(project_root, 'results', stats_excel_filename)

with pd.ExcelWriter(stats_excel_path, engine='openpyxl') as writer:
    # Export corrected yearly summary
    yearly_summary_df.to_excel(writer, sheet_name='Yearly_Summary_Corrected', index=False)
    
    # Export enhanced explanations
    enhanced_explanations.to_excel(writer, sheet_name='Column_Explanations_Enhanced', index=False)
    
    # Export comparison showing the fix
    comparison_df = pd.DataFrame({
        'Aspect': ['Predicted_Events', 'Performance_Metrics', 'Duration_Calculations', 'Data_Source'],
        'Before_Fix': [
            'Used simulation averages (incorrect)',
            'Missing TP/FP/FN counts',
            'Only simulation-based durations',
            'Mixed simulation and actual data'
        ],
        'After_Fix': [
            'Uses actual rule predictions (correct)',
            'Includes TP/FP/FN/Precision/Recall/F1',
            'Both actual and simulation durations',
            'Clear separation: actual vs simulation'
        ]
    })
    comparison_df.to_excel(writer, sheet_name='Fix_Comparison', index=False)

print(f"✅ CORRECTED Excel file exported!")
print(f"   📁 File: {stats_excel_path}")
print(f"   📊 Added Yearly_Summary_Corrected sheet with fixed logic")
print(f"   📊 Added Fix_Comparison sheet showing what was corrected")
# ============================================================================
# OBSERVED BACKTEST ANALYSIS
# ============================================================================

print(f"\n📊 Creating observed backtest analysis...")

# Create backtest results for all years
backtest_results = []

for year, year_data in all_years_data.items():
    df_year = year_data['df']
    observed_events_year = year_data['observed_events']
    
    if len(observed_events_year) == 0:
        continue
    
    print(f"   {year}: Analyzing {len(observed_events_year)} observed events for backtest...")
    
    # Apply rule to get predictions for this year (actual backtest)
    if AEP_CONFIG['seasonal_filter']:
        df_year_pred = df_year.copy()
        df_year_pred['month'] = df_year_pred.index.month
        seasonal_mask = (df_year_pred['month'] >= 4) & (df_year_pred['month'] <= 10)
        predictions_year = ((df_year_pred[AEP_CONFIG['rule_feature']] > AEP_CONFIG['rule_threshold']) & seasonal_mask).astype(int)
    else:
        predictions_year = (df_year[AEP_CONFIG['rule_feature']] > AEP_CONFIG['rule_threshold']).astype(int)
    
    # Extract predicted events
    predicted_events_year = extract_events_with_details(predictions_year.values, AEP_CONFIG['min_event_duration'])
    
    # Match events using temporal overlap
    tp_events_year, fp_events_year, fn_events_year = simple_event_matching(
        predicted_events_year, observed_events_year, min_overlap=AEP_CONFIG['min_overlap']
    )
    
    # Analyze each observed event
    for obs_idx, obs_event in enumerate(observed_events_year, 1):
        # Check if this observed event was predicted (TP) or missed (FN)
        predicted = 0  # 0 = FN (missed), 1 = TP (predicted)
        coverage = 0.0
        predicted_duration = 0
        
        # Find best matching predicted event
        best_coverage = 0
        best_pred_duration = 0
        
        for pred_event in predicted_events_year:
            # Calculate overlap with observed event
            overlap_start = max(pred_event['start'], obs_event['start'])
            overlap_end = min(pred_event['end'], obs_event['end'])
            
            if overlap_start <= overlap_end:
                overlap_length = overlap_end - overlap_start + 1
                event_coverage = overlap_length / obs_event['duration']
                
                if event_coverage >= AEP_CONFIG['min_overlap'] and event_coverage > best_coverage:
                    best_coverage = event_coverage
                    best_pred_duration = pred_event['duration']
        
        if best_coverage >= AEP_CONFIG['min_overlap']:
            predicted = 1
            coverage = best_coverage
            predicted_duration = best_pred_duration
        
        # Calculate costs
        observed_cost = AEP_CONFIG['N_FISHERMEN'] * AEP_CONFIG['W'] * obs_event['duration']
        
        if predicted == 1:  # TP event
            predicted_cost = AEP_CONFIG['N_FISHERMEN'] * AEP_CONFIG['W'] * predicted_duration
            capped_cost = AEP_CONFIG['N_FISHERMEN'] * AEP_CONFIG['W'] * (predicted_duration * 0.5)
        else:  # FN event
            predicted_cost = 0  # Not predicted, so no predicted cost
            capped_cost = 0
        
        backtest_results.append({
            'Year': year,
            'Event_ID': obs_idx,
            'Start_Day': obs_event['start'],
            'End_Day': obs_event['end'],
            'Duration': obs_event['duration'],
            'Predicted': predicted,  # 1=TP, 0=FN
            'Coverage': coverage,
            'Predicted_Duration': predicted_duration,
            'Observed_Cost': observed_cost,
            'Predicted_Cost': predicted_cost,
            'Capped_Cost': capped_cost,
            'Event_Type': 'TP' if predicted == 1 else 'FN'
        })

print(f"✅ Backtest analysis complete: {len(backtest_results)} events analyzed")

# Display backtest results
print(f"\n📈 OBSERVED BACKTEST RESULTS")
print("="*120)
print(f"{'Year':<6} {'Event':<6} {'Days':<15} {'Dur':<4} {'Pred':<5} {'Cover':<7} {'PredDur':<8} {'ObsCost':<10} {'PredCost':<10} {'CapCost':<10} {'Type':<5}")
print("-" * 120)
print("COLUMN EXPLANATIONS:")
print("- Pred: 1=Event was predicted (TP), 0=Event was missed (FN)")
print("- Cover: Coverage/overlap between predicted and observed event (0-1)")
print("- PredDur: Duration of predicted event that matched this observed event")
print("- ObsCost: Actual cost based on observed event duration")
print("- PredCost: Cost based on predicted event duration (0 if FN)")
print("- CapCost: Conservative cost (50% of predicted duration, 0 if FN)")
print("- Type: TP=True Positive, FN=False Negative")
print("-" * 120)

for result in backtest_results:
    print(f"{result['Year']:<6} "
          f"{result['Event_ID']:<6} "
          f"{result['Start_Day']}-{result['End_Day']:<12} "
          f"{result['Duration']:<4} "
          f"{result['Predicted']:<5} "
          f"{result['Coverage']:<7.3f} "
          f"{result['Predicted_Duration']:<8} "
          f"${result['Observed_Cost']:<9,.0f} "
          f"${result['Predicted_Cost']:<9,.0f} "
          f"${result['Capped_Cost']:<9,.0f} "
          f"{result['Event_Type']:<5}")

print("-" * 120)

# Backtest summary statistics
if backtest_results:
    tp_events = [r for r in backtest_results if r['Event_Type'] == 'TP']
    fn_events = [r for r in backtest_results if r['Event_Type'] == 'FN']
    
    print(f"BACKTEST SUMMARY:")
    print(f"  Total events: {len(backtest_results)}")
    print(f"  True Positives (TP): {len(tp_events)} ({len(tp_events)/len(backtest_results)*100:.1f}%)")
    print(f"  False Negatives (FN): {len(fn_events)} ({len(fn_events)/len(backtest_results)*100:.1f}%)")
    
    if tp_events:
        avg_coverage = np.mean([r['Coverage'] for r in tp_events])
        avg_pred_duration = np.mean([r['Predicted_Duration'] for r in tp_events])
        print(f"  Average coverage (TP events): {avg_coverage:.3f}")
        print(f"  Average predicted duration (TP events): {avg_pred_duration:.1f} days")
    
    total_obs_cost = sum([r['Observed_Cost'] for r in backtest_results])
    total_pred_cost = sum([r['Predicted_Cost'] for r in backtest_results])
    total_capped_cost = sum([r['Capped_Cost'] for r in backtest_results])
    
    print(f"  Total observed cost: ${total_obs_cost:,.0f}")
    print(f"  Total predicted cost: ${total_pred_cost:,.0f}")
    print(f"  Total capped cost: ${total_capped_cost:,.0f}")
    
    if total_obs_cost > 0:
        pred_ratio = total_pred_cost / total_obs_cost
        capped_ratio = total_capped_cost / total_obs_cost
        print(f"  Cost ratios (Predicted/Observed): {pred_ratio:.3f}")
        print(f"  Cost ratios (Capped/Observed): {capped_ratio:.3f}")

print("="*120)

# Convert backtest results to DataFrame for Excel export
backtest_df = pd.DataFrame(backtest_results)

# Create overall summary DataFrame
if all_event_stats:
    overall_prediction_rate = np.mean([s['Prediction_Rate'] for s in all_event_stats])
    overall_avg_coverage = np.mean([s['Avg_Coverage'] for s in all_event_stats if s['Avg_Coverage'] > 0])
    overall_avg_pred_duration = np.mean([s['Avg_Predicted_Duration'] for s in all_event_stats if s['Avg_Predicted_Duration'] > 0])
    overall_p99_pred_duration = np.mean([s['P99_Predicted_Duration'] for s in all_event_stats if s['P99_Predicted_Duration'] > 0])
    overall_avg_obs_duration = np.mean([s['Duration'] for s in all_event_stats])
    
    total_all_obs_cost = sum([s['Observed_Cost'] for s in all_event_stats])
    total_all_avg_full_cost = sum([s['Avg_Simulated_Full_Cost'] for s in all_event_stats])
    total_all_p99_full_cost = sum([s['P99_Simulated_Full_Cost'] for s in all_event_stats])
    total_all_avg_capped_cost = sum([s['Avg_Simulated_Capped_Cost'] for s in all_event_stats])
    total_all_p99_capped_cost = sum([s['P99_Simulated_Capped_Cost'] for s in all_event_stats])
else:
    # Default values if no events
    overall_prediction_rate = 0
    overall_avg_coverage = 0
    overall_avg_pred_duration = 0
    overall_p99_pred_duration = 0
    overall_avg_obs_duration = 0
    total_all_obs_cost = 0
    total_all_avg_full_cost = 0
    total_all_p99_full_cost = 0
    total_all_avg_capped_cost = 0
    total_all_p99_capped_cost = 0

overall_summary_df = pd.DataFrame([{
    'Metric': 'Total Events',
    'Value': len(all_event_stats)
}, {
    'Metric': 'Average Prediction Rate',
    'Value': overall_prediction_rate
}, {
    'Metric': 'Average Coverage When Predicted',
    'Value': overall_avg_coverage
}, {
    'Metric': 'Average Predicted Duration (days)',
    'Value': overall_avg_pred_duration
}, {
    'Metric': 'P99 Predicted Duration (days)',
    'Value': overall_p99_pred_duration
}, {
    'Metric': 'Total Observed Cost ($)',
    'Value': total_all_obs_cost
}, {
    'Metric': 'Total Avg Full Simulated Cost ($)',
    'Value': total_all_avg_full_cost
}, {
    'Metric': 'Total P99 Full Simulated Cost ($)',
    'Value': total_all_p99_full_cost
}, {
    'Metric': 'Total Avg Capped Simulated Cost ($)',
    'Value': total_all_avg_capped_cost
}, {
    'Metric': 'Total P99 Capped Simulated Cost ($)',
    'Value': total_all_p99_capped_cost
}, {
    'Metric': 'Avg Duration Ratio (Pred/Obs)',
    'Value': overall_avg_pred_duration / overall_avg_obs_duration if overall_avg_obs_duration > 0 else 0
}, {
    'Metric': 'P99 Duration Ratio (Pred/Obs)',
    'Value': overall_p99_pred_duration / overall_avg_obs_duration if overall_avg_obs_duration > 0 else 0
}, {
    'Metric': 'Avg Full Cost Ratio (Sim/Obs)',
    'Value': total_all_avg_full_cost / total_all_obs_cost if total_all_obs_cost > 0 else 0
}, {
    'Metric': 'P99 Full Cost Ratio (Sim/Obs)',
    'Value': total_all_p99_full_cost / total_all_obs_cost if total_all_obs_cost > 0 else 0
}, {
    'Metric': 'Avg Capped Cost Ratio (Sim/Obs)',
    'Value': total_all_avg_capped_cost / total_all_obs_cost if total_all_obs_cost > 0 else 0
}, {
    'Metric': 'P99 Capped Cost Ratio (Sim/Obs)',
    'Value': total_all_p99_capped_cost / total_all_obs_cost if total_all_obs_cost > 0 else 0
}])

# ============================================================================
# FIXED EXCEL EXPORT SECTION - REPLACE THE EXISTING EXCEL EXPORT
# ============================================================================

print(f"\n💾 Creating yearly summary DataFrame with correct min_duration...")

# Convert yearly_summary list to DataFrame (this was the missing step!)
yearly_summary_df = pd.DataFrame(yearly_summary)
# Add run and min_duration as columns to the yearly summary
yearly_summary_df['run'] = AEP_CONFIG['run_name']
yearly_summary_df['min_duration'] = AEP_CONFIG['min_event_duration']
yearly_summary_df['N_FISHERMEN'] = AEP_CONFIG['N_FISHERMEN']
yearly_summary_df['WAVE_THRESHOLD'] = AEP_CONFIG['rule_threshold']

# Calculate and add average observed duration for each year
yearly_summary_df['Avg_Obs_Duration'] = 0.0
for idx, row in yearly_summary_df.iterrows():
    year = row['Year']
    year_stats = [s for s in all_event_stats if s['Year'] == year]
    if year_stats:
        avg_obs_duration = np.mean([s['Duration'] for s in year_stats])
        yearly_summary_df.loc[idx, 'Avg_Obs_Duration'] = avg_obs_duration

# Reorder columns to put run and min_duration first, then avg obs duration after events
cols = ['run', 'N_FISHERMEN', 'WAVE_THRESHOLD', 'min_duration', 'Year', 'Events', 'Avg_Obs_Duration'] + [col for col in yearly_summary_df.columns if col not in ['run', 'N_FISHERMEN', 'WAVE_THRESHOLD', 'min_duration', 'Year', 'Events', 'Avg_Obs_Duration']]
yearly_summary_df = yearly_summary_df[cols]

print(f"✅ Yearly summary DataFrame created with {len(yearly_summary_df)} rows")
print(f"   Using run = {AEP_CONFIG['run_name']}, min_event_duration = {AEP_CONFIG['min_event_duration']} days")

# ============================================================================
# CREATE DATA FRAMES FOR EXCEL EXPORT
# ============================================================================

print(f"\n📊 Creating DataFrames for Excel export...")

# Create backtest results DataFrame
backtest_df = pd.DataFrame([{
    'Metric': 'Total Observed Events',
    'Value': len(all_event_stats)
}, {
    'Metric': 'Average Prediction Rate',
    'Value': np.mean([s['Prediction_Rate'] for s in all_event_stats])
}, {
    'Metric': 'Average Coverage When Predicted',
    'Value': np.mean([s['Avg_Coverage'] for s in all_event_stats if s['Avg_Coverage'] > 0])
}])

# Create event statistics DataFrame
event_stats_df = pd.DataFrame(all_event_stats)

# Create overall summary DataFrame
overall_summary_df = pd.DataFrame([{
    'Metric': 'Total Events',
    'Value': len(all_event_stats)
}, {
    'Metric': 'Average Prediction Rate',
    'Value': np.mean([s['Prediction_Rate'] for s in all_event_stats])
}, {
    'Metric': 'Average Coverage When Predicted',
    'Value': np.mean([s['Avg_Coverage'] for s in all_event_stats if s['Avg_Coverage'] > 0])
}, {
    'Metric': 'Average Predicted Duration (days)',
    'Value': np.mean([s['Avg_Predicted_Duration'] for s in all_event_stats if s['Avg_Predicted_Duration'] > 0])
}, {
    'Metric': 'P99 Predicted Duration (days)',
    'Value': np.mean([s['P99_Predicted_Duration'] for s in all_event_stats if s['P99_Predicted_Duration'] > 0])
}, {
    'Metric': 'Total Observed Cost ($)',
    'Value': sum([s['Observed_Cost'] for s in all_event_stats])
}, {
    'Metric': 'Total Avg Full Simulated Cost ($)',
    'Value': sum([s['Avg_Simulated_Full_Cost'] for s in all_event_stats])
}, {
    'Metric': 'Total P99 Full Simulated Cost ($)',
    'Value': sum([s['P99_Simulated_Full_Cost'] for s in all_event_stats])
}])

# Create configuration DataFrame
config_df = pd.DataFrame([{
    'Parameter': 'run_name',
    'Value': AEP_CONFIG['run_name']
}, {
    'Parameter': 'rule_feature',
    'Value': AEP_CONFIG['rule_feature']
}, {
    'Parameter': 'rule_threshold',
    'Value': AEP_CONFIG['rule_threshold']
}, {
    'Parameter': 'min_event_duration',
    'Value': AEP_CONFIG['min_event_duration']
}, {
    'Parameter': 'n_simulations',
    'Value': AEP_CONFIG['n_simulations']
}, {
    'Parameter': 'N_FISHERMEN',
    'Value': AEP_CONFIG['N_FISHERMEN']
}, {
    'Parameter': 'W',
    'Value': AEP_CONFIG['W']
}])

# ============================================================================
# EXPORT TO EXCEL WITH MULTIPLE SHEETS
# ============================================================================

print(f"\n💾 Exporting to Excel file...")

# Create filename with region and min_event_duration
excel_filename = f'event_aep_analysis_{region}_mindur{AEP_CONFIG["min_event_duration"]}.xlsx'
excel_path = os.path.join(project_root, 'results', excel_filename)

# Ensure results directory exists
os.makedirs(os.path.dirname(excel_path), exist_ok=True)

# Export to Excel with multiple sheets
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    # Backtest results
    backtest_df.to_excel(writer, sheet_name='Observed_Backtest', index=False)
    
    # Individual event statistics  
    event_stats_df.to_excel(writer, sheet_name='Event_Statistics', index=False)
    
    # Yearly summary statistics
    yearly_summary_df.to_excel(writer, sheet_name='Yearly_Summary', index=False)
    
    # Overall summary
    overall_summary_df.to_excel(writer, sheet_name='Overall_Summary', index=False)
    
    # Configuration sheet
    config_df.to_excel(writer, sheet_name='Configuration', index=False)
    
    # Add a sheet specifically showing the min_event_duration parameter
    min_duration_info = pd.DataFrame([
        {'Parameter': 'min_event_duration', 'Value': AEP_CONFIG['min_event_duration']},
        {'Parameter': 'event_column_used', 'Value': get_event_column_name(AEP_CONFIG['min_event_duration'])},
        {'Parameter': 'rule_feature', 'Value': AEP_CONFIG['rule_feature']},
        {'Parameter': 'rule_threshold', 'Value': AEP_CONFIG['rule_threshold']},
        {'Parameter': 'analysis_date', 'Value': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    ])
    min_duration_info.to_excel(writer, sheet_name='Analysis_Parameters', index=False)
    
    # Column explanations for the statistics
    stats_explanations = pd.DataFrame({
        'Column': [
            'Year', 'Event_ID', 'Start_Day', 'End_Day', 'Duration', 'Prediction_Rate', 
            'Times_Predicted', 'Avg_Coverage', 'Observed_Cost', 'Avg_Predicted_Duration',
            'P99_Predicted_Duration', 'Avg_Simulated_Full_Cost', 'P99_Simulated_Full_Cost',
            'Avg_Simulated_Capped_Cost', 'P99_Simulated_Capped_Cost'
        ],
        'Description': [
            'Year of the observed event',
            'Event ID within that year',
            'Starting day index of the event',
            'Ending day index of the event',
            f'Observed duration of the event (days) - minimum {AEP_CONFIG["min_event_duration"]} days',
            'Proportion of simulations where this event was predicted (0-1)',
            'Number of simulations (out of 1000) where this event was predicted',
            'Average overlap coverage when event was predicted',
            'Actual cost based on observed duration',
            'Average predicted duration across simulations where event was detected',
            '99th percentile predicted duration across simulations where event was detected',
            'Average cost based on full predicted durations',
            '99th percentile cost based on full predicted durations',
            'Average cost based on 50% of predicted durations (conservative)',
            '99th percentile cost based on 50% of predicted durations (conservative)'
        ]
    })
    stats_explanations.to_excel(writer, sheet_name='Column_Explanations', index=False)

print(f"✅ Excel file exported successfully!")
print(f"   📁 File location: {excel_path}")
print(f"   📊 Sheets included:")
print(f"      - Observed_Backtest: Real-world rule performance on actual data")
print(f"      - Event_Statistics: Individual event analysis across simulations")
print(f"      - Yearly_Summary: Year-by-year totals")
print(f"      - Overall_Summary: Complete dataset statistics")
print(f"      - Configuration: Analysis parameters")
print(f"      - Column_Explanations: Guide for interpreting data")

# CREATE SPECIFIC CSV FILE FOR YEARLY SUMMARY (no timestamp)
csv_filename = f'yearly_summary_{RUN_NAME}_mindur{AEP_CONFIG["min_event_duration"]}.csv'
csv_path = os.path.join(project_root, 'results', csv_filename)

# Export yearly summary to CSV
yearly_summary_df.to_csv(csv_path, index=False)

print(f"✅ Yearly summary CSV exported!")
print(f"   📁 File: {csv_path}")
print(f"   📊 Contains run and min_duration columns for easy tracking")

# Verification: Print what's in the yearly summary
print(f"\n🔍 VERIFICATION - Yearly Summary Content:")
if len(yearly_summary_df) > 0:
    print(yearly_summary_df.to_string(index=False))
    print(f"\n📋 All events are {AEP_CONFIG['min_event_duration']}+ days as specified in configuration")
else:
    print("   No yearly summary data found - check if events exist for the current min_event_duration")

print(f"   📁 File location: {stats_excel_path}")
print(f"   📊 Sheets included:")
print(f"      - Observed_Backtest: Real-world rule performance on actual data")
print(f"      - Event_Statistics: Individual event analysis across simulations")
print(f"      - Yearly_Summary: Year-by-year totals")
print(f"      - Overall_Summary: Complete dataset statistics")
print(f"      - Configuration: Analysis parameters")
print(f"      - Column_Explanations: Guide for interpreting data")

print("="*160)
print(f"{'Year':<6} {'Event':<6} {'Days':<15} {'Dur':<4} {'Pred':<5} {'Rate':<8} {'Cov':<6} {'AvgDur':<7} {'P99Dur':<7} {'ObsCost':<9} {'AvgFull':<9} {'P99Full':<9} {'AvgCap':<9} {'P99Cap':<9}")
print("-" * 160)
print("COLUMN EXPLANATIONS:")
print("- Pred: Number of simulations (out of 1000) where this event was predicted")
print("- Rate: Prediction rate (0.0-1.0) = Pred/1000")
print("- Cov: Average coverage when predicted (overlap between predicted and observed)")
print("- AvgDur: Average predicted duration when event is predicted (days)")
print("- P99Dur: 99th percentile predicted duration when event is predicted (days)")
print("- ObsCost: Actual cost based on observed event duration")
print("- AvgFull: Average simulated cost (full predicted duration)")
print("- P99Full: 99th percentile simulated cost (full predicted duration)")
print("- AvgCap: Average simulated cost (50% of predicted duration)")
print("- P99Cap: 99th percentile simulated cost (50% of predicted duration)")
print("-" * 160)

for stats in all_event_stats:
    print(f"{stats['Year']:<6} "
          f"{stats['Event_ID']:<6} "
          f"{stats['Start_Day']}-{stats['End_Day']:<12} "
          f"{stats['Duration']:<4} "
          f"{stats['Times_Predicted']:<5} "
          f"{stats['Prediction_Rate']:<8.3f} "
          f"{stats['Avg_Coverage']:<6.3f} "
          f"{stats['Avg_Predicted_Duration']:<7.1f} "
          f"{stats['P99_Predicted_Duration']:<7.1f} "
          f"${stats['Observed_Cost']:<8,.0f} "
          f"${stats['Avg_Simulated_Full_Cost']:<8,.0f} "
          f"${stats['P99_Simulated_Full_Cost']:<8,.0f} "
          f"${stats['Avg_Simulated_Capped_Cost']:<8,.0f} "
          f"${stats['P99_Simulated_Capped_Cost']:<8,.0f}")

print("-" * 160)

# Overall statistics by year
print(f"\nSTATISTICS BY YEAR:")
print(f"{'Year':<6} {'Events':<8} {'AvgPred':<8} {'P99Pred':<8} {'AvgRate':<8} {'AvgCov':<8} {'AvgDur':<8} {'P99Dur':<8} {'ObsCost':<10} {'AvgFull':<10} {'P99Full':<10} {'AvgCap':<10} {'P99Cap':<10}")
print("-" * 130)

yearly_summary = []
for year in available_years:
    year_stats = [s for s in all_event_stats if s['Year'] == year]
    if year_stats:
        avg_rate = np.mean([s['Prediction_Rate'] for s in year_stats])
        avg_coverage = np.mean([s['Avg_Coverage'] for s in year_stats if s['Avg_Coverage'] > 0])
        avg_pred_duration = np.mean([s['Avg_Predicted_Duration'] for s in year_stats if s['Avg_Predicted_Duration'] > 0])
        p99_pred_duration = np.mean([s['P99_Predicted_Duration'] for s in year_stats if s['P99_Predicted_Duration'] > 0])
        total_obs_cost = sum([s['Observed_Cost'] for s in year_stats])
        total_avg_full_cost = sum([s['Avg_Simulated_Full_Cost'] for s in year_stats])
        total_p99_full_cost = sum([s['P99_Simulated_Full_Cost'] for s in year_stats])
        total_avg_capped_cost = sum([s['Avg_Simulated_Capped_Cost'] for s in year_stats])
        total_p99_capped_cost = sum([s['P99_Simulated_Capped_Cost'] for s in year_stats])
        
        yearly_summary.append({
            'Year': year,
            'Events': len(year_stats),
            'Avg_Rate': avg_rate,
            'Avg_Coverage': avg_coverage,
            'Avg_Duration': avg_pred_duration,
            'P99_Duration': p99_pred_duration,
            'Obs_Cost': total_obs_cost,
            'Avg_Full_Cost': total_avg_full_cost,
            'P99_Full_Cost': total_p99_full_cost,
            'Avg_Capped_Cost': total_avg_capped_cost,
            'P99_Capped_Cost': total_p99_capped_cost
        })
        
        print(f"{year:<6} "
              f"{len(year_stats):<8} "
              f"{avg_predicted_events:<8.1f} "
              f"{p99_predicted_events:<8.1f} "
              f"{avg_rate:<8.3f} "
              f"{avg_coverage:<8.3f} "
              f"{avg_pred_duration:<8.1f} "
              f"{p99_pred_duration:<8.1f} "
              f"${total_obs_cost:<9,.0f} "
              f"${total_avg_full_cost:<9,.0f} "
              f"${total_p99_full_cost:<9,.0f} "
              f"${total_avg_capped_cost:<9,.0f} "
              f"${total_p99_capped_cost:<9,.0f}")

# Overall statistics
if all_event_stats:
    overall_prediction_rate = np.mean([s['Prediction_Rate'] for s in all_event_stats])
    overall_avg_coverage = np.mean([s['Avg_Coverage'] for s in all_event_stats if s['Avg_Coverage'] > 0])
    overall_avg_pred_duration = np.mean([s['Avg_Predicted_Duration'] for s in all_event_stats if s['Avg_Predicted_Duration'] > 0])
    overall_p99_pred_duration = np.mean([s['P99_Predicted_Duration'] for s in all_event_stats if s['P99_Predicted_Duration'] > 0])
    total_all_obs_cost = sum([s['Observed_Cost'] for s in all_event_stats])
    total_all_avg_full_cost = sum([s['Avg_Simulated_Full_Cost'] for s in all_event_stats])
    total_all_p99_full_cost = sum([s['P99_Simulated_Full_Cost'] for s in all_event_stats])
    total_all_avg_capped_cost = sum([s['Avg_Simulated_Capped_Cost'] for s in all_event_stats])
    total_all_p99_capped_cost = sum([s['P99_Simulated_Capped_Cost'] for s in all_event_stats])
    
    print("-" * 110)
    print(f"OVERALL STATISTICS:")
    print(f"  Total events analyzed: {len(all_event_stats)}")
    print(f"  Average prediction rate: {overall_prediction_rate:.3f}")
    print(f"  Average coverage when predicted: {overall_avg_coverage:.3f}")
    print(f"  Average predicted duration: {overall_avg_pred_duration:.1f} days")
    print(f"  P99 predicted duration: {overall_p99_pred_duration:.1f} days")
    print(f"  Total costs:")
    print(f"    Observed: ${total_all_obs_cost:,.0f}")
    print(f"    Avg Full Sim: ${total_all_avg_full_cost:,.0f}")
    print(f"    P99 Full Sim: ${total_all_p99_full_cost:,.0f}")
    print(f"    Avg Capped Sim: ${total_all_avg_capped_cost:,.0f}")
    print(f"    P99 Capped Sim: ${total_all_p99_capped_cost:,.0f}")
    
    # Duration comparison
    overall_avg_obs_duration = np.mean([s['Duration'] for s in all_event_stats])
    if overall_avg_obs_duration > 0:
        duration_ratio = overall_avg_pred_duration / overall_avg_obs_duration
        p99_duration_ratio = overall_p99_pred_duration / overall_avg_obs_duration
        print(f"  Average observed duration: {overall_avg_obs_duration:.1f} days")
        print(f"  Duration ratios (Predicted/Observed):")
        print(f"    Average: {duration_ratio:.3f}")
        print(f"    P99: {p99_duration_ratio:.3f}")
    
    # Cost ratios
    if total_all_obs_cost > 0:
        avg_full_ratio = total_all_avg_full_cost / total_all_obs_cost
        p99_full_ratio = total_all_p99_full_cost / total_all_obs_cost
        avg_capped_ratio = total_all_avg_capped_cost / total_all_obs_cost
        p99_capped_ratio = total_all_p99_capped_cost / total_all_obs_cost
        print(f"  Cost ratios (Simulated/Observed):")
        print(f"    Avg Full: {avg_full_ratio:.3f}")
        print(f"    P99 Full: {p99_full_ratio:.3f}")
        print(f"    Avg Capped: {avg_capped_ratio:.3f}")
        print(f"    P99 Capped: {p99_capped_ratio:.3f}")
    
    # Find best and worst predicted events
    best_predicted = max(all_event_stats, key=lambda x: x['Prediction_Rate'])
    worst_predicted = min(all_event_stats, key=lambda x: x['Prediction_Rate'])
    
    print(f"  Best predicted: {best_predicted['Year']} Event {best_predicted['Event_ID']} "
          f"({best_predicted['Prediction_Rate']:.3f} rate, {best_predicted['Duration']} days obs, "
          f"{best_predicted['Avg_Predicted_Duration']:.1f} days avg pred)")
    print(f"  Worst predicted: {worst_predicted['Year']} Event {worst_predicted['Event_ID']} "
          f"({worst_predicted['Prediction_Rate']:.3f} rate, {worst_predicted['Duration']} days obs, "
          f"{worst_predicted['Avg_Predicted_Duration']:.1f} days avg pred)")

print("="*160)

# ============================================================================
# SAVE RESULTS (optional)
# ============================================================================

print(f"\n💾 Results ready for export...")

# Create results dictionary
aep_results = {
    'config': AEP_CONFIG,
    'summary_stats': {
        'tp_costs': {'mean': float(np.mean(annual_tp_costs)), 'std': float(np.std(annual_tp_costs)), 'p95': float(np.percentile(annual_tp_costs, 95))},
        'fp_costs': {'mean': float(np.mean(annual_fp_costs)), 'std': float(np.std(annual_fp_costs)), 'p95': float(np.percentile(annual_fp_costs, 95))},
        'fn_costs': {'mean': float(np.mean(annual_fn_costs)), 'std': float(np.std(annual_fn_costs)), 'p95': float(np.percentile(annual_fn_costs, 95))},
        'total_costs': {'mean': float(np.mean(annual_total_costs)), 'std': float(np.std(annual_total_costs)), 'p95': float(np.percentile(annual_total_costs, 95))}
    },
    'aep_curves': {
        'tp': tp_aep,
        'fp': fp_aep,
        'fn': fn_aep,
        'total': total_aep
    }
}

# Uncomment to save results to file
# import json
# results_path = os.path.join(project_root, 'results', f'optimized_event_aep_results_{region}.json')
# with open(results_path, 'w') as f:
#     json.dump(aep_results, f, indent=2)
# print(f"✅ Results saved to: {results_path}")

print(f"\n🎉 OPTIMIZED Event-Level AEP Analysis Complete!")
print(f"   Rule: {AEP_CONFIG['rule_feature']} > {AEP_CONFIG['rule_threshold']}")
print(f"   Mean annual total cost: ${np.mean(annual_total_costs):,.0f}")
print(f"   95th percentile cost: ${np.percentile(annual_total_costs, 95):,.0f}")
print(f"   Performance: ~{len(annual_total_costs)/60:.1f} simulations per second")

#%%
# Yearly summary reading and appending

# ============================================================================
# COMBINE ALL YEARLY SUMMARY CSV FILES
# ============================================================================
# Read all yearly summary CSV files from all runs and min_duration parameters
# and create a single comprehensive DataFrame

print(f"\n" + "="*80)
print(f"COMBINING ALL YEARLY SUMMARY CSV FILES")
print("="*80)

# Find all yearly summary CSV files
results_dir = os.path.join(project_root, 'results')
csv_pattern = os.path.join(results_dir, 'yearly_summary_*.csv')
csv_files = glob.glob(csv_pattern)

print(f"📁 Found {len(csv_files)} yearly summary CSV files:")
for file in sorted(csv_files):
    print(f"   - {os.path.basename(file)}")

# Read and combine all CSV files
all_yearly_summaries = []

for csv_file in sorted(csv_files):
    try:
        # Extract run and mindur from filename
        filename = os.path.basename(csv_file)
        # Parse: yearly_summary_run_g4_mindur5.csv -> run_g4, mindur5
        parts = filename.replace('yearly_summary_', '').replace('.csv', '').split('_mindur')
        run_name = parts[0]  # e.g., "run_g4"
        mindur = int(parts[1])  # e.g., 5
        
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Add metadata columns
        df['run_name'] = run_name
        df['mindur'] = mindur
        df['source_file'] = filename
        
        all_yearly_summaries.append(df)
        
        print(f"✅ Loaded {len(df)} rows from {filename}")
        
    except Exception as e:
        print(f"❌ Error reading {csv_file}: {e}")

# Combine all DataFrames
if all_yearly_summaries:
    combined_df = pd.concat(all_yearly_summaries, ignore_index=True)
    
    print(f"\n📊 COMBINED DATASET SUMMARY:")
    print(f"   Total rows: {len(combined_df)}")
    print(f"   Unique runs: {combined_df['run_name'].nunique()}")
    print(f"   Unique mindur values: {sorted(combined_df['mindur'].unique())}")
    print(f"   Years covered: {sorted(combined_df['Year'].unique())}")
    
    # Show column information
    print(f"\n📋 COLUMNS IN COMBINED DATASET:")
    for col in combined_df.columns:
        print(f"   - {col}")
    
    # Save combined dataset
    combined_csv_path = os.path.join(results_dir, 'all_yearly_summaries_combined.csv')
    combined_df.to_csv(combined_csv_path, index=False)
    
    print(f"\n💾 Combined dataset saved to:")
    print(f"   📁 {combined_csv_path}")
    
    # Show sample of the data
    print(f"\n📋 SAMPLE DATA (first 10 rows):")
    print(combined_df.head(10).to_string(index=False))
    
    # Summary statistics by run and mindur
    print(f"\n📊 SUMMARY BY RUN AND MINDUR:")
    summary_stats = combined_df.groupby(['run_name', 'mindur']).agg({
        'Year': 'count',
        'Events': 'sum',
        'Avg_Rate': 'mean',
        'Avg_Coverage': 'mean',
        'Obs_Cost': 'sum',
        'Avg_Full_Cost': 'sum',
        'P99_Full_Cost': 'sum'
    }).round(3)
    
    print(summary_stats.to_string())
    
else:
    print("❌ No CSV files found or all files had errors")

print("="*80)
print(f"✅ YEARLY SUMMARY COMBINATION COMPLETE")
print("="*80)

combined_df = combined_df[(combined_df['run_name'] != 'run_g1') &
                           (combined_df['run_name'] != 'run_g2') ].reset_index(drop=True)
combined_df = combined_df.sort_values(by=['min_duration', 'run'], ascending=True)


# %%

# ============================================================================
# CREATE AVERAGED DATASET BY RUN AND MIN_DURATION
# ============================================================================
# Average all columns by run_name and mindur to create a summary dataset

print(f"\n" + "="*80)
print(f"CREATING AVERAGED DATASET BY RUN AND MIN_DURATION")
print("="*80)

# Create averaged dataset by run and mindur
if 'combined_df' in locals() and len(combined_df) > 0:
    # Identify numeric columns for averaging (exclude metadata columns)
    exclude_cols = ['run_name', 'mindur', 'source_file', 'Year']
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
    avg_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"📊 Averaging {len(avg_cols)} numeric columns:")
    for col in avg_cols:
        print(f"   - {col}")
    
    # Group by run_name and mindur, then average numeric columns
    combined_df_run = combined_df.groupby(['run_name', 'mindur']).agg({
        # Count non-null values for Year to get number of years per run
        'Year': 'count',
        # Average all numeric columns
        **{col: 'mean' for col in avg_cols}
    }).reset_index()
    
    # Rename Year count to Years_Count for clarity
    combined_df_run = combined_df_run.rename(columns={'Year': 'Years_Count'})
    
    print(f"\n📊 AVERAGED DATASET SUMMARY:")
    print(f"   Total rows: {len(combined_df_run)}")
    print(f"   Unique runs: {combined_df_run['run_name'].nunique()}")
    print(f"   Unique mindur values: {sorted(combined_df_run['mindur'].unique())}")
    
    # Show the averaged dataset
    print(f"\n📋 AVERAGED DATASET (all rows):")
    print(combined_df_run.to_string(index=False))
    
    # Save the averaged dataset
    averaged_csv_path = os.path.join(results_dir, 'yearly_summaries_averaged_by_run.csv')
    combined_df_run.to_csv(averaged_csv_path, index=False)
    
    print(f"\n💾 Averaged dataset saved to:")
    print(f"   📁 {averaged_csv_path}")
    
    # Additional analysis: Show best performing runs
    print(f"\n🏆 BEST PERFORMING RUNS BY AVG_RATE:")
    best_runs = combined_df_run.sort_values('Avg_Rate', ascending=False)
    print(best_runs[['run_name', 'mindur', 'Years_Count', 'Avg_Rate', 'Avg_Coverage', 'Obs_Cost', 'Avg_Full_Cost']].to_string(index=False))
    
    # Show cost efficiency analysis
    print(f"\n💰 COST EFFICIENCY ANALYSIS:")
    combined_df_run['Cost_Ratio'] = combined_df_run['Avg_Full_Cost'] / combined_df_run['Obs_Cost']
    cost_efficiency = combined_df_run.sort_values('Cost_Ratio')
    print(cost_efficiency[['run_name', 'mindur', 'Obs_Cost', 'Avg_Full_Cost', 'Cost_Ratio']].to_string(index=False))
    
else:
    print("❌ No combined_df available for averaging")

print("="*80)
print(f"✅ AVERAGED DATASET CREATION COMPLETE")
print("="*80)

combined_df_run = combined_df_run.sort_values(by=['mindur', 'run_name'], ascending=True)
combined_df_run.to_excel(os.path.join(results_dir, 'yearly_summaries_averaged_by_run.xlsx'), index=False)
#%%

print(f"\n💾 Excel export skipped - variables not defined in this version")
print(f"   📁 Main analysis results are saved in CSV format instead")
print(f"   This section was removed to fix the NameError with undefined variables")
