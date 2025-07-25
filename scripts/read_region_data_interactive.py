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
region = 'run_g8'  # Options: run_g1, run_g2, run_g3, run_g4, run_g5, run_g6, run_g7, run_g8, run_g9, run_g10

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

print(f"Loaded data for {region}")
print(f"Data path: {data_path}")
print(f"Shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Columns: {list(df.columns)}")

# Cell 4: Basic info
#%%
# Data info
print("Data info:")
print(df.info())

# Cell 6: Check for specific columns
#%%
# Check for wave height and event columns
wave_cols = [col for col in df.columns if 'swh' in col.lower()]
event_cols = [col for col in df.columns if 'event' in col.lower()]

print(f"Wave height columns: {wave_cols}")
print(f"Event columns: {event_cols}")

# Cell 7: Quick plot (if wave data available)
#%%
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
#%%
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

# %%
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
# %%
# go to notebook for rule evaluation
#%%

# Load rule evaluation results
#%%

# Load rule evaluation results
results_dir = os.path.join(project_root, 'results', 'cv_results', region)
results_path = os.path.join(results_dir, 'simplified_rule_cv_results.csv')

if os.path.exists(results_path):
    print(f"\n" + "="*80)
    print(f"RULE EVALUATION RESULTS - {region}")
    print("="*80)
    
    # Load and display results
    results_df = pd.read_csv(results_path)
    print(f"✅ Loaded rule evaluation results: {results_df.shape}")
    print(f"📁 Results file: {results_path}")
    
    # Display the results
    print(f"\n📊 RULE EVALUATION RESULTS:")
    print(results_df.to_string(index=False))
    
else:
    print(f"\n❌ Rule evaluation results not found: {results_path}")
    print(f"   Run the GP_rule_evaluation.py script first to generate results.")

# %%
# Rule threshold extraction
#%%

# Function to extract thresholds for a specific rule
def extract_rule_thresholds(rule_name):
    """Extract threshold values for a specific rule"""
    # Find the rule in the results
    rule_match = results_df[results_df['rule_name'] == rule_name]
    
    if len(rule_match) == 0:
        print(f"❌ Rule '{rule_name}' not found in results")
        return None
    
    rule = rule_match.iloc[0]
    print(f"\n🎯 THRESHOLDS FOR RULE: {rule_name}")
    print(f"   F1 Score: {rule['f1_mean']:.3f}")
    print(f"   Precision: {rule['precision_mean']:.3f}")
    print(f"   Recall: {rule['recall_mean']:.3f}")
    print(f"   Rule Type: {rule['rule_type']}")
    
    # Extract threshold columns
    threshold_cols = [col for col in rule.index if col.startswith('thresholds_')]
    
    if threshold_cols:
        print(f"\n📋 THRESHOLD VALUES:")
        for col in sorted(threshold_cols):
            if rule[col]:  # If threshold value exists
                if col == 'thresholds_mean':
                    print(f"   Mean across folds: {rule[col]}")
                else:
                    fold_num = col.split('_')[-1]
                    print(f"   Fold {fold_num}: {rule[col]}")
    else:
        print(f"   No threshold information available")
    
    return rule

# Show available rules for reference
print(f"Available rules (first 10):")
for i, rule_name in enumerate(results_df['rule_name'].head(10)):
    print(f"  {i+1}. {rule_name}")

# Example: Extract thresholds for the best rule
best_rule = results_df.iloc[0]['rule_name']
print(f"\n📊 Example: Extracting thresholds for the best rule:")
extract_rule_thresholds(best_rule)

# To extract thresholds for other rules, call:
extract_rule_thresholds("swh_max_swan > threshold")

# %%
# Plot swh_max_swan with event_dummy_5 shading and threshold line
#%%

# Set parameters for analysis
WAVE_THRESHOLD  = 1.8  # Wave height threshold
ROLLING_WINDOW = 7    # Rolling window size
ROLLING_THRESHOLD = 4 # Rolling sum threshold for event prediction
COVERAGE_THRESHOLD = 0.4  # Minimum coverage to count as correctly predicted event
MIN_EVENT_DURATION = 5   # Only evaluate events that are 5+ days

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
#%%
# ============================================================================
# STEP 1: CREATE PREDICTIONS AND BASIC CLASSIFICATIONS
# ============================================================================

print(f"\nStep 1: Computing predictions and basic classifications...")

# Create month column
df_2018['month'] = df_2018.index.month

# Create prediction column based on threshold with seasonal condition
# df_2018['prediction'] = (df_2018['swh_max_swan'] > WAVE_THRESHOLD).astype(int)
df_2018['prediction'] = ((df_2018['swh_max_swan'] > WAVE_THRESHOLD) &
                         (df_2018['month'] >= 4) & (df_2018['month'] <= 10)).astype(int)


#%%
# Compute TP, FP, TN, FN (day-level)
df_2018['TP'] = ((df_2018['prediction'] == 1) & (df_2018['event_dummy_5'] == 1)).astype(int)
df_2018['FP'] = ((df_2018['prediction'] == 1) & (df_2018['event_dummy_5'] == 0)).astype(int)
df_2018['TN'] = ((df_2018['prediction'] == 0) & (df_2018['event_dummy_5'] == 0)).astype(int)
df_2018['FN'] = ((df_2018['prediction'] == 0) & (df_2018['event_dummy_5'] == 1)).astype(int)

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
all_observed_events = find_continuous_periods(df_2018['event_dummy_5'])

# Filter for 5+ day events only
observed_events = [(start, end) for start, end in all_observed_events 
                   if (end - start).days + 1 >= MIN_EVENT_DURATION]

# Find all predicted event periods
all_predicted_events = find_continuous_periods(df_2018['prediction_event'])

# Filter for 5+ day events only
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

# Assign observed event durations and IDs (5+ days only)
for event_id, (start, end) in enumerate(observed_events, 1):
    duration = (end - start).days + 1
    df_2018.loc[start:end, 'event_duration'] = duration
    df_2018.loc[start:end, 'event_id'] = event_id

# Assign predicted event durations and IDs (5+ days only)
for pred_id, (start, end) in enumerate(predicted_events, 1):
    duration = (end - start).days + 1
    df_2018.loc[start:end, 'predicted_event_duration'] = duration
    df_2018.loc[start:end, 'predicted_event_id'] = pred_id

print(f"✅ Assigned IDs to {len(observed_events)} observed events (5+ days)")
print(f"✅ Assigned IDs to {len(predicted_events)} predicted events (5+ days)")

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
    
    # Add vertical black lines at months 4 and 10
    for m in [4, 10]:
        vline_date = pd.Timestamp(f'{year}-{m:02d}-01')
        ax.axvline(vline_date, color='black', linestyle='-', linewidth=1.2)
    ax.set_title(f'{year} - Event Analysis (5+ Day Events Only)', 
                 fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    if i == 0:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

plt.tight_layout()
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

print(f"\n🎯 EVENT-LEVEL PERFORMANCE (5+ DAY EVENTS ONLY)")
print("-" * 60)
print(f"Total Events (All Durations):      {len(all_observed_events)}")
print(f"Events 5+ Days:                    {total_observed_events}")
print(f"Predicted Events (5+ Days):        {total_predicted_events}")
print(f"Correctly Detected Events:         {true_positives}")
print(f"Missed Events:                     {total_observed_events - true_positives}")
print(f"Valid Predictions:                 {true_positives}")
print(f"False Alarms:                      {total_predicted_events - true_positives}")

print(f"\n📈 Event-Level Performance Metrics (5+ Day Events):")
print(f"  Event Recall:    {event_recall:.3f} ({true_positives}/{total_observed_events})")
print(f"  Event Precision: {event_precision:.3f} ({true_positives}/{total_predicted_events})")
print(f"  Event F1 Score:  {event_f1:.3f}")

# ============================================================================
# CONFUSION MATRIX DISPLAY (CORRECTED)
# ============================================================================

print(f"\n📊 CONFUSION MATRIX (5+ Day Events)")
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

print(f"\n🔍 DETAILED EVENT ANALYSIS (5+ DAY EVENTS)")
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

print(f"\n💡 INTERPRETATION (5+ Day Events Only)")
print("-" * 50)
if event_f1 > 0.7:
    print("🟢 EXCELLENT: Event-level F1 > 0.7 indicates strong predictive capability")
elif event_f1 > 0.5:
    print("🟡 GOOD: Event-level F1 > 0.5 indicates reasonable predictive capability")
elif event_f1 > 0.3:
    print("🟠 MODERATE: Event-level F1 > 0.3 indicates some predictive capability")
else:
    print("🔴 POOR: Event-level F1 < 0.3 indicates limited predictive capability")

print(f"\n🎯 SUMMARY FOR 5+ DAY EVENTS:")
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
df_2018_export = df_2018[[
    'month',
    'swh_max_swan', 
    'event_dummy_5', 
    'event_count_5',
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
#%%



