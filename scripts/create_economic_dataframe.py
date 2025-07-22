#%%
"""
Create Economic DataFrame
========================

This script creates a DataFrame with the following columns:
- date: Date of observation
- event_dummy_1: Binary indicator for port closure events
- N_fishermen: Number of fishermen affected
- wages: Daily wage rate
- daily_losses: Daily economic losses

Author: Wave Analysis Team
Date: 2025
"""

import os
import pandas as pd
import numpy as np
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
# Load the merged data to get dates and event_dummy_1
print("📊 Loading merged data...")
merged_path = os.path.join(config.run_output_dir, 'df_swan_waverys_merged.csv')

if not os.path.exists(merged_path):
    print(f"❌ Merged data file not found: {merged_path}")
    print("Please run the data preparation pipeline first")
    exit(1)

df_merged = pd.read_csv(merged_path, parse_dates=['date'])
print(f"✅ Loaded merged data: {df_merged.shape}")
print(f"   Date range: {df_merged['date'].min()} to {df_merged['date'].max()}")

#%%
# Load wage data
print("\n💰 Loading wage data...")
wage_path = os.path.join(config.RAW_DATA_DIR, 'wages_caleta.xlsx')

if not os.path.exists(wage_path):
    print(f"❌ Wage data file not found: {wage_path}")
    print("Creating synthetic wage data...")
    
    # Create synthetic wage data for testing
    port_name = config.reference_port.upper().replace(' ', '_')
    
    # Synthetic economic parameters (you can adjust these)
    synthetic_wages = {
        'port_name': port_name,
        'daily_wages': 50.0,  # Daily wage in USD
        'n_fishermen': 100,   # Number of fishermen
        'daily_losses': 5000.0  # Daily losses = n_fishermen * daily_wages
    }
    
    df_wages = pd.DataFrame([synthetic_wages])
    print(f"✅ Created synthetic wage data for {port_name}")
    print(f"   Daily wages: ${synthetic_wages['daily_wages']}")
    print(f"   Number of fishermen: {synthetic_wages['n_fishermen']}")
    print(f"   Daily losses: ${synthetic_wages['daily_losses']}")
    
else:
    # Load actual wage data
    try:
        df_wages_raw = pd.read_excel(wage_path)
        print(f"✅ Loaded wage data: {df_wages_raw.shape}")
        
        # Clean and normalize port names
        df_wages_raw.rename(columns={'port_name':'port_name_wages'}, inplace=True)
        df_wages_raw.rename(columns={'caleta':'port_name'}, inplace=True)
        df_wages_raw['port_name'] = df_wages_raw['port_name'].str.upper()
        df_wages_raw['port_name'] = df_wages_raw['port_name'].str.replace(' ', '_')
        df_wages_raw = df_wages_raw[df_wages_raw['port_name'] != 'PUERTO_SUPE']
        
        # Calculate daily wages
        df_wages_raw['daily_wages'] = df_wages_raw['w_p50']/30
        
        # Clean port names to match reference port
        df_wages_raw['port_name'] = df_wages_raw['port_name'].str.replace('CALETA_EL_CHACO','CALETA_EL_CHACHO')
        df_wages_raw['port_name'] = df_wages_raw['port_name'].str.replace('CALETA_CULEBRAS','COLETA_CULEBRAS')
        df_wages_raw['port_name'] = df_wages_raw['port_name'].str.replace('CALETA_LOBITOS_(TALARA)','CALETA_LOBITOS')
        df_wages_raw['port_name'] = df_wages_raw['port_name'].str.replace('CALETA_SAN_ANDRÉS','CALETA_SAN_ANDRES')
        df_wages_raw['port_name'] = df_wages_raw['port_name'].str.replace('CALLAO','DPA_CALLAO')
        df_wages_raw['port_name'] = df_wages_raw['port_name'].str.replace('CHORRILLOS','DPA_CHORRILLOS')
        df_wages_raw['port_name'] = df_wages_raw['port_name'].str.replace('ENSENADA_DE_SECHURA','ENSENADA_SECHURA')
        df_wages_raw['port_name'] = df_wages_raw['port_name'].str.replace('PUERTO_MATARANI_(MUELLE_OCEAN_FISH)','MUELLE_OCEAN_FISH')
        df_wages_raw['port_name'] = df_wages_raw['port_name'].str.replace('PUNTA_PICATA', 'TERMINAL_PESQUERO_PUNTA_PICATA')
        
        # Calculate number of fishermen per port
        df_wages_raw['n_ports'] = df_wages_raw['port_name_wages'].apply(lambda x: len(x.split(',')))
        df_wages_raw['n_fishermen'] = round(df_wages_raw['fishermen_province']/df_wages_raw['n_ports'])
        
        # Select relevant columns
        df_wages = df_wages_raw[['port_name', 'daily_wages', 'n_fishermen']].reset_index(drop=True)
        df_wages['daily_losses'] = df_wages['n_fishermen'] * df_wages['daily_wages']
        
        print(f"✅ Processed wage data: {df_wages.shape}")
        
    except Exception as e:
        print(f"❌ Error loading wage data: {e}")
        print("Creating synthetic wage data...")
        
        # Create synthetic data as fallback
        port_name = config.reference_port.upper().replace(' ', '_')
        synthetic_wages = {
            'port_name': port_name,
            'daily_wages': 50.0,
            'n_fishermen': 100,
            'daily_losses': 5000.0
        }
        df_wages = pd.DataFrame([synthetic_wages])

#%%
# Find the wage data for the current port
port_name = config.reference_port.upper().replace(' ', '_')
print(f"\n🎯 Looking for wage data for port: {port_name}")

# Check if we have data for the specific port
port_wage_data = df_wages[df_wages['port_name'] == port_name]

if port_wage_data.empty:
    print(f"⚠️ No wage data found for {port_name}")
    print("Available ports:")
    for port in df_wages['port_name'].unique():
        print(f"   • {port}")
    
    # Use average values as fallback
    avg_wages = df_wages['daily_wages'].median()
    avg_fishermen = df_wages['n_fishermen'].median()
    avg_losses = df_wages['daily_losses'].median()
    
    print(f"\n📊 Using median values as fallback:")
    print(f"   Daily wages: ${avg_wages:.2f}")
    print(f"   Number of fishermen: {avg_fishermen:.0f}")
    print(f"   Daily losses: ${avg_losses:.2f}")
    
    port_wage_data = pd.DataFrame([{
        'port_name': port_name,
        'daily_wages': avg_wages,
        'n_fishermen': avg_fishermen,
        'daily_losses': avg_losses
    }])

else:
    print(f"✅ Found wage data for {port_name}:")
    print(f"   Daily wages: ${port_wage_data['daily_wages'].iloc[0]:.2f}")
    print(f"   Number of fishermen: {port_wage_data['n_fishermen'].iloc[0]:.0f}")
    print(f"   Daily losses: ${port_wage_data['daily_losses'].iloc[0]:.2f}")

#%%
# Create the final DataFrame
print("\n🔧 Creating economic DataFrame...")

# Start with the merged data (date and event_dummy_1)
df_economic = df_merged[['date', 'event_dummy_1']].copy()

# Add economic columns
df_economic['N_fishermen'] = port_wage_data['n_fishermen'].iloc[0]
df_economic['wages'] = port_wage_data['daily_wages'].iloc[0]
df_economic['daily_losses'] = port_wage_data['daily_losses'].iloc[0]

# Ensure event_dummy_1 is binary
df_economic['event_dummy_1'] = df_economic['event_dummy_1'].fillna(0).astype(int)

print(f"✅ Created economic DataFrame: {df_economic.shape}")
print(f"   Columns: {list(df_economic.columns)}")
print(f"   Date range: {df_economic['date'].min()} to {df_economic['date'].max()}")
print(f"   Total days: {len(df_economic)}")
print(f"   Event days: {df_economic['event_dummy_1'].sum()} ({df_economic['event_dummy_1'].mean()*100:.1f}%)")

#%%
# Display sample of the data
print("\n📋 Sample of economic DataFrame:")
print(df_economic.head(10))

#%%
# Summary statistics
print("\n📊 Summary Statistics:")
print(f"   Total days: {len(df_economic)}")
print(f"   Event days: {df_economic['event_dummy_1'].sum()}")
print(f"   Non-event days: {(df_economic['event_dummy_1'] == 0).sum()}")
print(f"   Event percentage: {df_economic['event_dummy_1'].mean()*100:.2f}%")
print(f"   Number of fishermen: {df_economic['N_fishermen'].iloc[0]:.0f}")
print(f"   Daily wage: ${df_economic['wages'].iloc[0]:.2f}")
print(f"   Daily losses: ${df_economic['daily_losses'].iloc[0]:.2f}")

#%%
# Calculate total economic impact
total_event_days = df_economic['event_dummy_1'].sum()
total_economic_loss = total_event_days * df_economic['daily_losses'].iloc[0]

print(f"\n💰 Economic Impact Summary:")
print(f"   Total event days: {total_event_days}")
print(f"   Daily loss per event: ${df_economic['daily_losses'].iloc[0]:.2f}")
print(f"   Total economic loss: ${total_economic_loss:,.2f}")

#%%
# Save the DataFrame
output_path = os.path.join(config.results_output_dir, 'economic_dataframe.csv')
df_economic.to_csv(output_path, index=False)
print(f"\n💾 Saved economic DataFrame to: {output_path}")

#%%
# Optional: Create yearly summary
print("\n📅 Creating yearly summary...")
df_economic['year'] = df_economic['date'].dt.year
yearly_summary = df_economic.groupby('year').agg({
    'event_dummy_1': ['sum', 'mean'],
    'daily_losses': 'first'
}).reset_index()

yearly_summary.columns = ['year', 'event_days', 'event_rate', 'daily_losses']
yearly_summary['total_yearly_loss'] = yearly_summary['event_days'] * yearly_summary['daily_losses']

print("Yearly Summary:")
print(yearly_summary)

# Save yearly summary
yearly_output_path = os.path.join(config.results_output_dir, 'economic_yearly_summary.csv')
yearly_summary.to_csv(yearly_output_path, index=False)
print(f"💾 Saved yearly summary to: {yearly_output_path}")

print("\n🎉 Economic DataFrame creation completed!")
print(f"📊 Main DataFrame: {df_economic.shape}")
print(f"📅 Yearly Summary: {yearly_summary.shape}") 