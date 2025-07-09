"""
Plot Pipeline Summary
====================

This script generates summary plots for the pipeline run. Plots are saved to the results directory for the current run (e.g., results/cv_results/run_g3/).

First plot: Comparison of swh_max_swan and swh_max_waverys with event background shading.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from config import config

# --- CONFIGURE RUN ---
run_dir = config.results_output_dir  # e.g., results/cv_results/run_g3/
os.makedirs(run_dir, exist_ok=True)

# --- LOAD DATA ---
# Assume the merged daily dataset is used (same as rule evaluation pipeline)
data_path = os.path.join(config.run_output_dir, 'df_swan_waverys_merged.csv')
df = pd.read_csv(data_path, parse_dates=['date'])

# --- PLOT 1: swh_max_swan vs swh_max_waverys with event shading and horizontal grid lines ---
import seaborn as sns

# --- PLOT 1: swh_max_swan vs swh_max_waverys with event shading and horizontal grid lines ---
fig, ax = plt.subplots(figsize=(15, 5))
black = '#222222'
forest_green = '#228B22'
light_forest_green = '#5cb85c'

ax.plot(df['date'], df['swh_max_swan'], label='swh_max_swan', color=black, linewidth=1.6)
ax.plot(df['date'], df['swh_max_waverys'], label='swh_max_waverys', color=forest_green, linewidth=1.6)

# Shade event periods
event_mask = df['event_dummy_1'] == 1
for i in range(len(df)):
    if event_mask.iloc[i]:
        ax.axvspan(df['date'].iloc[i], df['date'].iloc[i], color='red', alpha=0.18)

ax.set_ylabel('Significant Wave Height (m)')
ax.set_xlabel('Date')
ax.set_title('Comparison of swh_max_swan and swh_max_waverys with Observed Events')
ax.legend()
ax.yaxis.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
plt.tight_layout()

plot_path = os.path.join(run_dir, 'swh_max_swan_vs_waverys_events.png')
plt.savefig(plot_path)
print(f"✅ Saved plot: {plot_path}")
plt.close()

# --- PLOT 2: Density comparison of swh_max_swan and swh_max_waverys (seaborn, filled, no grid) ---
plt.figure(figsize=(8, 5))
sns.kdeplot(df['swh_max_swan'].dropna(), label='swh_max_swan', color=black, fill=True, alpha=0.55, linewidth=2)
sns.kdeplot(df['swh_max_waverys'].dropna(), label='swh_max_waverys', color=forest_green, fill=True, alpha=0.55, linewidth=2)
plt.xlabel('Significant Wave Height (m)')
plt.ylabel('Density')
plt.title('Density Comparison: swh_max_swan vs swh_max_waverys')
plt.legend()
plt.tight_layout()
density_plot_path = os.path.join(run_dir, 'densities_swan_vs_waverys.png')
plt.savefig(density_plot_path)
print(f"✅ Saved plot: {density_plot_path}")
plt.close()

# --- PLOT 3: anom_swh_max_swan vs anom_swh_max_waverys with event shading ---
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(df['date'], df['anom_swh_max_swan'], label='anom_swh_max_swan', color=black, linewidth=1.6)
ax.plot(df['date'], df['anom_swh_max_waverys'], label='anom_swh_max_waverys', color=forest_green, linewidth=1.6)

# Shade event periods
event_mask = df['event_dummy_1'] == 1
for i in range(len(df)):
    if event_mask.iloc[i]:
        ax.axvspan(df['date'].iloc[i], df['date'].iloc[i], color='red', alpha=0.18)

ax.set_ylabel('Anomaly of Max Significant Wave Height (m)')
ax.set_xlabel('Date')
ax.set_title('Comparison of anom_swh_max_swan and anom_swh_max_waverys with Observed Events')
ax.legend()
ax.yaxis.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
plt.tight_layout()

anom_plot_path = os.path.join(run_dir, 'anom_swh_max_swan_vs_waverys_events.png')
plt.savefig(anom_plot_path)
print(f"✅ Saved plot: {anom_plot_path}")
plt.close()

# --- PLOT 4: Density comparison of anom_swh_max_swan and anom_swh_max_waverys (seaborn, filled, no grid) ---
plt.figure(figsize=(8, 5))
sns.kdeplot(df['anom_swh_max_swan'].dropna(), label='anom_swh_max_swan', color=black, fill=True, alpha=0.55, linewidth=2)
sns.kdeplot(df['anom_swh_max_waverys'].dropna(), label='anom_swh_max_waverys', color=forest_green, fill=True, alpha=0.55, linewidth=2)
plt.xlabel('Anomaly of Max Significant Wave Height (m)')
plt.ylabel('Density')
plt.title('Density Comparison: anom_swh_max_swan vs anom_swh_max_waverys')
plt.legend()
plt.tight_layout()
anom_density_plot_path = os.path.join(run_dir, 'densities_anom_swh_max_swan_vs_waverys.png')
plt.savefig(anom_density_plot_path)
print(f"✅ Saved plot: {anom_density_plot_path}")
plt.close()

# --- PLOT 5: Scatter plot of swh_max_swan vs swh_max_waverys with correlation annotation ---
plt.figure(figsize=(7, 7))
sns.scatterplot(
    x=df['swh_max_swan'],
    y=df['swh_max_waverys'],
    color=black,
    edgecolor=forest_green,
    alpha=0.6,
    s=40
)
plt.xlabel('swh_max_swan')
plt.ylabel('swh_max_waverys')
plt.title('Scatter: swh_max_swan vs swh_max_waverys')
corr = df[['swh_max_swan', 'swh_max_waverys']].corr().iloc[0,1]
plt.annotate(f'Pearson r = {corr:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=13, ha='left', va='top', color=forest_green, fontweight='bold')
plt.tight_layout()
scatter_path = os.path.join(run_dir, 'scatter_swh_max_swan_vs_waverys.png')
plt.savefig(scatter_path)
print(f"✅ Saved plot: {scatter_path}")
plt.close()

# --- PLOT 6: Scatter plot of anom_swh_max_swan vs anom_swh_max_waverys with correlation annotation ---
plt.figure(figsize=(7, 7))
sns.scatterplot(
    x=df['anom_swh_max_swan'],
    y=df['anom_swh_max_waverys'],
    color=black,
    edgecolor=forest_green,
    alpha=0.6,
    s=40
)
plt.xlabel('anom_swh_max_swan')
plt.ylabel('anom_swh_max_waverys')
plt.title('Scatter: anom_swh_max_swan vs anom_swh_max_waverys')
corr_anom = df[['anom_swh_max_swan', 'anom_swh_max_waverys']].corr().iloc[0,1]
plt.annotate(f'Pearson r = {corr_anom:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=13, ha='left', va='top', color=forest_green, fontweight='bold')
plt.tight_layout()
scatter_anom_path = os.path.join(run_dir, 'scatter_anom_swh_max_swan_vs_waverys.png')
plt.savefig(scatter_anom_path)
print(f"✅ Saved plot: {scatter_anom_path}")
plt.close()

# --- PLOT 7: AEP curve with observed yearly losses overlay ---
import glob
import matplotlib.ticker as mticker
import seaborn as sns

curve_files = sorted(glob.glob(os.path.join(run_dir, 'aep_curve_*.csv')))
obs_losses_files = sorted(glob.glob(os.path.join(run_dir, 'observed_yearly_losses_*.csv')))

if curve_files:
    aep_curve_path = curve_files[-1]
    aep_curve = pd.read_csv(aep_curve_path)
    plt.figure(figsize=(12, 7))
    # AEP curve as smooth blue line
    plt.plot(aep_curve['loss'], aep_curve['probability'], color='blue', lw=2.5, marker='o', markersize=3)
    plt.xlabel('Loss ($)', fontsize=14)
    plt.ylabel('Exceedance Probability', fontsize=14)
    plt.title('Annual Exceedance Probability Curve', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 1)

    # Overlay observed yearly losses
    if obs_losses_files:
        obs_path = obs_losses_files[-1]
        obs_df = pd.read_csv(obs_path)
        print('DEBUG: observed losses loaded:')
        print(obs_df)
        palette = sns.color_palette('tab10', n_colors=len(obs_df))
        for idx, row in obs_df.iterrows():
            plt.axvline(row['observed_loss'], color=palette[idx], linestyle='--', linewidth=3, alpha=0.8)
            plt.text(row['observed_loss'], 0.98, str(row['year']), rotation=90, color=palette[idx], fontsize=16, ha='center', va='top', fontweight='bold')
        # Median line
        median_loss = obs_df['observed_loss'].median()
        plt.axvline(median_loss, color='green', linestyle=':', linewidth=2.5)
        plt.text(median_loss, 0.92, f'Median\n${median_loss/1000:.0f}K', color='green', fontsize=14, fontweight='bold', ha='center', va='top')

    # Format x-axis as $X,XXXK
    def k_fmt(x, pos):
        return f'${int(x/1000):,}K'
    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(k_fmt))

    plt.tight_layout()
    out_path = os.path.join(run_dir, 'aep_with_observed_losses.png')
    plt.savefig(out_path)
    print(f"✅ Saved plot: {out_path}")
    plt.close()
