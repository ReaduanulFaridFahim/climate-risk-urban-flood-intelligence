"""
Generate publication-quality diagnostic plots for:
1. Advance 20-Day Weather & Hazard Forecast Outlook (ECMWF Ensemble) across Bangladesh
2. Live Satellite Telemetry & Automated Cell Broadcast Dispatcher Architecture
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import datetime

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8

os.makedirs('assets/models', exist_ok=True)

# 1. Load 60-Day & 20-Day Weather Forecasts
fc_60_path = 'data/processed/bangladesh_60day_weather_forecast.json'
fc_20_path = 'data/processed/bangladesh_20day_weather_forecast.json'

with open(fc_60_path if os.path.exists(fc_60_path) else fc_20_path, encoding='utf-8') as f:
    fc_60_data = json.load(f)

# Plot 1: Advance 60-Day (2-Month) Weather & Winter Cold Wave Outlook across 4 Critical Climate Zones
fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
fig.suptitle('Advance 60-Day (2-Month) Weather & Winter Cold Wave Natural Disaster Outlook (ECMWF & Sub-Seasonal Ensemble)', fontsize=16, fontweight='bold', y=0.98)

target_divs = ['Rangpur', 'Rajshahi', 'Sylhet', 'Dhaka']

for idx, div in enumerate(target_divs):
    ax = axes[idx // 2, idx % 2]
    div_fc = fc_60_data.get(div, [])
    if not div_fc:
        continue
    
    dates = [d['date'][5:] for d in div_fc] # MM-DD
    t_max = [d['temp_max_c'] for d in div_fc]
    t_min = [d['temp_min_c'] for d in div_fc]
    w_chill = [d.get('wind_chill_c', d['temp_min_c']) for d in div_fc]
    fog_hrs = [d.get('fog_duration_hrs', 0.0) for d in div_fc]
    precip = [d['precipitation_mm'] for d in div_fc]
    
    x = np.arange(len(dates))
    
    # BMD Official Cold Wave Danger Bands
    ax.axhspan(8.0, 10.0, color='#fef08a', alpha=0.35, label='Mild Cold Wave (8-10°C)')
    ax.axhspan(6.0, 8.0, color='#fed7aa', alpha=0.45, label='Moderate Cold Wave (6-8°C)')
    ax.axhspan(0.0, 6.0, color='#fecaca', alpha=0.55, label='Severe Cold Wave (<6°C)')
    
    # Temperature and Wind Chill Lines
    line_max, = ax.plot(x, t_max, color='#ea580c', marker='.', markersize=4, linewidth=2, label='Max Temp (°C)')
    line_min, = ax.plot(x, t_min, color='#0284c7', marker='s', markersize=3, linewidth=2, label='Min Temp (°C)')
    line_chill, = ax.plot(x, w_chill, color='#4338ca', linestyle='--', linewidth=1.8, label='Wind Chill (°C)')
    ax.fill_between(x, t_min, t_max, color='#38bdf8', alpha=0.12)
    
    # Danger Threshold Reference Lines
    ax.axhline(10.0, color='#eab308', linestyle=':', linewidth=1.2)
    ax.axhline(6.0, color='#dc2626', linestyle=':', linewidth=1.2)
    
    ax.set_ylabel('Temperature (°C)', color='#0f172a', fontweight='bold', fontsize=10)
    ax.set_title(f'{div} Division - 60-Day Winter Horizon & Cold Incursion Profile', fontweight='bold', fontsize=12, pad=10)
    ax.set_xticks(x[::5])
    ax.set_xticklabels(dates[::5], rotation=45, ha='right', fontsize=9)
    ax.set_ylim(-1, 37)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Twin axis for Dense Fog Duration (hours) and Precipitation
    ax2 = ax.twinx()
    bars_fog = ax2.bar(x, fog_hrs, width=0.5, color='#94a3b8', alpha=0.35, label='Dense Fog (Hrs/Day)')
    bars_p = ax2.bar(x, precip, width=0.3, color='#0284c7', alpha=0.5, label='Rainfall (mm)')
    ax2.set_ylabel('Fog (Hours) / Precip (mm)', color='#475569', fontweight='bold', fontsize=10)
    ax2.set_ylim(0, 16)
    ax2.grid(False)
    
    # Combined Legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=7.5, framealpha=0.9, ncol=2)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('assets/models/national_60day_weather_and_coldwave_outlook.png', dpi=300)
plt.savefig('assets/models/national_20day_weather_forecast_outlook.png', dpi=300)
plt.close()
print("Saved assets/models/national_60day_weather_and_coldwave_outlook.png and national_20day_weather_forecast_outlook.png")


# Plot 1B: Nationwide 8-Division Winter Cold Wave Vulnerability & Risk Matrix
div_stats = []
for div, days in fc_60_data.items():
    min_t = min(d['temp_min_c'] for d in days)
    min_chill = min(d.get('wind_chill_c', d['temp_min_c']) for d in days)
    cold_days = sum(1 for d in days if d['temp_min_c'] <= 10.0)
    severe_days = sum(1 for d in days if d['temp_min_c'] <= 6.0)
    avg_fog = np.mean([d.get('fog_duration_hrs', 0) for d in days[30:]]) # Month 2 fog
    div_stats.append({
        'Division': div,
        'Min_Temp': min_t,
        'Min_Wind_Chill': min_chill,
        'Cold_Wave_Days': cold_days,
        'Severe_Days': severe_days,
        'Peak_Month2_Fog_Hrs': avg_fog
    })

df_cstats = pd.DataFrame(div_stats).sort_values(by='Min_Temp')

fig_mat, (ax_t, ax_d) = plt.subplots(1, 2, figsize=(16, 6))
fig_mat.suptitle('Nationwide Regional Winter Cold Wave & Hypothermia Exposure Index (60-Day Outlook)', fontsize=15, fontweight='bold', y=0.98)

# Left: Lowest Minimum Temp & Wind Chill by Division
y_pos = np.arange(len(df_cstats))
width = 0.35
ax_t.barh(y_pos - width/2, df_cstats['Min_Temp'], width, label='Lowest Min Temp (°C)', color='#38bdf8', edgecolor='#0284c7')
ax_t.barh(y_pos + width/2, df_cstats['Min_Wind_Chill'], width, label='Lowest Perceived Wind Chill (°C)', color='#6366f1', edgecolor='#4338ca')
ax_t.axvline(10.0, color='#eab308', linestyle='--', linewidth=1.5, label='Mild Cold Wave (10°C)')
ax_t.axvline(6.0, color='#dc2626', linestyle='--', linewidth=1.5, label='Severe Cold Wave (6°C)')
ax_t.set_yticks(y_pos)
ax_t.set_yticklabels(df_cstats['Division'], fontweight='bold', fontsize=10)
ax_t.set_xlabel('Temperature (°C)', fontweight='bold')
ax_t.set_title('Lowest Temperature & Himalayan Wind Chill Depressions', fontweight='bold', pad=12)
ax_t.legend(loc='lower right', fontsize=8.5)
ax_t.grid(True, linestyle=':', alpha=0.6)

# Right: Total Cold Wave Days (BMD Threshold <= 10°C)
colors = ['#dc2626' if c > 10 else '#f97316' if c > 5 else '#eab308' if c > 0 else '#22c55e' for c in df_cstats['Cold_Wave_Days']]
bars_cw = ax_d.barh(y_pos, df_cstats['Cold_Wave_Days'], color=colors, edgecolor='#334155', height=0.6)
ax_d.set_yticks(y_pos)
ax_d.set_yticklabels(df_cstats['Division'], fontweight='bold', fontsize=10)
ax_d.set_xlabel('Total Projected Cold Wave Days (T_min ≤ 10°C)', fontweight='bold')
ax_d.set_title('Projected Cold Wave Duration across 60-Day Window', fontweight='bold', pad=12)
ax_d.grid(True, linestyle=':', alpha=0.6)

for bar in bars_cw:
    w = bar.get_width()
    ax_d.text(w + 0.3, bar.get_y() + bar.get_height()/2, f'{int(w)} days', ha='left', va='center', fontweight='bold', fontsize=9.5)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('assets/models/national_winter_coldwave_vulnerability_matrix.png', dpi=300)
plt.close()
print("Saved assets/models/national_winter_coldwave_vulnerability_matrix.png")


# Plot 2: Automated Cell Broadcast Alert Dispatch Audit Breakdown
with open('data/processed/bangladesh_automated_live_dispatches.json', encoding='utf-8') as f:
    dispatches = json.load(f)

df_disp = pd.DataFrame(dispatches)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Automated Emergency Cell Broadcast (CBS) Live Dispatch Analysis across Bangladesh', fontsize=15, fontweight='bold', y=0.98)

# Left: Dispatches by Hazard Type
hazard_counts = df_disp['hazard_type'].value_counts()
colors = ['#dc2626', '#f97316', '#eab308', '#06b6d4']
ax1.pie(hazard_counts, labels=hazard_counts.index, autopct='%1.1f%%', colors=colors[:len(hazard_counts)],
        startangle=140, explode=[0.05]*len(hazard_counts), textprops={'fontweight':'bold', 'fontsize':10})
ax1.set_title('Automated Live Dispatches by Hazard Category', fontweight='bold', pad=12)

# Right: Target Population Reach by Division
div_pop = df_disp.groupby('division')['target_recipients_est'].sum() / 1e6
div_pop = div_pop.sort_values(ascending=True)

bars = ax2.barh(div_pop.index, div_pop.values, color='#ef4444', edgecolor='#991b1b', height=0.6)
ax2.set_xlabel('Estimated Citizens Directly Reached via Mobile Broadcast (Millions)', fontweight='bold')
ax2.set_title('Geo-Fenced Mobile Phone Alert Penetration by Division', fontweight='bold', pad=12)
ax2.grid(True, linestyle=':', alpha=0.6)

for bar in bars:
    w = bar.get_width()
    ax2.text(w + 0.2, bar.get_y() + bar.get_height()/2, f'{w:.1f}M', ha='left', va='center', fontweight='bold', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('assets/models/automated_cell_broadcast_dispatch_analysis.png', dpi=300)
plt.close()
print("Saved assets/models/automated_cell_broadcast_dispatch_analysis.png")
