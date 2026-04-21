import pandas as pd
import numpy as np

clouds = pd.read_csv('bkn_sct_800_1500_no_lower_no_precip_no_fog.csv')
clear = pd.read_csv('clear_skies.csv')

reg = pd.read_csv('combined_df_reg_withday.csv')
resam = pd.read_csv('combined_df_resam_withday.csv')

# Prepare the lidar-based dfs.
# combined_df_reg['Unnamed: 0'] like "YYYY-MM-DD HH:MM:SS+00.00" -> normalize to "+00:00"
# reg = combined_df_reg.copy()
reg_time_str = reg['Unnamed: 0'].astype(str).str.replace(
    r'([+-]\d{2})[.:]?(\d{2})?$',
    lambda m: f"{m.group(1)}:{m.group(2) if m.group(2) else '00'}",
    regex=True,
)
reg['dt'] = pd.to_datetime(reg_time_str, utc=True, errors='coerce')

# Repeat for resampled
resam_time_str = resam['Center Datetime'].astype(str).str.replace(
    r'([+-]\d{2})[.:]?(\d{2})?$',
    lambda m: f"{m.group(1)}:{m.group(2) if m.group(2) else '00'}",
    regex=True,
)
resam['dt'] = pd.to_datetime(resam_time_str, utc=True, errors='coerce')

# Optional: drop rows with unparsable timestamps
reg = reg.dropna(subset=['dt'])
resam = resam.dropna(subset=['dt'])

# --- Sort for merge_asof ---
reg = reg.sort_values('dt')
resam = resam.sort_values('dt')


# ** Match with cloudy METAR **

# --- Parse datetimes ---
# clouds['datetime'] like "YYYY-MM-DD HH:MM:SS" (assume UTC; make it tz-aware)
clouds_dt = clouds.copy()
clouds_dt['dt'] = pd.to_datetime(clouds_dt['datetime'], utc=True, errors='coerce')

# Optional: drop rows with unparsable timestamps
clouds_dt = clouds_dt.dropna(subset=['dt'])

# --- Sort for merge_asof ---
clouds_dt = clouds_dt.sort_values('dt')

# --- As-of merge with 10-minute tolerance ---
matched = pd.merge_asof(
    reg,
    clouds_dt[['dt']].rename(columns={'dt': 'cloud_time'}),
    left_on='dt',
    right_on='cloud_time',
    tolerance=pd.Timedelta('30min'),
    direction='nearest'
)

# Keep only rows that found a cloud_time within the tolerance
within_10_min = matched[matched['cloud_time'].notna()].drop(columns=['cloud_time'])
# within_10_min contains all rows from combined_df_reg that are within 10 minutes of any clouds datetime


# Repeat for resampled values:# --- As-of merge with 10-minute tolerance ---
matched = pd.merge_asof(
    resam,
    clouds_dt[['dt']].rename(columns={'dt': 'cloud_time'}),
    left_on='dt',
    right_on='cloud_time',
    tolerance=pd.Timedelta('30min'),
    direction='nearest'
)

# Keep only rows that found a cloud_time within the tolerance
within_10_min_resam = matched[matched['cloud_time'].notna()].drop(columns=['cloud_time'])


# ** Match with clear METAR **

# --- Parse datetimes ---
# clouds['datetime'] like "YYYY-MM-DD HH:MM:SS" (assume UTC; make it tz-aware)
clear_dt = clear.copy()
clear_dt['dt'] = pd.to_datetime(clear_dt['datetime'], utc=True, errors='coerce')

# Optional: drop rows with unparsable timestamps
clear_dt = clear_dt.dropna(subset=['dt'])

# --- Sort for merge_asof ---
clear_dt = clear_dt.sort_values('dt')

# --- As-of merge with 10-minute tolerance ---
matched = pd.merge_asof(
    reg,
    clear_dt[['dt']].rename(columns={'dt': 'clear_time'}),
    left_on='dt',
    right_on='clear_time',
    tolerance=pd.Timedelta('30min'),
    direction='nearest'
)

# Keep only rows that found a cloud_time within the tolerance
within_10_min_clear = matched[matched['clear_time'].notna()].drop(columns=['clear_time'])
# within_10_min contains all rows from combined_df_reg that are within 10 minutes of any clouds datetime


# Repeat for resampled values:# --- As-of merge with 10-minute tolerance ---
matched = pd.merge_asof(
    resam,
    clear_dt[['dt']].rename(columns={'dt': 'clear_time'}),
    left_on='dt',
    right_on='clear_time',
    tolerance=pd.Timedelta('30min'),
    direction='nearest'
)

# Keep only rows that found a cloud_time within the tolerance
within_10_min_resam_clear = matched[matched['clear_time'].notna()].drop(columns=['clear_time'])



cond = (within_10_min['Invalid Adjacent'] == False) * (within_10_min['isDay'] == True) * (within_10_min['Chord Length'] > 200) #* (combined_df_reg['Chord Length'] < 1000)
condr = (within_10_min_resam['Invalid Adjacent'] == False) * (within_10_min_resam['isDay'] == True) * (within_10_min_resam['Chord Length'] > 200) #* (combined_df_resam['Chord Length'] < 1000)

condclr = (within_10_min_clear['Invalid Adjacent'] == False) * (within_10_min_clear['isDay'] == True) * (within_10_min_clear['Chord Length'] > 200) #* (combined_df_reg['Chord Length'] < 1000)
condrclr = (within_10_min_resam_clear['Invalid Adjacent'] == False) * (within_10_min_resam_clear['isDay'] == True) * (within_10_min_resam_clear['Chord Length'] > 200) #* (combined_df_resam['Chord Length'] < 1000)

A1_count = within_10_min_resam[condr].groupby('Height')['Chord Length'].count()
A2_count = within_10_min[cond].groupby('Height')['Chord Length'].count()

# Percentile
A1 = within_10_min_resam[condr].groupby('Height')['Chord Length'].quantile(.99)
A2 = within_10_min[cond].groupby('Height')['Chord Length'].quantile(.99)
diff = A2.values - A1.values
dLdz = np.diff(diff)    # <--- This looks like a bad idea.

# Median
A1_median = within_10_min_resam[condr].groupby('Height')['Chord Length'].median()
A2_median = within_10_min[cond].groupby('Height')['Chord Length'].median()
median_diff = A2_median.values - A1_median.values
dmediandz = np.diff(median_diff)

# Mean
A1_mean = within_10_min_resam[condr].groupby('Height')['Chord Length'].mean()
A2_mean = within_10_min[cond].groupby('Height')['Chord Length'].mean()


# Plot median of largest X chords.
A1_large = within_10_min_resam[condr].groupby('Height', group_keys=False).apply(lambda x: x.nlargest(100000, 'Chord Length')['Chord Length'].median())
A2_large = within_10_min[cond].groupby('Height', group_keys=False).apply(lambda x: x.nlargest(100000, 'Chord Length')['Chord Length'].median())


from matplotlib import pyplot as plt, cm
# Plot count.
fig, ax = plt.subplots(1,1,figsize=(6,6))
# ax.plot(A1_count,A1.index*1000,'r',label='Resampled at 795 m')
# ax.plot(A2_count,A2.index,'b',label='By Height')
ax.plot(A1_count.values-A2_count.values,A1.index*1000,'k',label='Difference')
h,l = ax.get_legend_handles_labels()
ax.legend(h,l)
ax.set_xlabel('Number of eddies')
ax.set_ylabel('Height (m)')
ax.set_ylim([0,2000])
# fig.savefig('eddy_count_vs_height.pdf')

# Plot Xth percentile.
fig, ax = plt.subplots(1,1,figsize=(6,6))
# ax.plot(A1,A1.index*1000,'r',label='Resampled at 795 m')
# ax.plot(A2,A2.index,'b',label='By Height')
ax.plot(A1.values-A2.values,A1.index*1000,'k',label='Difference')
h,l = ax.get_legend_handles_labels()
ax.legend(h,l)
ax.set_xlabel('95th percentile chord length (m)')
ax.set_ylabel('Height (m)')
ax.set_ylim([0,2000])
# fig.savefig('chordlength_95pct_vs_height.pdf')

# Plot median.
fig, ax = plt.subplots(1,1,figsize=(6,6))
# ax.plot(A1_median,A1.index*1000,'r',label='Resampled at 795 m')
# ax.plot(A2_median,A2.index,'b',label='By Height')
ax.plot(A1_median.values-A2_median.values,A1_median.index*1000,'k',label='Difference')
h,l = ax.get_legend_handles_labels()
ax.legend(h,l)
ax.set_xlabel('Median chord length (m)')
ax.set_ylabel('Height (m)')
ax.set_ylim([0,2000])
# fig.savefig('chordlength_median_vs_height.pdf')

# Plot mean.
fig, ax = plt.subplots(1,1,figsize=(6,6))
# ax.plot(A1_mean,A1.index*1000,'r',label='Resampled at 795 m')
# ax.plot(A2_mean,A2.index,'b',label='By Height')
ax.plot(A1_mean.values-A2_mean.values,A1.index*1000,'k',label='Difference')
h,l = ax.get_legend_handles_labels()
ax.legend(h,l)
ax.set_xlabel('Mean chord length (m)')
ax.set_ylabel('Height (m)')
ax.set_ylim([0,2000])
# fig.savefig('chordlength_mean_vs_height.pdf')

# # Plot median of largest X chords.
fig, ax = plt.subplots(1,1,figsize=(6,6))
ax.plot(A1_large,A1_large.index*1000,'r',label='Resampled at 795 m')
ax.plot(A2_large,A2_large.index,'b',label='By Height')
# ax.plot(A1.values-A2.values,A1.index*1000,'k',label='Difference')
h,l = ax.get_legend_handles_labels()
ax.legend(h,l)
ax.set_xlabel('Median length of largest 100,000 chords (m)')
ax.set_ylabel('Height (m)')
ax.set_ylim([0,2000])
# fig.savefig('chordlength_largest100000_median_vs_height.pdf')
