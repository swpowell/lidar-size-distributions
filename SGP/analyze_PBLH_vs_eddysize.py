import pandas as pd
from glob import glob
import numpy as np
from matplotlib import pyplot as plt

clouds = pd.read_csv('bkn_sct_1500_5000_no_lower_no_precip_no_fog.csv')
clear = pd.read_csv('clear_skies.csv')

basedir = '/thumper/users/scott.powell/code-data/research-code/lidar/SGP/updraft_output/'
fnames = sorted(glob(basedir+"/*.csv"))

# Read each CSV file and concatenate them into a single DataFrame
# df_list = [pd.read_csv(fname) for fname in fnames[:2142] + fnames[2287:]]
df_list = [pd.read_csv(fname) for fname in fnames]
combined_df_reg = pd.concat(df_list, ignore_index=True)

# Heights of most files:
values_to_keep = np.array([  45.,   75.,  105.,  135.,  165.,  195.,  225.,  255.,  285.,
        315.,  345.,  375.,  405.,  435.,  465.,  495.,  525.,  555.,
        585.,  615.,  645.,  675.,  705.,  735.,  765.,  795.,  825.,
        855.,  885.,  915.,  945.,  975., 1005., 1035., 1065., 1095.,
       1125., 1155., 1185., 1215., 1245., 1275., 1305., 1335., 1365.,
       1395., 1425., 1455., 1485., 1515., 1545., 1575., 1605., 1635.,
       1665., 1695., 1725., 1755., 1785., 1815., 1845., 1875., 1905.,
       1935., 1965., 1995.,   15.])

# Remove rows where Height is approximately equal to any value in the array
mask = np.isclose(combined_df_reg['Height'].values[:, None], values_to_keep, rtol=1e-5, atol=1e-8).any(axis=1)
combined_df_reg = combined_df_reg[mask]


# *********** 

reg_time_str = combined_df_reg['Center Datetime'].astype(str).str.replace(
    r'([+-]\d{2})[.:]?(\d{2})?$',
    lambda m: f"{m.group(1)}:{m.group(2) if m.group(2) else '00'}",
    regex=True,
)
combined_df_reg['dt'] = pd.to_datetime(reg_time_str, utc=True, errors='coerce')

# Optional: drop rows with unparsable timestamps
combined_df_reg = combined_df_reg.dropna(subset=['dt'])

# --- Sort for merge_asof ---
combined_df_reg = combined_df_reg.sort_values('dt')

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
    combined_df_reg,
    clouds_dt[['dt']].rename(columns={'dt': 'cloud_time'}),
    left_on='dt',
    right_on='cloud_time',
    tolerance=pd.Timedelta('30min'),
    direction='nearest'
)

# Keep only rows that found a cloud_time within the tolerance
within_10_min = matched[matched['cloud_time'].notna()].drop(columns=['cloud_time'])
# within_10_min contains all rows from combined_df_reg that are within 10 minutes of any clouds datetime


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
    combined_df_reg,
    clear_dt[['dt']].rename(columns={'dt': 'clear_time'}),
    left_on='dt',
    right_on='clear_time',
    tolerance=pd.Timedelta('30min'),
    direction='nearest'
)

# Keep only rows that found a cloud_time within the tolerance
within_10_min_clear = matched[matched['clear_time'].notna()].drop(columns=['clear_time'])
# within_10_min contains all rows from combined_df_reg that are within 10 minutes of any clouds datetime








# *********** Make plots ************


combined_df_reg['PBLH Bin'] = 100*np.floor(0.01*combined_df_reg['PBL Height'])
within_10_min['PBLH Bin'] = 100*np.floor(0.01*within_10_min['PBL Height'])
within_10_min_clear['PBLH Bin'] = 100*np.floor(0.01*within_10_min_clear['PBL Height'])


# Plot all.

# binned = combined_df_reg.groupby(['PBLH Bin','Height'])['Chord Length'].quantile(.5)
binned = combined_df_reg.groupby(['Height'])['Chord Length'].quantile(.5)
binned_cloud = within_10_min.groupby(['Height'])['Chord Length'].quantile(.5)
binned_clear = within_10_min_clear.groupby(['Height'])['Chord Length'].quantile(.5)

count = combined_df_reg.groupby(['Height'])['Chord Length'].count()
count_cloud = within_10_min.groupby(['Height'])['Chord Length'].count()
count_clear = within_10_min_clear.groupby(['Height'])['Chord Length'].count()


# binned = binned.reset_index()
# binned_cloud = binned_cloud.reset_index()
# binned_clear = binned_clear.reset_index()

# Plot unconditioned chord length vs height.

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(binned.values[1:],binned.index[1:],'k', label='All Eddies')
ax.plot(binned_cloud.values[1:],binned_cloud.index[1:],'b', label='Cloudy conditions')
ax.plot(binned_clear.values[1:],binned_clear.index[1:],'r', label='Clear conditions')

h, l = ax.get_legend_handles_labels()
ax.legend(h, l)
ax.set_xlabel('Chord length (m)')
ax.set_ylabel('Height (m)')
fig.savefig('Cloudy_clear_eddy_chordlength.pdf')

# Conditioned on PBLH if correct line above is uncommented.

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

ax.plot(binned[binned['PBLH Bin'] == 500]['Chord Length'].values[1:], round(binned[binned['PBLH Bin'] == 500]['Height']).values[1:], 'b', label='500m')
ax.plot(binned[binned['PBLH Bin'] == 700]['Chord Length'].values[1:], round(binned[binned['PBLH Bin'] == 700]['Height']).values[1:], 'r', label='700m')
ax.plot(binned[binned['PBLH Bin'] == 900]['Chord Length'].values[1:], round(binned[binned['PBLH Bin'] == 900]['Height']).values[1:], 'm', label='900m')
ax.plot(binned[binned['PBLH Bin'] == 1100]['Chord Length'].values[1:], round(binned[binned['PBLH Bin'] == 1100]['Height']).values[1:], 'k', label='1100m')
ax.plot(binned[binned['PBLH Bin'] == 1300]['Chord Length'].values[1:], round(binned[binned['PBLH Bin'] == 1300]['Height']).values[1:], 'b--', label='1300m')
ax.plot(binned[binned['PBLH Bin'] == 1500]['Chord Length'].values[1:], round(binned[binned['PBLH Bin'] == 1500]['Height']).values[1:], 'r--', label='1500m')
ax.plot(binned[binned['PBLH Bin'] == 1700]['Chord Length'].values[1:], round(binned[binned['PBLH Bin'] == 1700]['Height']).values[1:], 'm--', label='1700m')
ax.plot(binned[binned['PBLH Bin'] == 1900]['Chord Length'].values[1:], round(binned[binned['PBLH Bin'] == 1900]['Height']).values[1:], 'k--', label='1900m')

h, l = ax.get_legend_handles_labels()
ax.legend(h, l, title='PBLH Height')
ax.set_xlabel('Chord length (m)')
ax.set_ylabel('Height (m)')
# fig.savefig('Size_vs_PBLH.pdf')

# Plot clear.

binned = within_10_min_clear.groupby(['PBLH Bin','Height'])['Chord Length'].quantile(.5)
binned = binned.reset_index()

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

ax.plot(binned[binned['PBLH Bin'] == 500]['Chord Length'].values[1:], round(binned[binned['PBLH Bin'] == 500]['Height']).values[1:], 'b', label='500m')
ax.plot(binned[binned['PBLH Bin'] == 700]['Chord Length'].values[1:], round(binned[binned['PBLH Bin'] == 700]['Height']).values[1:], 'r', label='700m')
ax.plot(binned[binned['PBLH Bin'] == 900]['Chord Length'].values[1:], round(binned[binned['PBLH Bin'] == 900]['Height']).values[1:], 'm', label='900m')
ax.plot(binned[binned['PBLH Bin'] == 1100]['Chord Length'].values[1:], round(binned[binned['PBLH Bin'] == 1100]['Height']).values[1:], 'k', label='1100m')
ax.plot(binned[binned['PBLH Bin'] == 1300]['Chord Length'].values[1:], round(binned[binned['PBLH Bin'] == 1300]['Height']).values[1:], 'b--', label='1300m')
ax.plot(binned[binned['PBLH Bin'] == 1500]['Chord Length'].values[1:], round(binned[binned['PBLH Bin'] == 1500]['Height']).values[1:], 'r--', label='1500m')
ax.plot(binned[binned['PBLH Bin'] == 1700]['Chord Length'].values[1:], round(binned[binned['PBLH Bin'] == 1700]['Height']).values[1:], 'm--', label='1700m')
ax.plot(binned[binned['PBLH Bin'] == 1900]['Chord Length'].values[1:], round(binned[binned['PBLH Bin'] == 1900]['Height']).values[1:], 'k--', label='1900m')

h, l = ax.get_legend_handles_labels()
ax.legend(h, l, title='PBLH Height')
ax.set_xlabel('Chord length (m)')
ax.set_ylabel('Height (m)')
# fig.savefig('Size_vs_PBLH.pdf')