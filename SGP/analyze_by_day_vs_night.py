import pandas as pd
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
from concurrent.futures import ThreadPoolExecutor
from astral import LocationInfo
from astral.sun import sun
import datetime
import pytz

# Define location for sunrise/sunset calculations.
location = LocationInfo(name="SGP", region="USA", timezone="US/Central", latitude=36.605, longitude=-97.485)
local_tz = pytz.timezone(location.timezone)

def getsuntimes(date):
    return {
    date: sun(location.observer, date=date, tzinfo=local_tz)
}

# Get file list.
basedir = '/thumper/users/scott.powell/code-data/research-code/lidar/SGP/'
locations = ['C1','E13','E32','E37','E39','E41']
fnames = []
for loc in locations:
    fnames.append(sorted(glob(basedir+loc+"/*.csv")))
fnames = [i for j in fnames for i in j]

# Read each CSV file and concatenate them into a single DataFrame
# df_list = [pd.read_csv(fname) for fname in fnames[:2142] + fnames[2287:]]
df_list = [pd.read_csv(fname,index_col=0) for fname in fnames]
combined_df_reg = pd.concat(df_list)

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

# Get file list.
fnames500 = []
for loc in locations:
    fnames500.append(sorted(glob(basedir+loc+"_resample/*.csv")))
# fnames = sorted(glob(fdir+"*.csv"))
fnames500 = [i for j in fnames500 for i in j]

# Read each CSV file and concatenate them into a single DataFrame
# df_list = [pd.read_csv(fname) for fname in fnames[:2142] + fnames[2287:]]
df_list = [pd.read_csv(fname,index_col=0) for fname in fnames500]
combined_df_resam = pd.concat(df_list)

# Remove rows where Height is approximately equal to any value in the array
mask = np.isclose(combined_df_resam['Height'].values[:, None], values_to_keep*0.001, rtol=1e-5, atol=1e-8).any(axis=1)
combined_df_resam = combined_df_resam[mask]

# For some reason, the dataframes have a few rows that did not get the index convert to YYYY-MM-DD HH:MM:SS format.
# Remove these rows.

# NOTE: NEED TO CONVERT INDICES NOT FROM C1 TO PD.DATETIME TIMESTAMPS.

combined_df_reg = combined_df_reg[combined_df_reg.index.map(type) == str]
combined_df_resam = combined_df_resam[combined_df_resam.index.map(type) == str]

combined_df_reg.index = pd.to_datetime(combined_df_reg.index,utc=True)
combined_df_resam.index = pd.to_datetime(combined_df_resam.index,utc=True)

combined_df_reg['local_time'] = combined_df_reg.index.tz_convert(local_tz)
combined_df_resam['local_time'] = combined_df_resam.index.tz_convert(local_tz)



#%% Get sunrise/sunset for unique dates
# Get unique dates
unique_dates = np.unique(combined_df_reg.index.date)

# Compute sunrise/sunset for each date
with ThreadPoolExecutor() as executor:
    sun_times = list(executor.map(getsuntimes, unique_dates))
sun_times = {k:v for d in sun_times for k, v in d.items()}

combined_df_reg['sunrise'] = pd.Series(combined_df_reg.index.date,index=combined_df_reg.index).map(lambda d: sun_times[d]['sunrise'])
combined_df_reg['sunset'] = pd.Series(combined_df_reg.index.date,index=combined_df_reg.index).map(lambda d: sun_times[d]['sunset'])

combined_df_reg['isDay'] = (combined_df_reg['sunrise'] < combined_df_reg['local_time']) & (combined_df_reg['sunset'] > combined_df_reg['local_time'])




#%% Repeat for resampled period.
# Get unique dates
unique_dates = np.unique(combined_df_resam.index.date)

# Compute sunrise/sunset for each date
with ThreadPoolExecutor() as executor:
    sun_times = list(executor.map(getsuntimes, unique_dates))
sun_times = {k:v for d in sun_times for k, v in d.items()}

combined_df_resam['sunrise'] = pd.Series(combined_df_resam.index.date,index=combined_df_resam.index).map(lambda d: sun_times[d]['sunrise'])
combined_df_resam['sunset'] = pd.Series(combined_df_resam.index.date,index=combined_df_resam.index).map(lambda d: sun_times[d]['sunset'])

combined_df_resam['isDay'] = (combined_df_resam['sunrise'] < combined_df_resam['local_time']) & (combined_df_resam['sunset'] > combined_df_resam['local_time'])


#%%
combined_df_reg = pd.read_csv('combined_df_reg_withday.csv')
combined_df_resam = pd.read_csv('combined_df_resam_withday.csv')


# %%
# Now let's analyze only the daytime rows.
cond = (combined_df_reg['Invalid Adjacent'] == False) * (combined_df_reg['isDay'] == True) * (combined_df_reg['Chord Length'] > 200) #* (combined_df_reg['Chord Length'] < 1000)
condr = (combined_df_resam['Invalid Adjacent'] == False) * (combined_df_resam['isDay'] == True) * (combined_df_resam['Chord Length'] > 200) #* (combined_df_resam['Chord Length'] < 1000)

#Let's grab just the daytime rows with Chord length exceeding 99th pctile.

# percentiles = combined_df_reg.groupby('Height')['Chord Length'].quantile(0.99).reset_index()
# percentiles.columns = ['Height', 'Chord_99th']


# combined_df_reg = combined_df_reg.merge(percentiles, on='Height')
# bigchords = combined_df_reg[cond * (combined_df_reg['Chord Length'] > combined_df_reg['Chord_99th'])]




# Look at only eddies that are "large".
# combined_df_reg.loc[combined_df_reg['Chord Length'] < 100, 'Chord Length'] = None
# combined_df_resam.loc[combined_df_resam['Chord Length'] < 100, 'Chord Length'] = None

A1_count = combined_df_resam[condr].groupby('Height')['Chord Length'].count()
A2_count = combined_df_reg[cond].groupby('Height')['Chord Length'].count()

# Percentile
A1 = combined_df_resam[condr].groupby('Height')['Chord Length'].quantile(.99)
A2 = combined_df_reg[cond].groupby('Height')['Chord Length'].quantile(.99)
diff = A2.values - A1.values
dLdz = np.diff(diff)

# Median
A1_median = combined_df_resam[condr].groupby('Height')['Chord Length'].median()
A2_median = combined_df_reg[cond].groupby('Height')['Chord Length'].median()
median_diff = A2_median.values - A1.median.values
dmediandz = np.diff(median_diff)

# Mean
A1_mean = combined_df_resam[condr].groupby('Height')['Chord Length'].mean()
A2_mean = combined_df_reg[cond].groupby('Height')['Chord Length'].mean()


# Plot median of largest X chords.
A1_large = combined_df_resam[condr].groupby('Height', group_keys=False).apply(lambda x: x.nlargest(100000, 'Chord Length')['Chord Length'].median())
A2_large = combined_df_reg[cond].groupby('Height', group_keys=False).apply(lambda x: x.nlargest(100000, 'Chord Length')['Chord Length'].median())



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
ax.plot(A1_median.values-A2_median.values,A1.index*1000,'k',label='Difference')
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



# %%

# Grab the TSI data and put cloud fraction (or nighttime in a column).
# TSI data directory
# tsipath = '/thumper/metdata/TSI/SGP/'
# tsifiles = sorted(glob(tsipath+"*.cdf"))
