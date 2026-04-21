import pandas as pd
from glob import glob
import numpy as np
from matplotlib import pyplot as plt

def bootstrap(data, num_bootstrap_samples=1000, confidence_level=0.95):
    # Calculate the actual mean of the data
    actual_mean = np.mean(data)
    
    # Generate bootstrap samples and calculate their means
    bootstrap_means = [np.nanmean(np.random.choice(data, size=len(data), replace=True)) for _ in range(num_bootstrap_samples)]
    
    # Calculate the percentiles for the confidence interval
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 + confidence_level) / 2 * 100
    confidence_interval = np.percentile(bootstrap_means, [lower_percentile, upper_percentile])
    
    return actual_mean, confidence_interval[0], confidence_interval[1]

import numpy as np

def bootstrap_tail_comparison(data1, data2, num_bootstrap_samples=1000, tail_percentile=0.99):
    # Function to calculate the tail statistic (e.g., 95th percentile)
    def tail_statistic(data, percentile):
        return np.nanpercentile(data, percentile * 100)
    
    # Generate bootstrap samples and calculate tail statistics
    bootstrap_tail_stats1 = [tail_statistic(np.random.choice(data1, size=len(data1), replace=True), tail_percentile) for _ in range(num_bootstrap_samples)]
    bootstrap_tail_stats2 = [tail_statistic(np.random.choice(data2, size=len(data2), replace=True), tail_percentile) for _ in range(num_bootstrap_samples)]
    
    # Calculate the difference in tail statistics
    tail_differences = np.array(bootstrap_tail_stats1) - np.array(bootstrap_tail_stats2)
    
    # Calculate the confidence interval for the difference in tail statistics
    lower_bound = np.percentile(tail_differences, 2.5)
    upper_bound = np.percentile(tail_differences, 97.5)
    
    return lower_bound, upper_bound

# Example usage:
# Assuming data1 and data2 are numpy arrays containing your inverse exponential data
# lower_bound, upper_bound = bootstrap_tail_comparison(data1, data2)
# print(f"95% Confidence Interval for the difference in tails: ({lower_bound}, {upper_bound})")

# Get file list.
basedir = '/thumper/users/scott.powell/code-data/research-code/lidar/SGP/'
locations = ['C1','E13','E32','E37','E39','E41']
# locations = ['E41']
fnames = []
for loc in locations:
    fnames.append(sorted(glob(basedir+loc+"/*.csv")))
# fnames = sorted(glob(fdir+"*.csv"))
fnames = [i for j in fnames for i in j]

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

# Get file list.
fnames500 = []
for loc in locations:
    fnames500.append(sorted(glob(basedir+loc+"_resample/*.csv")))
# fnames = sorted(glob(fdir+"*.csv"))
fnames500 = [i for j in fnames500 for i in j]

# Read each CSV file and concatenate them into a single DataFrame
# df_list = [pd.read_csv(fname) for fname in fnames[:2142] + fnames[2287:]]
df_list = [pd.read_csv(fname) for fname in fnames500]
combined_df_resam = pd.concat(df_list, ignore_index=True).drop('Unnamed: 0', axis=1)

# Remove rows where Height is approximately equal to any value in the array
mask = np.isclose(combined_df_resam['Height'].values[:, None], values_to_keep*0.001, rtol=1e-5, atol=1e-8).any(axis=1)
combined_df_resam = combined_df_resam[mask]

# combined_df_reg.to_csv('combined_df_reg.csv')
# combined_df_resam.to_csv('combined_df_resam.csv')

combined_df_reg = pd.read_csv('combined_df_reg.csv')
combined_df_resam = pd.read_csv('combined_df_resam.csv')

##

cond = (combined_df_reg['Invalid Adjacent'] == False) * (combined_df_reg['Chord Length'] > 200) #* (combined_df_reg['Chord Length'] < 1000)
condr = (combined_df_resam['Invalid Adjacent'] == False) * (combined_df_resam['Chord Length'] > 200) #* (combined_df_resam['Chord Length'] < 1000)


# Look at only eddies that are "large".
# combined_df_reg.loc[combined_df_reg['Chord Length'] < 100, 'Chord Length'] = None
# combined_df_resam.loc[combined_df_resam['Chord Length'] < 100, 'Chord Length'] = None

# Percentile
A1 = combined_df_resam[condr].groupby('Height')['Chord Length'].quantile(.95)
A2 = combined_df_reg[cond].groupby('Height')['Chord Length'].quantile(.95)

# Median
A1_median = combined_df_resam[condr].groupby('Height')['Chord Length'].median()
A2_median = combined_df_reg[cond].groupby('Height')['Chord Length'].median()

# Mean
A1_mean = combined_df_resam[condr].groupby('Height')['Chord Length'].mean()
A2_mean = combined_df_reg[cond].groupby('Height')['Chord Length'].mean()


# Plot median of largest X chords.
A1_large = combined_df_resam[condr].groupby('Height', group_keys=False).apply(lambda x: x.nlargest(100000, 'Chord Length')['Chord Length'].median())
A2_large = combined_df_reg[cond].groupby('Height', group_keys=False).apply(lambda x: x.nlargest(100000, 'Chord Length')['Chord Length'].median())

A1_count = combined_df_resam[condr].groupby('Height')['Chord Length'].count()
A2_count = combined_df_reg[cond].groupby('Height')['Chord Length'].count()


# Plot count.
fig, ax = plt.subplots(1,1,figsize=(6,6))
ax.plot(A1_count,A1.index*1000,'r',label='Resampled at 795 m')
ax.plot(A2_count,A2.index,'b',label='By Height')
# ax.plot(A1_count.values-A2_count.values,A1.index*1000,'k',label='Difference')
h,l = ax.get_legend_handles_labels()
ax.legend(h,l)
ax.set_xlabel('Number of eddies')
ax.set_ylabel('Height (m)')
ax.set_ylim([0,2000])
# fig.savefig('eddy_count_vs_height.pdf')

# Plot Xth percentile.
fig, ax = plt.subplots(1,1,figsize=(6,6))
ax.plot(A1,A1.index*1000,'r',label='Resampled at 795 m')
ax.plot(A2,A2.index,'b',label='By Height')
# ax.plot(A1.values-A2.values,A1.index*1000,'k',label='Difference')
h,l = ax.get_legend_handles_labels()
ax.legend(h,l)
ax.set_xlabel('95th percentile chord length (m)')
ax.set_ylabel('Height (m)')
ax.set_ylim([0,2000])
# fig.savefig('chordlength_95pct_vs_height.pdf')

# Plot median.
fig, ax = plt.subplots(1,1,figsize=(6,6))
ax.plot(A1_median,A1.index*1000,'r',label='Resampled at 795 m')
ax.plot(A2_median,A2.index,'b',label='By Height')
# ax.plot(A1.values-A2.values,A1.index*1000,'k',label='Difference')
h,l = ax.get_legend_handles_labels()
ax.legend(h,l)
ax.set_xlabel('Median chord length (m)')
ax.set_ylabel('Height (m)')
ax.set_ylim([0,2000])
# fig.savefig('chordlength_median_vs_height.pdf')

# Plot mean.
fig, ax = plt.subplots(1,1,figsize=(6,6))
ax.plot(A1_mean,A1.index*1000,'r',label='Resampled at 795 m')
ax.plot(A2_mean,A2.index,'b',label='By Height')
# ax.plot(A1.values-A2.values,A1.index*1000,'k',label='Difference')
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