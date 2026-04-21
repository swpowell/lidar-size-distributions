import xarray as xr
from glob import glob
import pandas as pd

ceilometer_dir = '/thumper/metdata/ceilometer/SGP/C1/'
TSI_dir = '/thumper/metdata/TSI/SGP/'

ceil_files = sorted(glob(ceilometer_dir + "*.nc"))
TSI_files = sorted(glob(TSI_dir + "*.cdf"))

# NOTE: Only need data from 2014 to 2023 to match lidar data.
# NOTE: Currently adding 2010 - 2013 to database.

ceil_files = [
    f for f in ceil_files
    if 2013 <= int(f.split('.')[-3][:4]) <= 2024
]

TSI_files = [
    f for f in TSI_files
    if 2013 <= int(f.split('.')[-3][:4]) <= 2024
]


# Open all the TSI data into a single object (total about 1 GB).
# TSI_data = xr.open_mfdataset(
#     TSI_files,
#     combine='by_coords',
#     parallel=True,
#     preprocess=lambda ds: ds[['time','percent_opaque']]
# )

TSI_data = xr.open_mfdataset(TSI_files)

timestamp = TSI_data['time'].values
cloud_cover = TSI_data['percent_opaque'].values #+ TSI_data['percent_thin'].values

# Create the DataFrame
cc = pd.DataFrame({
    'cloud_cover': cloud_cover
}, index=timestamp)

# Sort by time if not already sorted
cc = cc.sort_index()

# Remove rows with negative/missing cloud cover.
cc = cc[cc['cloud_cover'] >= 0]




# Next, let's work with the ceilometer data (total about 20 GB).
ceil_data = xr.open_mfdataset(
    ceil_files,
    combine='by_coords',
    parallel=True,
    preprocess=lambda ds: ds[['time', 'first_cbh', 'qc_first_cbh']]
)


# Convert to pandas DataFrame
ceil = ceil_data[['first_cbh', 'qc_first_cbh']].to_dataframe()

# Set time as index
ceil.index = pd.to_datetime(ceil_data['time'].values)

# Filter out invalid rows
ceil = ceil[
    (ceil['first_cbh'] >= 0) &
    (ceil['qc_first_cbh'] != 1)
].dropna(subset=['first_cbh'])

# Sort by time if not already sorted
ceil = ceil.sort_index()

# Drop qc column
ceil = ceil.drop(columns=['qc_first_cbh'])



# Perform an asof merge to find the nearest timestamp in cc for each timestamp in ceil
merged = pd.merge_asof(
    ceil,
    cc,
    left_index=True,
    right_index=True,
    direction='nearest',
    tolerance=pd.Timedelta(seconds=300)
)

merged.to_csv('cloud_base_height_and_cover_V2.csv')


# Filter rows where first_cbh is between 500 and 1500
# and cloud_cover is between 0.25 and 0.75
filtered = merged[
    (merged['first_cbh'] >= 500) & (merged['first_cbh'] <= 1500) &
    (merged['cloud_cover'] >= 25) & (merged['cloud_cover'] <= 75)
]

# 2013-01-01 14:26:17 is an example. Check TSI images.
# BUG: This time in image was obviously 100% cloud cover with precipitation.
# BUG: However, the dataframe has 25-45% cloud cover at this time.
# BUG: Dataframe cloud cover is not close to 100% even when adding percent_opaque and percent_thin.