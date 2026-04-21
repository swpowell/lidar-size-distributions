import xarray as xr
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
import pandas as pd
from joblib import Parallel, delayed


# ***************************************************

# Function to find the nearest value in heights_selected for each time in stattimes
def find_nearest_values(stattimes, heights_selected):
    nearest_values = []
    for time in stattimes:
        # Find the index of the nearest time in heights_selected
        nearest_time_index = np.abs(heights_selected.time - time).argmin()
        # Get the value at the nearest time
        nearest_value = heights_selected.isel(time=nearest_time_index).values
        nearest_values.append(nearest_value)
    return xr.DataArray(nearest_values, coords=[stattimes], dims="time")

def find_nearest_indices(hgt, stats_height):
    nearest_indices = []
    for value in hgt:
        if np.isnan(value):
            nearest_indices.append(np.nan)
        else:
            # Find the index of the nearest value in stats.height
            nearest_index = int(np.abs(stats_height - value).argmin().values)
            nearest_indices.append(nearest_index)
    return nearest_indices

# ***************************************************

def getPBLH(pblh,stattimes):

    stacked = np.stack([pblh.bl_index_1, pblh.bl_index_2, pblh.bl_index_3])
    stacked[stacked == np.nan] = 0
    # Mask NaN positions
    valid_mask = ~np.isnan(stacked)  # True where values are NOT NaN
    # Set NaN positions to a very low value before applying argmax
    stacked_filled = np.where(valid_mask, stacked, -np.inf)
    # Compute max indices safely
    max_indices = np.argmax(stacked_filled, axis=0)
    # Set indices to -1 where all values were NaN
    max_indices[~valid_mask.any(axis=0)] = -1  # Mark all-NaN positions
    # Get heights.
    heights = np.stack([pblh.bl_height_1, pblh.bl_height_2, pblh.bl_height_3])
    heights_selected = np.where(max_indices != -1, heights[max_indices, np.arange(stacked.shape[1])], np.nan)
    # Make into xarray dataarray.
    heights_selected = xr.DataArray(
        heights_selected,
        coords={"time": pblh.bl_height_1.time},  # Copy time coordinate
        attrs={"long_name": "Best boundary layer height", "units": 'meters'},
        dims="time"
    )

    # Find the nearest values
    pbl_height = find_nearest_values(stattimes, heights_selected)

    return pbl_height

# ***************************************************

statdir = '/thumper/metdata/doppler-lidar/SGP/stats/'
pblhdir = '/thumper/metdata/PBLH/SGP/'

statnames = sorted(glob(statdir+"*.nc"))
pblhnames = sorted(glob(pblhdir+"*.cdf")) + sorted(glob(pblhdir+"*.nc"))

sblvals, cblvals = None, None

# This can be coded in parallel. 
for file in statnames[2000:2250]:

    print(file)

    # Find the corresponding file for PBL height.
    pblhname = [i for i in pblhnames if file[-18:-10] in i]

    if len(pblhname) == 1:

        stats = xr.open_dataset(file)
        pblh = xr.open_dataset(pblhname[0])

        # Get PBLH at each time.
        hgt = getPBLH(pblh,stats.time)

        # Convert to index in stats.
        nearest_indices = find_nearest_indices(hgt, stats.height)

        blclass = np.full(len(stats.time),'Unclassified')

        # Get the w variance
        wvar = stats['w_variance']

        wvarmean = np.full(len(nearest_indices),np.nan)
        for time, i in enumerate(nearest_indices):
            if ~np.isnan(i):
                wvarmean[i] = wvar[time,:i].mean()

        # Classified based on average value up to PBL. 
        blclass[wvarmean >= 0.25] = 'CBL'
        blclass[wvarmean <= 0.1] = 'SBL'

        if 'SBL' in blclass:
            if sblvals is None:
                # sblvals = wvar[blclass=='SBL',:100]
                sblvals = hgt[blclass=='SBL']
            else:
                # sblvals = xr.concat((sblvals,wvar[blclass=='SBL',:100]), dim="time")
                sblvals = xr.concat((sblvals,hgt[blclass=='SBL']),dim="time")

        if 'CBL' in blclass:
            if cblvals is None:
                # cblvals = wvar[blclass=='CBL',:100]
                cblvals = hgt[blclass=='CBL']
            else:
                # cblvals = xr.concat((cblvals,wvar[blclass=='CBL',:100]), dim="time")
                cblvals = xr.concat((cblvals,hgt[blclass=='CBL']),dim="time")


    
