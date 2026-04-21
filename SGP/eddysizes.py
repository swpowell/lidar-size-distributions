import xarray as xr
import numpy as np
from scipy.ndimage import label
from glob import glob
import multiprocessing as mp
import re
from matplotlib import pyplot as plt
import pandas as pd
import datetime as dte
from joblib import Parallel, delayed
import os
from copy import deepcopy

# ****************************************************************************

# Function to get the date string in the format YYYYMMDD
def get_date_str(date):
    return date.strftime('%Y%m%d')

# ****************************************************************************

def simplesmoother(input2D,sigma=1):
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(input2D, sigma=sigma, axis=0)

# ****************************************************************************

def velocity_running_mean(name, previous_file, next_file, output_dir):
    # Open the current file using xarray
    ds = xr.open_dataset(name)
    
    # Load limited data from the previous file if provided
    if previous_file:
        ds_prev = xr.open_dataset(previous_file)
        format_ts = np.datetime_as_string(ds_prev['time'].values[0], unit='h')
        ds_prev = ds_prev.sel(time=slice(format_ts+':49:59', None))  # Load data at or after 23:50 UTC
    else:
        ds_prev = None
    
    # Load limited data from the next file if provided
    if next_file:
        ds_next = xr.open_dataset(next_file)
        format_ts = np.datetime_as_string(ds_next['time'].values[0], unit='h')
        ds_next = ds_next.sel(time=slice(None, format_ts+':10'))  # Load data at or before 00:10 UTC
    else:
        ds_next = None

    # Combine data from previous, current, and next files for running mean calculation
    if ds_prev is not None and ds_next is not None:
        combined_data = xr.concat([ds_prev['radial_velocity'], ds['radial_velocity'], ds_next['radial_velocity']], dim='time')
    elif ds_prev is not None:
        combined_data = xr.concat([ds_prev['radial_velocity'], ds['radial_velocity']], dim='time')
    elif ds_next is not None:
        combined_data = xr.concat([ds['radial_velocity'], ds_next['radial_velocity']], dim='time')
    else:
        combined_data = ds['radial_velocity']
    
    # Calculate the 10-minute running mean of the radial velocity at each range
    time_values = combined_data['time'].values
    radial_velocity_values = combined_data.values
    running_mean_10min_values = []

    for i in range(len(time_values)):
        start_time = time_values[i] - np.timedelta64(5, 'm')
        end_time = time_values[i] + np.timedelta64(5, 'm')
        mask = (time_values >= start_time) & (time_values <= end_time)
        running_mean_10min_values.append(np.mean(radial_velocity_values[mask], axis=0))

    running_mean_10min = xr.DataArray(
        data=np.array(running_mean_10min_values),
        dims=combined_data.dims,
        coords=combined_data.coords,
        name='radial_vel_mean_10min'
    )
        
    if previous_file is None:
        format_ts = np.datetime_as_string(ds['time'].values[0], unit='h')
        nanout = running_mean_10min.sel(time=slice(None, format_ts+':05:00'))
        running_mean_10min.loc[dict(time=nanout['time'])] = np.nan

    if next_file is None:
        format_ts = np.datetime_as_string(ds['time'].values[0], unit='h')
        nanout = running_mean_10min.sel(time=slice(format_ts+':55:00',None))
        running_mean_10min.loc[dict(time=nanout['time'])] = np.nan

    # Add the running means as new variables to the dataset
    ds['radial_vel_mean_10min'] = running_mean_10min.sel(time=ds['time'])
    ds['wprime'] = ds['radial_velocity'] - ds['radial_vel_mean_10min']

    # Save the modified dataset with the new variables to a separate file
    output_filename = os.path.join(output_dir, os.path.basename(name).replace('.cdf', '_with_running_means.cdf'))
    ds.to_netcdf(output_filename)

# ****************************************************************************

def lidarisraining(lidar,radar):
    """For each lidar profile, find the nearest KAZR profile and determine if it is raining."""

    # First, generate a flag for the radar data.
    # We will simply say that if >= 10 dBZ is found below 1500 m, then it's raining.

    # Maximum reflectivity at each time at <= 1500 m.
    maxdBZ = radar.reflectivity.T[radar.height<=1500].max(axis=0)
    radarflag = maxdBZ >= 10

    lidarflag = radarflag.sel(time=lidar.time, method='nearest')

    return lidarflag

# ****************************************************************************

def getBLdepthforlidar(lidar, heights_selected):
    # Get the time coordinates from lidar and heights_selected
    lidar_time = lidar.time
    heights_time = heights_selected.time

    # Find the nearest time index in heights_selected for each time in lidar
    nearest_indices = np.searchsorted(heights_time, lidar_time, side='left')

    # Handle edge cases where the nearest index is out of bounds
    nearest_indices = np.clip(nearest_indices, 0, len(heights_time) - 1)

    # Create the output DataArray with the same length as the time dimension in lidar
    lidar_pbl_height = xr.DataArray(
        heights_selected[nearest_indices].values,
        coords={"time": lidar_time},
        attrs={"long_name": "Boundary layer height nearest in time", "units": 'meters'},
        dims="time"
    )

    return lidar_pbl_height

# ****************************************************************************

# def process_lidar_time_sonde(t, lidar_time, lidar_ranges, sonde_launch_times, z, wind_speed, offset=3):
#     # Find all indices of sonde times within 3 hours of the lidar time
#     time_diff = np.abs(sonde_launch_times - np.datetime64(lidar_time.values))
#     valid_time_indices = np.where(time_diff <= np.timedelta64(offset, 'h'))[0]

#     if len(valid_time_indices) > 0:
#         # Vectorized operation to find the nearest sonde height for all lidar ranges
#         z_valid = z[valid_time_indices]
#         nearest_height_indices = np.abs(z_valid - lidar_ranges).argmin(axis=0)
#         # Assign the corresponding wind speeds to the lidar grid points
#         return wind_speed[valid_time_indices][nearest_height_indices].values
#     else:
#         return np.full(len(lidar_ranges), np.nan)

# ****************************************************************************

def process_lidar_time_lidar(t, lidar_time, lidar_ranges, wind_times, z, wind_speed, offset=3):
    # Find all indices of sonde times within 3 hours of the lidar time
    time_diff = np.abs(wind_times - np.datetime64(lidar_time.values))
    time_idx = np.argmin(time_diff.values)
    if time_diff[time_idx] <= np.timedelta64(offset, 'h'):
        valid_time_idx = time_idx
    else:
        valid_time_idx = None

    if valid_time_idx is not None:
        # Vectorized operation to find the nearest sonde height for all lidar ranges
        nearest_height_indices = np.abs(z - lidar_ranges).argmin(axis=0)
        # Assign the corresponding wind speeds to the lidar grid points
        return wind_speed[valid_time_idx][nearest_height_indices].values
    else:
        return np.full(len(lidar_ranges), np.nan)
    
# ****************************************************************************

# def getwindfromsonde(lidar, sondes, sonde_launch_times):
#     # Calculate wind speed from sonde data
#     wind_speed = np.sqrt(sondes.u_wind**2 + sondes.v_wind**2)

#     # Height of sonde above ground
#     z = sondes.alt - 1139

#     # Get lidar times and ranges
#     lidar_times = lidar.time
#     lidar_ranges = lidar.range

#     # Parallel processing for each lidar time
#     results = Parallel(n_jobs=120)(delayed(process_lidar_time_sonde)(t, lidar_time, lidar_ranges, sonde_launch_times, z, wind_speed) for t, lidar_time in enumerate(lidar_times))

#     # Combine results into a single array
#     # lidar_wind_speed = xr.concat(results,dim='time',coords='minimal')
#     lidar_wind_speed = np.vstack(results)

#     # Create the output DataArray with the same dimensions as the lidar time and range
#     lidar_wind_speed_da = xr.DataArray(
#         lidar_wind_speed,
#         coords={"time": lidar_times, "range": lidar_ranges},
#         attrs={"long_name": "Wind speed", "units": 'm/s'},
#         dims=["time", "range"]
#     )

#     return lidar_wind_speed_da

# ****************************************************************************

def getwindfromlidar(lidar,wind):
    # Calculate wind speed from sonde data
    wind_speed = wind.wind_speed
    wind_times = wind.time

    # Discard wind speed if error is too large (i.e., SNR is low).
    # pass

    # Height of sonde above ground
    z = wind.height

    # Get lidar times and ranges
    lidar_times = lidar.time
    lidar_ranges = lidar.range

    # Parallel processing for each lidar time
    results = Parallel(n_jobs=120)(delayed(process_lidar_time_lidar)(t, lidar_time, lidar_ranges, wind_times, z, wind_speed) for t, lidar_time in enumerate(lidar_times))

    # Combine results into a single array
    # lidar_wind_speed = xr.concat(results,dim='time',coords='minimal')
    lidar_wind_speed = np.vstack(results)

    # Create the output DataArray with the same dimensions as the lidar time and range
    lidar_wind_speed_da = xr.DataArray(
        lidar_wind_speed,
        coords={"time": lidar_times, "range": lidar_ranges},
        attrs={"long_name": "Wind speed", "units": 'm/s'},
        dims=["time", "range"]
    )

    return lidar_wind_speed_da

# ****************************************************************************

def definechords(smoothedV,lidar_pbl_height,height_km,time_hours,ws,threshold=0.5):

    # NOTE: Eventually, we want to get rid of eddies that butt up against NaNs in the 
    # velocity data record for any reason. Otherwise, these eddies could be larger 
    # than characterized.

    time_sec = time_hours * 3600

    # Create a mask for updrafts where smoothedV > 0.5
    updraft_mask = (smoothedV > threshold)

    # Label the contiguous updraft objects in 2D.
    # NOTE: These will be used as unique IDs to identify updraft objects in the saved output.
    updraft_labels, num = label(updraft_mask)
    
    # Initialize an array to store the size of updrafts along the time dimension at each height
    updraft_df = pd.DataFrame(columns=['Updraft ID','Center Time','Height','Chord Time','Wind Speed','PBL Height','Chord Length'])

    # Mask smoothedV so we can restrict analysis of eddies to lowest levels.
    mask = ~np.isnan(smoothedV)
    maxz = height_km[mask.any(axis=0).values].max()

    # Iterate over each height (range dimension)
    updraft_dicts = []
    for idh, height in enumerate(height_km):
        
        if height <= min(maxz,lidar_pbl_height.max()):

            upeddies = updraft_labels[:,idh]

            for eddy in np.unique(upeddies[upeddies >= 1])[1:]:

                # First, we need to throw out eddies if they are adjacent to invalid lidar data.
                # For example, if signal is too low or rain occurs, then we might be missing 
                # part of the eddy so we don't want to classify it based on incomplete information.

                t = time_sec[upeddies==eddy]
                dt = t.max() - t.min()

                if dt > 0:

                    idt = np.where(upeddies==eddy)[0]

                    # Next, deal with eddies at the start or end of a file by loading lidar data from previous
                    # and next times. Only characterize if center time is in this time window.

                    if idt.min() == 0: # Eddy at start time
                        # Grab data from previous file and determine whether the center time is in this file
                        # or in previous file. If in this file, then characterize and write.
                        
                        print('Eddy at start of file.')

                        # if np.isnan(smoothedV[idt.min()-1,idh].values) or np.isnan(smoothedV[idt.max(),idh].values):
                        #     continue

                        continue

                    elif idt.max() == smoothedV.shape[0]-1:   # Eddy at end time
                        # Grab data from next file and determine whether the center time is in this file
                        # or in next file. If in this file, then characterize and write.

                        print('Eddy at end of file.')

                        # if np.isnan(smoothedV[idt.min()-1,idh].values) or np.isnan(smoothedV[idt.max(),idh].values):
                        #     continue

                        continue

                    else:

                        center_time = int(np.median(t))
                        center_time_index = np.argmin(np.abs(time_sec-center_time))

                    # Check if the current height is below the lidar_pbl_height at each time
                    # Exclude single time observations.
                    if height*1000 <= lidar_pbl_height[center_time_index]:

                        updraft_dicts.append( {'Updraft ID': eddy,
                                        'Center Time': center_time,
                                        'Height': 1000*height.values,
                                        'Chord Time': dt,
                                        'Wind Speed': np.round(ws[center_time_index,idh],2),
                                        'PBL Height': lidar_pbl_height[center_time_index].values,
                                        'Chord Length': round(dt*ws[center_time_index,idh],2)
                                        } )

    # Concatenate the new row to the existing DataFrame
    updraft_df = pd.concat([updraft_df, pd.DataFrame(updraft_dicts)], ignore_index=True)

    return updraft_df

# ****************************************************************************

def concatenateday(dayfiles,kazrfile,pblhfile,windfiles,date_str):

    """
    
    Inputs:
    dayfiles: Lidar data
    kazrfile: Corresponding radar data
    pblhfile: Corresponding PBL height data
    sondefile: Corresponding rawinsonde data

    Outpus: 
    ???

    Key variables:
    smoothedV: The smoothed lidar Doppler velocity.
    lidar_pbl_height: PBL height at each lidar time.
    lidarflag: Rain flag based on KAZR data at each lidar time.
    wind: Wind speed at each lidar height and time derived from nearest sonde.

    """

    lidar = xr.open_mfdataset(dayfiles)
    time = lidar.variables['time'][:]
    range = lidar.variables['range'][:]
    velocity = lidar.variables['wprime'][:]
    intensity = lidar.variables['intensity'][:]

    time_hours = ((time-time[0]).values/1e9).astype('float')/3600 # assuming time is in seconds; adjust as needed
    height_km = range / 1000

    radar = xr.open_dataset(kazrfile)
    kazrtime = radar.variables['time'][:]
    
    # BUG: Doppler velocities appear to be biased. Need to compute a mean and remove this bias.
    # NOTE: For example, COR velocities skewed > 0, indicating upward motion everywhere all the time!


    # Smooth field in time
    smoothedV = deepcopy(velocity)
    smoothedV[:] = simplesmoother(velocity,sigma=1)


    # Get rain flag and NaN out raining lidar columns
    lidarflag = lidarisraining(lidar,radar)
    smoothedV[lidarflag==True,:] = np.nan
    
    # If signal is too weak, don't use the data.
    # NOTE: Intensity is SNR + 1, and documentation suggests throwing out data with SNR < 0.008. 
    smoothedV = smoothedV.where(intensity >= 1.008, np.nan)

    # Throw out data above 2 km.
    smoothedV[:,height_km>2] = np.nan

    # Retrieve boundary layer depth.
    # Load file
    pblh = xr.open_dataset(pblhfile)
    # Get quality indices
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

    # Now get the height at each lidar time.
    lidar_pbl_height = getBLdepthforlidar(lidar,heights_selected)

    # Retrieve low level winds from nearest-in-time rawinsonde.

    # # First. Need to get the launch times into 1D array that will match the sonde data 
    # # once opened.
    # # Extract sonde launch times from the launch_status attribute
    # sonde_launch_times = []
    # for sondefile in sondefiles:
    #     sonde = xr.open_dataset(sondefile)
    #     launch_status = sonde.attrs['launch_status']
    #     match = re.search(r'(\d{6}) (\d{4})', launch_status)
    #     if match:
    #         launch_time_str = match.group(1) + match.group(2)
    #         launch_time = np.datetime64(dte.datetime.strptime(launch_time_str, '%y%m%d%H%M'))
    #     else:
    #         launch_time = np.datetime64('NaT')  # Handle missing launch times

    #     # Repeat the launch time for the length of the time dimension in the sonde file
    #     time_length = len(sonde.time)
    #     sonde_launch_times.extend([launch_time] * time_length)

    # sonde_launch_times = np.array(sonde_launch_times)

    # # Now open the sonde data and get the wind.
    # sondes = xr.open_mfdataset(sondefiles)

    # Open the horizontal wind profiles from lidar data.
    wind = xr.open_mfdataset(windfiles)

    # Check if windspeed file exists. If not, calculate and write to disk.

    windspeed_dir = '/thumper/users/scott.powell/code/research-code/BNF/lidar/SGP/windspeed/'

    if os.path.exists(windspeed_dir + 'windspeed_' + date_str + '.cdf'):
        windspeed = xr.open_dataset(windspeed_dir + 'windspeed_' + date_str + '.cdf')
        ws = windspeed.__xarray_dataarray_variable__.values

    else:
        # windspeed = getwindfromsonde(lidar, sondes, sonde_launch_times)
        windspeed = getwindfromlidar(lidar, wind)

        # # Write windspeed to disk        
        encoding = {
                        'range': {
                            '_FillValue': -9999.0
                        }
                    }

        windspeed.to_netcdf(windspeed_dir + 'windspeed_' + date_str + '.cdf', encoding=encoding)
        ws = windspeed.values

    # Next, we need to identify the eddy updrafts from smoothedV.
    threshold = 0.5     # Threshold for upward motion.

    # Convert chord time to chord length.
    # NOTE: Throw out data if lidarflag indicates rain, if PBL height is not recorded, or
    # if wind speed is not recorded.
    updraft_df = definechords(smoothedV,lidar_pbl_height,height_km,time_hours,ws,threshold)

    # Store the updraft characteristics to disk.
    updraft_output_dir = '/thumper/users/scott.powell/code/research-code/BNF/lidar/SGP/updraft_output/'
    updraft_df.to_csv(updraft_output_dir+'updrafts_' + date_str + '.csv')

    # Make plot
    # fig = plt.figure(figsize=(10, 5))    

    # plt.pcolormesh(time_hours, height_km, 
    #         smoothedV.T, shading='auto', 
    #         cmap='bwr', vmin=-1, vmax=1,rasterized=True)

    # plt.ylim(0, 3)
    # plt.xlim(6, 9)
    # color_bar = plt.colorbar(label='Velocity (m/s)')
    # plt.xlabel(f'Time (hours after 00:00 UTC)')
    # plt.ylabel('Height (km)')

# ****************************************************************************

def process_lidar_file(i, name, names, output_dir):
    previous_file = names[i-1] if i > 0 else None
    next_file = names[i+1] if i < len(names)-1 else None
    velocity_running_mean(name, previous_file, next_file, output_dir)

# ****************************************************************************

def main():
    # Load the CDF file
    fdir = '/thumper/metdata/doppler-lidar/SGP/vert/'
    kazrdir = '/thumper/metdata/radar/KAZR_SGP/'
    pblhdir = '/thumper/metdata/PBLH/SGP/'
    # sondedir = '/thumper/metdata/sonde/CACTI/M1/'
    winddir = '/thumper/metdata/doppler-lidar/SGP/wind/'
    names = sorted(glob(fdir+"*.cdf"))
    kazrnames = sorted(glob(kazrdir+"*.nc"))
    pblhnames = sorted(glob(pblhdir+"sgpceil*"))
    # sondenames = sorted(glob(sondedir+"*.cdf"))
    windnames = sorted(glob(winddir+"*.nc"))

    # Get all the dates in the Lidar and KAZR data.
    lidardates = np.unique([re.search(r'\.(\d{8})\.\d{6}\.cdf', fname).group(1) for fname in names])
    dates = np.unique([re.search(r'\.(\d{8})\.\d{6}\.nc', fname).group(1) for fname in kazrnames])

    # Calculate the 10-minute running mean Doppler velocity as function of height from lidar data.
    output_dir = '/thumper/metdata/doppler-lidar/SGP_withrunningmeans/'
    # Parallel(n_jobs=120)(delayed(process_lidar_file)(i, name, names, output_dir) for i, name in enumerate(names))

    # Reset names to the new lidar files with running means.
    names = sorted(glob(output_dir+"*.cdf"))

    # # Concatenate all the data for each date
    for date_str in lidardates:

        # Do a check for lidar data.
        if date_str in dates:
        
            try:

                print('Now processing ' + date_str)

                date = dte.datetime.strptime(date_str, '%Y%m%d')
                
                # Get the date strings for the day before and the day after
                day_before = get_date_str(date - dte.timedelta(days=1))
                day_after = get_date_str(date + dte.timedelta(days=1))
                
                dayfiles = [i for i in names if date_str in i]
                kazrfile = [i for i in kazrnames if date_str in i][0]
                pblhfile = [i for i in pblhnames if date_str in i][0]

                # # Get sonde files for the specified date, the day before, and the day after
                # sondefiles = [i for i in sondenames if date_str in i or day_before in i or day_after in i]

                # Get lidar wind files for the specified date, the day before, and the day after
                windfiles = [i for i in windnames if date_str in i or day_before in i or day_after in i]


                concatenateday(dayfiles,kazrfile,pblhfile,windfiles,date_str)

            except:
                print('Something went wrong in processing ' + date_str)

            finally:
                print('Processing ' + date_str + ' complete!')

        else:
            print('No KAZR data available to scan for precipitation. Skipping ' + date_str)

if __name__ == "__main__":
    main()