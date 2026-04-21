#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 5 15:46:17 2024

@author: ivan.ariashernandez

This file contains fuctions to analyze observations from the BNF site
"""

import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import glob
import xarray as xr

def plot_backscatter(file):
    dataset = nc.Dataset(file, 'r')
    fig = plt.figure(figsize=(12, 10))
    time = dataset.variables['time'][:]
    range = dataset.variables['range'][:]
    backscatter = dataset.variables['attenuated_backscatter'][:]
    backscatter_dB = 10*np.log10(backscatter)
    backscatter_dB = np.ma.masked_less(backscatter_dB, -90)
    
    time_hours = time / 3600  # assuming time is in seconds; adjust as needed
    height_km = range/1000 
    plt.figure(figsize=(10, 6))

    plt.pcolormesh(time_hours, height_km, 
               backscatter_dB.T, shading='auto', cmap="viridis")
    plt.ylim(0, 6)
    plt.colorbar(label='Backscatter Intensity (dB)')
    plt.xlabel('Time (hours after 2024/11/01 00:00 UTC)')
    plt.ylabel('Height (km)')
    plt.title('LiDAR Attenuated Backscatter')
    
    return fig

def plot_backscatter_day(input_path, day):
    day_str = str(day)
    files = [f for f in glob.glob(input_path +f'/*{day_str}*.cdf', recursive=True)]
    files.sort()

    ds = xr.open_mfdataset(files)
    time = ds.variables['time'][:]
    range = ds.variables['range'][:]
    backscatter = ds.variables['attenuated_backscatter'][:]
    backscatter_dB = 10*np.log10(backscatter)
    # backscatter_dB = np.ma.masked_less(backscatter_dB, -90)
        
    time_hours = ((time-time[0]).values/1e9).astype('float')/3600 # assuming time is in seconds; adjust as needed
    height_km = range / 1000
    
    fig = plt.figure(figsize=(10, 5))    
    
    plt.pcolormesh(time_hours, height_km, 
            backscatter_dB.T, shading='auto', cmap="viridis")

    plt.ylim(0, 3)
    plt.xlim(0, 1)
    plt.colorbar(label='Backscatter Intensity (dB)')
    plt.xlabel(f'Time (hours after {day_str} 00:00 UTC)')
    plt.ylabel('Height (km)')
    plt.title('LiDAR Attenuated Backscatter')
    
    return fig

def simplesmoother(input2D,sigma=1):
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(input2D, sigma=sigma, axis=1)


def plot_velocity_day(input_path, day):
    day_str = str(day)
    files = [f for f in glob.glob(input_path +f'/*{day_str}*.*.cdf', recursive=True)]
    files.sort()

    ds = xr.open_mfdataset(files)
    time = ds.variables['time'][:]
    range = ds.variables['range'][:]
    velocity = ds.variables['radial_velocity'][:].T

    time_hours = ((time-time[0]).values/1e9).astype('float')/3600 # assuming time is in seconds; adjust as needed
    height_km = range / 1000

    # Smooth field in time
    smoothedV = simplesmoother(velocity,sigma=1)

    # Make plot
    fig = plt.figure(figsize=(10, 5))    

    plt.pcolormesh(time_hours, height_km, 
            smoothedV, shading='auto', 
            cmap='bwr', vmin=-1, vmax=1,rasterized=True)

    plt.ylim(0, 1.5)
    plt.xlim(0, 3)
    color_bar = plt.colorbar(label='Velocity (m/s)')
    plt.xlabel(f'Time (hours after {day_str} 00:00 UTC)')
    plt.ylabel('Height (km)')
    # plt.title('LiDAR Radial Velocity')
    # fig.savefig('lidar_1hour.pdf')

    return fig

def plot_velocity(file):
    dataset = nc.Dataset(file, 'r')
    fig = plt.figure(figsize=(12, 10))
    time = dataset.variables['time'][:]
    range = dataset.variables['range'][:]
    velocity = dataset.variables['radial_velocity'][:]    
    time_hours = time / 3600  # assuming time is in seconds; adjust as needed
    height_km = range/1000 
    
    # plt.figure(figsize=(10, 6))
    plt.pcolormesh(time_hours, height_km, 
               velocity.T, shading='auto', cmap='bwr')
    plt.ylim(0, 6)
    plt.colorbar(label='Velocity (m/s)')
    plt.xlabel('Time (hours after 2024/11/01 00:00 UTC)')
    plt.ylabel('Height (km)')
    # plt.title('LiDAR Radial Velocity')
    
    return fig

def main():
    # Load the CDF file
    file = '/thumper/metdata/doppler-lidar/COR/cordlfptM1.b1.20190111.200055.cdf'
    ds = nc.Dataset(file, 'r')

    # Check available variables in the dataset
    print(ds.variables.keys())

    input_path = '/thumper/metdata/doppler-lidar/COR/'
    day = 20190222
    fig = plot_velocity_day(input_path, day)
q
    # fig = plot_velocity(file)


if __name__ == "__main__":
    main()

