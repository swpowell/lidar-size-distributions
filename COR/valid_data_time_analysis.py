import xarray as xr
import numpy as np
from scipy.ndimage import label
from glob import glob
from matplotlib import pyplot as plt
from joblib import Parallel, delayed


# Calculate the 10-minute running mean Doppler velocity as function of height from lidar data.
output_dir = '/thumper/metdata/doppler-lidar/COR_withrunningmeans/'
# Parallel(n_jobs=120)(delayed(process_lidar_file)(i, name, names, output_dir) for i, name in enumerate(names))

# Reset names to the new lidar files with running means.
names = sorted(glob(output_dir+"*.cdf"))

monnames = [i for i in names if '2019030' in i]
lidar = xr.open_mfdataset(monnames)

z = lidar.range.values

def calcsizes(i, lidar):

    size = []
    vals = lidar.intensity[:,i].values
    
    valid = [1 if j >= 1.008 else 0 for j in vals]
    labels, num = label(valid)
    for lbl in np.unique(labels)[1:]:
        S = labels[labels == lbl].size
        if S > 1:
            size.append(S)
    sizes = {}
    sizes[str(i)] = size

    return sizes

results = Parallel(n_jobs=50)(delayed(calcsizes)(i, lidar) for i in range(1, 51))

def combine_dicts(dict_list):
    combined_dict = {}
    for d in dict_list:
        combined_dict.update(d)
    return combined_dict

# Combine the dictionaries
sizes = combine_dicts(results)

# Now let's analyze this stuff.
# H = {}
# for k in sizes.keys():
#     H[str(k)], bins = np.histogram(sizes[str(k)],bins=range(2,82),density=True)

# fig, ax = plt.subplots(1,1,figsize=(6,6))
# ax.plot(H['5'],'b')
# ax.plot(H['25'],'r')
# ax.plot(H['45'],'k')

pct95 = []
for i in np.arange(1,51):
    pct95.append(np.nanpercentile(sizes[str(i)],95))

plt.plot(z[1:51],pct95)


## Next, let's try resampling at 500 m using SNR from other heights.

def resample500(lev, lidar, wprime_500, inten_500):

    sizes = {}
    vals = lidar.intensity[:,lev].values
    
    wprime_500[vals < 1.008] = np.nan
    wprime_500[inten_500 < 1.008] = np.nan
    
    cond = [1 if i > 0.5 else 0 for i in wprime_500]

    labels, num = label(cond)
    sizes[str(lev)] = [labels[labels==i].size for i in np.unique(labels)[1:]]
    sizes[str(lev)] = [i for i in sizes[str(lev)] if i > 1]

    return sizes


wprime_500 = lidar.wprime[:,16].values
inten_500 = lidar.intensity[:,16].values
output = Parallel(n_jobs=50)(delayed(resample500)(i, lidar, wprime_500, inten_500) for i in range(1, 51))
