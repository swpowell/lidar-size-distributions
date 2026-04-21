import pandas as pd
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import expon


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
fdir = '/thumper/users/scott.powell/code/research-code/BNF/lidar/COR/updraft_output_resample/'
fnames = sorted(glob(fdir+"*.csv"))

# Read each CSV file and concatenate them into a single DataFrame
df_list = [pd.read_csv(fname) for fname in fnames]
combined_df = pd.concat(df_list, ignore_index=True).drop('Unnamed: 0', axis=1)

in_pbl = combined_df[combined_df['PBL Height']>=495]
pct95 = in_pbl[['Chord Length','Height']].groupby('Height').quantile(.95)
pct95['Chord Length'].plot()


pct95 = combined_df.groupby('Height').quantile(.95)





step = 100
chord_length_hist = {}
datastore = {}
# means = []
pct95 = []
for pblh in np.arange(300,2600,step):
    cond = (combined_df['PBL Height'] > pblh) * (combined_df['PBL Height'] <= pblh + step)
    subset = combined_df[cond]
    datastore[str(pblh)] = subset
    chord_length_hist[str(pblh)], bins = np.histogram(subset['Chord Length'],density=True,bins=np.arange(10,201,10))
    # means.append(bootstrap(subset['Chord Length']))
    pct95.append(np.nanpercentile(subset['Chord Length'], 95))

# Characterize size by height for a given PBLH.
hgtbins = np.arange(0,1001,50)
for key in datastore:

    subset = datastore[key][datastore[key]['Invalid Adjacent'] == False]

    chord, maxchord = [], []
    for hgt in hgtbins:
        cond = (subset['Height']>hgt) * (subset['Height'] <= hgt+step)
        chord.append(np.nanpercentile(subset[cond]['Chord Time'],95))
        maxchord.append(np.max(subset[cond]['Chord Length']))
    
chord, maxchord = [], []
for hgt in hgtbins:
    cond = (combined_df['Height']>hgt) * (combined_df['Height'] <= hgt+step)
    chord.append(np.nanpercentile(combined_df[cond]['Chord Time'],95))


# fig, ax = plt.subplots(1,1,figsize=(6,6))
# # ax.plot(chord_length_hist['300'],'b')
# ax.plot(chord_length_hist['600'],'g')
# ax.plot(chord_length_hist['900'],'m')
# ax.plot(chord_length_hist['1200'],'k')
# ax.plot(chord_length_hist['1500'],'r')



# Function to convert chord length distribution to radii distribution
def chord_to_radii(chord_lengths, scale):
    # Define the PDF of the radii (exponential distribution)
    def pdf_radii(r, scale):
        return expon.pdf(r, scale=scale)
    
    # Define the PDF of the chord lengths given the radii
    def pdf_chord_given_radii(l, r):
        return l / (r * np.sqrt(r**2 - (l/2)**2))
    
    # Define the joint PDF of chord lengths and radii
    def joint_pdf(l, r, scale):
        return pdf_chord_given_radii(l, r) * pdf_radii(r, scale)
    
    # Define the marginal PDF of chord lengths
    def marginal_pdf_chord(l, scale):
        r_values = np.linspace(l/2, 10, 1000)  # Adjust the range as needed
        integral = np.array([np.trapz(joint_pdf(l, r_values, scale), r_values) for l in chord_lengths])
        return integral
    
    # Calculate the marginal PDF of chord lengths
    marginal_pdf = marginal_pdf_chord(chord_lengths, scale)
    
    # Normalize the marginal PDF
    marginal_pdf /= np.trapz(marginal_pdf, chord_lengths)
    
    return marginal_pdf

# Convert chord length distribution to radii distribution
mydata = datastore['1500']['Chord Length']
scale = np.pi/4 * mydata.mean() # Scale parameter for the exponential distribution
radii_distribution = chord_to_radii(mydata, scale)




A = combined_df_all.groupby('Height').quantile(.5)['Chord Length']
B = combined_df_500.groupby('Height').quantile(.5)['Chord Length'][0.045:]
fig, ax = plt.subplots(1,1,figsize=(6,6))
ax.plot(A.index,A,'r')
ax.plot(A.index,B,'b')


fdir = '/thumper/users/scott.powell/code/research-code/BNF/lidar/COR/updraft_output/'
rdir = '/thumper/users/scott.powell/code/research-code/BNF/lidar/COR/updraft_output_resample/'

fname = fdir + 'updrafts_20181111.csv'
rname = rdir + 'updrafts_resample_20181111.csv'

pdf = pd.read_csv(fname)
pdr = pd.read_csv(rname)

pdf.groupby('Height')['Chord Length'].quantile(.95)
pdr.groupby('Height')['Chord Length'].quantile(.95)



cond = (~np.isnan(combined_df_reg['PBL Height']))*(combined_df_reg['Height']<=combined_df_reg['PBL Height'])*(combined_df_reg['PBL Height'] == 1040.0)
condr = (~np.isnan(combined_df_resam['PBL Height']))*(combined_df_resam['Height']*1000<=combined_df_resam['PBL Height'])*(combined_df_resam['Height']>0.03)*(combined_df_resam['PBL Height']==1240.0)

A1 = combined_df_resam[condr].groupby('Height').quantile(.95)['Chord Length']
A2 = combined_df_reg[cond].groupby('Height').quantile(.95)['Chord Length']

fig, ax = plt.subplots(1,1,figsize=(6,6))
ax.plot(A2.index,A1,'r',label='Resampled 500')
ax.plot(A2.index,A2,'b',label='By Height')
h,l = ax.get_legend_handles_labels()
ax.legend(h,l)