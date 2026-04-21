def define500chords(resample500_level,lidar_pbl_height,wvarmean,wvarmed,time_hours,ws500,key,threshold=0.5):

    import numpy as np
    import pandas as pd
    from scipy.ndimage import label

    # NOTE: Eventually, we want to get rid of eddies that butt up against NaNs in the 
    # velocity data record for any reason. Otherwise, these eddies could be larger 
    # than characterized.

    wprime = resample500_level.values

    time_sec = time_hours * 3600

    # Create a mask for updrafts where wprime > 0.5
    updraft_mask = (wprime > threshold)

    # Label eddy objects
    upeddies, num = label(updraft_mask)


    # Classify boundary layer type.
    # Replace None with np.nan
    wvarmean = np.where(wvarmean == None, np.nan, wvarmean)
    wvarmed = np.where(wvarmed == None, np.nan, wvarmed)
    # Get the type.
    BLtype = np.full(len(wvarmean),'Unclassified')
    BLtype[(wvarmean >= 0.2) * (wvarmed >= 0.2)] = 'CBL'
    BLtype[(wvarmean <= 0.1) * (wvarmed <= 0.1)] = 'Stable'

    # Initialize an array to store the size of updrafts along the time dimension at each height
    # updraft_df = pd.DataFrame(columns=['Updraft ID','Center Time','Height','Chord Time','Wind Speed','PBL Height','Chord Length'])

    if lidar_pbl_height.max() >= 500:

        updraft_dicts = []

        for eddy in np.unique(upeddies[upeddies >= 1])[1:]:

            t = time_sec[upeddies==eddy]
            dt = t.max() - t.min()

            if dt > 0:
                
                idt = np.where(upeddies==eddy)[0]

                # Next, deal with eddies at the start or end of a file by loading lidar data from previous
                # and next times. Only characterize if center time is in this time window.

                if idt.min() == 0: # Eddy at start time
                    # Grab data from previous file and determine whether the center time is in this file
                    # or in previous file. If in this file, then characterize and write.
                    # NOTE: Code currently just skips over this eddy without loading other data.

                    print('Eddy at start of file.')

                    continue

                elif idt.max() == wprime.shape[0]-1:   # Eddy at end time
                    # Grab data from next file and determine whether the center time is in this file
                    # or in next file. If in this file, then characterize and write.
                    # NOTE: Code currently just skips over this eddy without loading other data.

                    print('Eddy at end of file.')

                    continue

                # Proceed to find its duration and chord length.
                
                else:

                    center_time = int(np.median(t))
                    center_time_index = np.argmin(np.abs(time_sec-center_time))

                    # Check if the current height is below the lidar_pbl_height at each time
                    # Exclude single time observations.
                    if 495 <= lidar_pbl_height[center_time_index]:
    
                        # We need to record if eddies are adjacent to invalid lidar data.
                        # For example, if signal is too low or rain occurs, then we might be missing 
                        # part of the eddy so we don't want to classify it based on incomplete information.
                        cond = np.isnan(wprime[idt.min()-1]) or np.isnan(wprime[idt.max()+1]) 
                        
                        updraft_dicts.append( {'Updraft ID': eddy,
                                        'Center Time': center_time,
                                        'Chord Time': dt,
                                        'Height': float(key),
                                        'Wind Speed': np.round(ws500[center_time_index],2),
                                        'PBL Height': lidar_pbl_height[center_time_index].values,
                                        'BL Type': BLtype[center_time_index],
                                        'Chord Length': np.round(dt*ws500[center_time_index],2),
                                        'Invalid Adjacent': True if cond else False
                                        } )


    updraft_df = pd.DataFrame(updraft_dicts)

    # Concatenate the new row to the existing DataFrame
    # updraft_df = pd.concat([updraft_df, pd.DataFrame(updraft_dicts)], ignore_index=True)

    return updraft_df
