import os
import pandas as pd
import datetime as dte
import re
import numpy as np
import concurrent.futures

def add_dt(updraft_output_dir,date_str):

    if os.path.exists(updraft_output_dir+'updrafts_' + date_str + '.csv'):
        # Open them and add the date and time to the appropriate file.
        updraft_df = pd.read_csv(updraft_output_dir+'updrafts_' + date_str + '.csv')
        updraft_df.drop('Unnamed: 0',axis=1,inplace=True)

        # Convert the date string to a datetime object
        base_date = dte.datetime.strptime(date_str, '%Y%m%d')

        # Add the date and time to the files.
        updraft_df['Center Datetime'] = updraft_df['Center Time'].apply(lambda x: base_date + dte.timedelta(seconds=x))
        updraft_df.set_index('Center Datetime',inplace=True)

        updraft_df.to_csv(updraft_output_dir+'updrafts_' + date_str + '.csv')


if '__name__' == '__main__':

    # Where are the files?
    updraft_output_dir = '/thumper/users/scott.powell/code-data/research-code/lidar/SGP/updraft_output/'

    # List the files
    updraft_files = [f for f in os.listdir(updraft_output_dir) if os.path.isfile(os.path.join(updraft_output_dir, f))]

    # Only process the dates that are present in both directories (they should all overlap)
    # Extract dates using regex and get unique values
    updraft_dates = {re.search(r'\d{8}', name).group() for name in updraft_files if re.search(r'\d{8}', name)}

    # Assuming date_strings is your list of dates
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(add_dt, updraft_output_dir, date) for date in updraft_dates]
        concurrent.futures.wait(futures)
