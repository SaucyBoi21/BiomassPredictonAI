# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 15:30:01 2023
Bring together all csv files created as a result of the image analysis.
Create a data frame that puts together all files. 
@author: jcard
"""

import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

path_database = 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/GH13_JC01/rgb_imgs/tray_1/timeseries_rgb'
path = r'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/GH13_JC01/rgb_imgs/tray_1/results_csv' # use your path
all_files = glob.glob(os.path.join(path , "T01*.csv"))


li = []

# Here I'm reading each csv as dataframe, then in each csv file 
# I'm adding a new column with the file's date by spliting the files name stored as name. 

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    name = os.path.basename(filename)
    df["date"] = name.split('_',4)[3]
    first_column = df.pop('date')
    df.insert(0, 'Date', first_column)
    li.append(df)

# Here I'm creating a frame that includes all csv files 
# we will use this file to create the time lapse visualizations. 
frame = pd.concat(li, axis=0, ignore_index=True)
# Let's make sure 'date' is actually a date in pandas
frame["Date"] = pd.to_datetime(frame["Date"])

frame.to_csv(os.path.join(path_database,'tray1_database.csv'))

# What I need now is to display a graph with the average of all plants 
# in a tray per day. 

new_frame = frame.groupby(frame['Date']).mean()

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(new_frame["Date"], new_frame["plant_area"])


