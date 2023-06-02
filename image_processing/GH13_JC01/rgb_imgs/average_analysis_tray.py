# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 12:19:50 2023

Display graphs with the files available at results folder. 
Value in the timelapse would be a mean  of 55 plants, it will be
a pixel value per tray. 
@author: jcard
"""

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import glob

# Step1: Specify directories where to take files from.
# We are storing all file names extracted using glob.glob function. 
print(sorted(glob.glob('C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/GH13_JC01/rgb_imgs/tray_1/results_csv/T01*.csv')))
filenames = sorted(glob.glob('C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/GH13_JC01/rgb_imgs/tray_1/results_csv/T01*.csv'))
filenames =  filenames[0:3]


# Since we are mixing numbers and strings in this csv file makes sense to use pandas instead of numpy
data_1 = filenames[1]
data_2 =  pd.read_csv('C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/GH13_JC01/rgb_imgs/tray_1/results_csv/T01_GH13_JC01_Jan-30-2023_1641_rgb_data.csv')

#just select plant area
save = data_2.iloc[:,3:9]

plant_area =  data_2.iloc[:,[3]]
plant_area.plot.scatter()

save.mean()


fig = plt.figure(figsize=(10.0,3.0))
                
axes_1 = fig.add_subplot(1,3,1)
axes_2 = fig.add_subplot(1,3,2)

axes_1.set_ylabel("Area (px) Average")
axes_1.hist(plant_area)
    
axes_2.set_ylabel("Area (px) Median")
axes_2.boxplot(plant_area)


# Access each csv file, calculate the mean for certain variables, put all those means together into a single graph. 
# Want we want to do is to calculate first the mean values for each numerical variable in the csv file. 
# Display a timeseries graph 

# Step 2: Loop through filenames

for filename in filenames:
    print(filename)
    data = np.loadtxt(fname=filename,delimiter=',')
    fig = plt.figure(figsize=(10.0,3.0))
    
    axes1 = fig.add_subplot(1,3,1)
    axes2 = 