# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 08:26:55 2023

@author: jcard
"""
import pandas as pd
import numpy as np 
import cv2
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
from plantcv.plantcv.visualize import histogram

def nothing(x):
    pass

path = 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/thermal_img/data_trays/VF_2_lettuce/May-16-2023_1055_thermography-E2929D780611.csv'

data = pd.read_csv(path ,delimiter=" ", header=None)
data.to_csv("thermal_data.csv",header=None,index=None)


# def analyse_thermal(csv_img,) We will define a function after we are sure it runs 

thermal_data,_,_ = pcv.readimage(filename= "thermal_data.csv" , mode='csv')
pcv.plot_image(thermal_data)
# print(thermal_data.min())
# print(thermal_data.max())
# print(thermal_data.mean())


histogram(thermal_data, hist_data=True, bins = 25,title="Temperature Distribution")


# Rescale the thermal data to a colorspace with range 0-255 rather than raw data 
    
# Inputs:
#   gray_img - Grayscale image data 
#   min_value - New minimum value for range of interest. default = 0
#   max_value - New maximum value for range of interest. default = 255
scaled_thermal_img = pcv.transform.rescale(gray_img=thermal_data)
pcv.plot_image(scaled_thermal_img)


l_h, u_h = 0,0
#if img.shape[0] < img.shape[1]:
    #img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
cap = thermal_data
frame = cap


# Create a window named trackbars.
cv2.namedWindow("Trackbars")

# Now create 6 trackbars that will control the lower and upper range of
# H,S and V channels. The Arguments are like this: Name of trackbar,
# window name, range,callback function. For Hue the range is 0-179 and
# for S,V its 0-255.
cv2.createTrackbar("L - H", "Trackbars", l_h, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", u_h, 255, nothing)


while True:


    # Convert the BGR image to HSV image.
    hsv = scaled_thermal_img

    # Get the new values of the trackbar in real time as the user changes
    # them
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")

    # Set the lower and upper HSV range according to the value selected
    # by the trackbar
    lower_range = np.array(l_h)
    upper_range = np.array(u_h)

    # Filter the image and get the binary mask, where white represents
    # your target color
    mask = cv2.inRange(hsv, lower_range, upper_range)
    cont, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
 

    # You can also visualize the real part of the target color (Optional)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Converting the binary mask to 3 channel image, this is just so
    # we can stack it with the others
    mask_3 = thermal_data

    # stack the mask, orginal frame and the filtered result
    stacked = np.hstack((mask_3, frame, res))

    # Show this stacked frame at 40% of the size.
    cv2.imshow('Trackbars', cv2.resize(stacked, None, fx=0.5, fy=0.5))

    # If the user presses ESC then exit the program
    key = cv2.waitKey(1)
    if key == 27:
        break

    # If the user presses `s` then print this array.
    if key == ord('s'):
        thearray = [[l_h], [u_h]]
        print(thearray)

        # Also save this array as penval.npy
        #np.save('hsv_value', thearray)
        break


# Release the camera & destroy the windows.
cv2.destroyAllWindows()


# Threshold the thermal data to make a binary mask
    
# Inputs:
#   gray_img - Grayscale image data 
#   threshold- Threshold value (between 0-255)
#   max_value - Value to apply above threshold (255 = white) 
#   object_type - 'light' (default) or 'dark'. If the object is lighter than the background then standard 
#                 threshold is done. If the object is darker than the background then inverse thresholding is done. 
#bin_mask = pcv.threshold.binary(gray_img=thermal_data, threshold=thermal_data.mean(), max_value=255, object_type='dark')
#pcv.plot_image(bin_mask)


pseudo_img = pcv.visualize.pseudocolor(gray_img = thermal_data, mask=mask, cmap='jet', 
                                       min_value=22, max_value= 26)

pseudo_img

masked_thermal = thermal_data[np.where(mask > 0)]
maxtemp = np.amax(masked_thermal)
mintemp = np.amin(masked_thermal)
avgtemp = np.average(masked_thermal)
mediantemp = np.median(masked_thermal)

histogram(thermal_data, mask=mask, hist_data=True, bins = 25)
#pcv.print_image(pseudo_img, "thermal_06-55.jpg")
