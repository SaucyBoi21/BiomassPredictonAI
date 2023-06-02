# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:45:39 2023
# Replace Random Forest Segmentation for Threshold segmentation temporarily. 
# Keep a file with the raft reference taken at the beginning of the experiment. 

@author: jsc00615
"""

import os

import matplotlib.pyplot as plt
import cv2
import pandas as pd
import glob
import numpy as np
import plotly.express as px
from plantcv import plantcv as pcv
from scipy import stats

# A required callback method that goes into the trackbar function.
def nothing(x):
    pass
np.set_printoptions(threshold=np.inf)

# give img location use "/" instead of "\"
path = 'RGB/T01_GH13_JC01_Jun-01-2023_1017_rgb.jpg'
#path_imgs = 'C:/Users/jcard/OneDrive - University of Georgia/side_projects/seedlings_Ferrarezi_Lab/2nd_trial/original_imgs/cropped/final_img'
#path_result = 'C:/Users/jcard/OneDrive - University of Georgia/side_projects/seedlings_Ferrarezi_Lab/2nd_trial/original_imgs/cropped/results_csv'

#DEBUG = False

def get_centeroid(cnt):
    length = len(cnt)
    sum_x = np.sum(cnt[..., 0])
    sum_y = np.sum(cnt[..., 1])
    return (int(sum_x / length), int(sum_y / length))

def get_y(coord):
    c_no,centerx,centery = coord
    return centerx

def generate_border(image, border_size=5, n_erosions=1):
    erosion_kernel = np.ones((3, 3), np.uint8)  ## Start by eroding edge pixels
    eroded_image = cv2.erode(image, erosion_kernel, iterations=n_erosions)

    ## Define the kernel size for dilation based on the desired border size (Add 1 to keep it odd)
    kernel_size = 2 * border_size + 1
    dilation_kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Kernel to be used for dilation
    dilated = cv2.dilate(eroded_image, dilation_kernel, iterations=1)
    # plt.imshow(dilated, cmap='gray')

    ## Replace 255 values to 127 for all pixels. Eventually we will only define border pixels with this value
    dilated_127 = np.where(dilated == 255, 127, dilated)

    # In the above dilated image, convert the eroded object parts to pixel value 255
    # What's remaining with a value of 127 would be the boundary pixels.
    original_with_border = np.where(eroded_image > 127, 255, dilated_127)

    # plt.imshow(original_with_border,cmap='gray')

    return original_with_border



# tray_number = 1
# pcv.params.debug = "print"
# Let's develop the code for a single image first
# PCV is reading this img as BGR instead of RGB
img,_,_ = pcv.readimage(filename=path, mode="native")
pcv.plot_image(img)

#rot_img = pcv.rotate(img,90, False)
#pcv.plot_image(rot_img)


l_h, l_s, l_v, u_h, u_s, u_v,size_ig = 0,0,0,255,255,255,0
#if img.shape[0] < img.shape[1]:
    #img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
cap = img
frame = cap


# Create a window named trackbars.
cv2.namedWindow("Trackbars")

# Now create 6 trackbars that will control the lower and upper range of
# H,S and V channels. The Arguments are like this: Name of trackbar,
# window name, range,callback function. For Hue the range is 0-179 and
# for S,V its 0-255.
cv2.createTrackbar("L - H", "Trackbars", l_h, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", l_s, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", l_v, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", u_h, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", u_s, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", u_v, 255, nothing)
cv2.createTrackbar("Size", "Trackbars", size_ig, 1000, nothing)

while True:


    # Convert the BGR image to HSV image.
    hsv = cv2.cvtColor(cap, cv2.COLOR_BGR2LAB)

    # Get the new values of the trackbar in real time as the user changes
    # them
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    size_ig = cv2.getTrackbarPos("Size", "Trackbars")

    # Set the lower and upper HSV range according to the value selected
    # by the trackbar
    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])

    # Filter the image and get the binary mask, where white represents
    # your target color
    mask = cv2.inRange(hsv, lower_range, upper_range)
    cont, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in cont:
        if cv2.contourArea(contour) < size_ig:
            cv2.fillPoly(mask, pts=[contour], color=(0, 0, 0))

    # You can also visualize the real part of the target color (Optional)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Converting the binary mask to 3 channel image, this is just so
    # we can stack it with the others
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # stack the mask, orginal frame and the filtered result
    stacked = np.hstack((mask_3, frame, res))

    # Show this stacked frame at 40% of the size.
    cv2.imshow('Trackbars', cv2.resize(stacked, None, fx=0.3, fy=0.3))

    # If the user presses ESC then exit the program
    key = cv2.waitKey(1)
    if key == 27:
        break

    # If the user presses `s` then print this array.
    if key == ord('s'):
        thearray = [[l_h, l_s, l_v], [u_h, u_s, u_v]]
        print(thearray)

        # Also save this array as penval.npy
        #np.save('hsv_value', thearray)
        break


# Release the camera & destroy the windows.
cv2.destroyAllWindows()

# =============================================================================
#  THIS IS PLANTCV histogram option. Creates a ggplot object. 

# =============================================================================
pcv.plot_image(mask)
pcv.print_image(mask,"test_imgs/mask_reference.jpg")

#photos = glob.glob(path_images)
#img = cv2.imread(photos[0])
#flat_img = feature_extraction(img)
'_'.join(path.split("_")[2:6])
depth_img = np.array(pd.read_csv('DEPTH_CSV/T01_GH13_JC01_Jun-01-2023_1017_depth_values.csv'))
    #mask = cv2.imread(r"height_harvest/"+file.split("\\")[-1])
pcv.plot_image(depth_img,cmap="jet")

#Keeping height values according to the mask. 
masked_depth = depth_img[np.where(mask > 0)]
stats.describe(masked_depth)
round(masked_depth.mean(),2)

stats.describe(masked_depth)
height_mean = round(masked_depth.mean(),2)
mode = stats.mode(masked_depth)
height_mode = mode[0].max()


# Show distribution of distance (in mm) across the raft
size =12
fig,(ax1,ax2) = plt.subplots(nrows =2,ncols=1,figsize=(5,5))
ax1.set_title('Original Depth Image (csv)',fontsize = size)
ax1.set_axis_off()
ax1.imshow(depth_img)
ax2.hist(masked_depth,bins = np.arange(1000,1250, 5))
ax2.set_title('Single Plant Heights Distribution',fontsize = size)
plt.show()

# Obtaining the minimum values of the array
# sorting the array
arr1 = np.sort(masked_depth)
  
# k smallest number of array
print(4, "smallest elements of the array")
print(arr1[:500])

# So far we have the mask ready, as well as the depth values. Now we need to put that mask on top of the 








 

