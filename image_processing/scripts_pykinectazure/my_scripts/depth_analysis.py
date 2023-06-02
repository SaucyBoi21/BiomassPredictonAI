# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 08:37:19 2022
Retrieving back a transformed image stored as csv.

Aim: Upload back to variable explorer the transformation numpy array
stored as a csv file. The uploaded file must have the same format:
    
uint16 
size = 1920 x 1080
@author: jcard
"""

# Use np.genfromtxt function from numpy. Make sure you specify the format for the values. In this case UINT16
from skimage import measure
from skimage.measure import label, regionprops
import sys
import cv2
import time
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpllimg
from plantcv import plantcv as pcv
from skimage.color import label2rgb,rgb2gray

transformed_img = np.genfromtxt("C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/scripts_pykinectazure/my_scripts/transf_imgs/2lettuceNov-22-2022_1616rgbd_values_comparison.csv",
                    delimiter=",", dtype='uint16')

plt.imshow(transformed_img, cmap = 'gray')
plt.savefig('gray_scale_depth.png',dpi = 360)

# Aditionally, we can add the original RGB to compare 
rgb_img = mpllimg.imread('C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/scripts_pykinectazure/my_scripts/rgb_imgs/2lettuceNov-22-2022_1616_rgb.jpg')
plt.imshow(rgb_img)


# Now we can include a comparison of both images before going forward with the depth analysis 

# code for displaying multiple images in one figure
  
# create figure
comparison_fig = plt.figure(figsize=(10, 7))
  
# setting values to rows and column variables
rows = 2
columns = 1
  
# reading images
Image1 = rgb_img
Image2 = transformed_img

# Adds a subplot at the 1st position
comparison_fig.add_subplot(rows, columns, 1)
  
# showing image
plt.imshow(Image1)
plt.axis('off')
plt.title("RGB Image (1920x1080 px)")
  
# Adds a subplot at the 2nd position
comparison_fig.add_subplot(rows, columns, 2)
  
# showing image
plt.imshow(Image2, cmap = 'gray')
plt.axis('off')
plt.title("Tranformed Depth Image (1920x1080 px)")
  
###############################################################################################################
# DEPTH IMAGE VALUES ANALYSIS STARTS FROM HERE ################################################################
# Now that we have a numpy array that we retrieve from csv file we can start to play with distance values. 
## DISTANCES DESCRIPTIONS:
#plt.hist(transformed_img.flat, bins = 100, range = (1250,2500))
# From background to camera = 460 mm
# Mean distance plants from camera = 370 - 400 mm 

for x in range(transformed_img.shape[0]):
        for y in range(transformed_img.shape[1]):
            # for the given pixel at w,h, lets check its value against the threshold
            if transformed_img[x,y]> 1480: #note that the first parameter is actually a tuple object
                # lets set this to zero
                transformed_img[x,y] = 0
           
plt.imshow(transformed_img, cmap = 'gray')
plt.savefig('comparison_rgb_depth.png',dpi = 360)
#transformed_img.shape[0]

#plt.hist(transformed_img.flat, bins = 100, range = (300,transformed_img.max()))

#transformed_img.max()

#label_image, plants = measure.label(transformed_img,  connectivity=2,return_num=True)
#plt.imshow(label_image)


#image_label = label2rgb(label_image, image = transformed_img)
#plt.imshow(image_label)
