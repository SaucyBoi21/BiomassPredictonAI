# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 16:35:54 2023

@author: jcard
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 18:07:27 2023

@author: jcard
"""

#import os
import cv2
import matplotlib.pyplot as plt
import numpy as np 
#import glob
from matplotlib import pyplot as plt
from plantcv import plantcv as pcv
#from plotnine import ggplot, aes, geom_point
#import pandas as pd



# REMEMBER: We are plotting in console using PCV. Alternatives are: plt.imshow(img) or 
# TO SAVE IMAGES we are using plantcv.print_image(img, filename) however we could use Python: cv2.imwrite(filename, img[, params])

#img_name to avoid changing name in every path or when storing results  
img_name = 'T01_GH13_JC01_Feb-07-2023_0759_rgb'

# give img location use "/" instead of "\"
path = f'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/GH13_JC01/rgb_imgs/tray_1/raw_images/{img_name}.jpg'
path_analyzed_imgs = 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/GH13_JC01/rgb_imgs/tray_1/analyzed_imgs'
path_result = 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/GH13_JC01/rgb_imgs/tray_1/results_csv'

file_segmentation = 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/GH13_JC01/naives_segmentation/plant_segmentation_2.txt'

# tray_number = 1
# pcv.params.debug = "print"
# Let's develop the code for a single image first
# PCV is reading this img as BGR instead of RGB
raw_img, path, img_filename = pcv.readimage(filename=path, mode="native")
pcv.plot_image(raw_img)

# (X and Y are starting X,Y coordinate respectively)
# h = y_axis total lenght , w = x_axis total lenght
img = pcv.crop(img=raw_img, x=45, y=80, h=885, w=1850)
pcv.plot_image(img)
#pcv.print_image(img,'C:/Users/jcard/OneDrive - University of Georgia/side_projects/seedlings_Ferrarezi_Lab/process_imgs/img.jpg')

    # Make sure you are analizing an RGB for this to work. 
    #   rbg_img      = original image
    #   original_img = whether to include the original RGB images in the display: True (default) or False
colorspace_img = pcv.visualize.colorspaces(rgb_img=img,original_img=False)
pcv.plot_image(colorspace_img)
#pcv.print_image(colorspace_img,'C:/Users/jcard/OneDrive - University of Georgia/side_projects/seedlings_Ferrarezi_Lab/process_imgs/colorspace_options.jpg')

#   channel = desired colorspace ('l', 'a', or 'b')
hsv_h = pcv.rgb2gray_hsv(rgb_img=img, channel='h')
pcv.plot_image(hsv_h)

#pcv.print_image(lab_a,'C:/Users/jcard/OneDrive - University of Georgia/side_projects/seedlings_Ferrarezi_Lab/process_imgs/lab_a.jpg')

## To create a histrogram to obtain our threshold value using opencv
histogram, bin_edges = np.histogram(hsv_h, bins=256, range=(0, 255))
fig, ax = plt.subplots()
plt.plot(bin_edges[0:-1], histogram)


hsv_s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
pcv.plot_image(hsv_s)


histogram, bin_edges = np.histogram(hsv_s, bins=256, range=(0, 255))
fig, ax = plt.subplots()
plt.plot(bin_edges[0:-1], histogram)


# Explain how the value in threshold works in the image. Everything over or under is discarded? 
# We use img color channels as gray_img. 
thresh = pcv.threshold.binary(gray_img=hsv_h, threshold=, max_value=255, object_type='dark')
pcv.plot_image(thresh)


