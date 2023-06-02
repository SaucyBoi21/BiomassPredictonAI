# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 18:07:27 2023

@author: jcard
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np 
import glob
from matplotlib import pyplot as plt
from plantcv import plantcv as pcv
from plotnine import ggplot, aes, geom_point
import pandas as pd
import seaborn as sns


filenames = sorted(glob.glob('C:/Users/jcard\OneDrive - University of Georgia/kinect_imaging/GH13_JC01/naives_segmentation/morning_images/T03*.jpg'))

#stripped = filenames[1].rsplit('_',5)
#os.path.basename(filenames[1])

for j,filename in enumerate(filenames): 

    img_name = os.path.basename(filename)


# give img location use "/" instead of "\"
    path = f'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/GH13_JC01/naives_segmentation/morning_images/{img_name}.jpg'
    path_rgb= 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/GH13_JC01/naives_segmentation/morning_images/cut_images'
    path_mask= 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/GH13_JC01/naives_segmentation/morning_images/masks'
    
    
    # tray_number = 1
    # pcv.params.debug = "print"
    # Let's develop the code for a single image first
    # PCV is reading this img as BGR instead of RGB
    raw_img, path, img_filename = pcv.readimage(filename, mode="native")
    pcv.plot_image(raw_img)
    
    # (X and Y are starting X,Y coordinate respectively)
    # h = y_axis total lenght , w = x_axis total lenght
    img = pcv.crop(img=raw_img, x=0, y=110, h=880, w=1900)
    pcv.plot_image(img)
    
    pcv.print_image(img,os.path.join(path_rgb, f'{img_name}.jpg'))
    #pcv.print_image(img,'C:/Users/jcard/OneDrive - University of Georgia/side_projects/seedlings_Ferrarezi_Lab/process_imgs/img.jpg')
    
        # Make sure you are analizing an RGB for this to work. 
        #   rbg_img      = original image
        #   original_img = whether to include the original RGB images in the display: True (default) or False
    colorspace_img = pcv.visualize.colorspaces(rgb_img=img,original_img=False)
    pcv.plot_image(colorspace_img)
    #pcv.print_image(colorspace_img,'C:/Users/jcard/OneDrive - University of Georgia/side_projects/seedlings_Ferrarezi_Lab/process_imgs/colorspace_options.jpg')
    
    #   channel = desired colorspace ('l', 'a', or 'b')
    hsv_s = pcv.rgb2gray_lab(rgb_img=img, channel='b')
    pcv.plot_image(hsv_s)
    #pcv.print_image(lab_a,'C:/Users/jcard/OneDrive - University of Georgia/side_projects/seedlings_Ferrarezi_Lab/process_imgs/lab_a.jpg')
    
    ## To create a histrogram to obtain our threshold value using opencv
    histogram, bin_edges = np.histogram(hsv_s, bins=256, range=(0, 255))
    fig, ax = plt.subplots()
    plt.plot(bin_edges[0:-1], histogram)
    
    
    # Explain how the value in threshold works in the image. Everything over or under is discarded? 
    # We use img color channels as gray_img. 
    thresh = pcv.threshold.binary(gray_img=hsv_s, threshold=138, max_value=255, object_type='light')
    pcv.plot_image(thresh)
    #pcv.print_image(thresh,'C:/Users/jcard/OneDrive - University of Georgia/side_projects/seedlings_Ferrarezi_Lab/process_imgs/thresh_laba.jpg')
    # =============================================================================
    #  THIS IS PLANTCV histogram option. Creates a ggplot object. 
    #hist2 = pcv.visualize.histogram(img=img,hist_data=True)
    # =============================================================================
    
    # Remove small background noise
    ## The fill function removes "salt" noise from the background by filtering white regions by size.
    
    # Inputs:
    #   bin_img = Binary image data
    #   size    = minimum object area size in pixels (integer), smaller objects will be filled
    h_fill = pcv.fill(bin_img=thresh, size=100)
    pcv.plot_image(h_fill)
    pcv.print_image(h_fill,os.path.join(path_mask, f'{img_name}.jpg'))

