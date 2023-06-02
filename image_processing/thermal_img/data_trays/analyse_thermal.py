# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 09:18:18 2023
READ THERMAL INFO DATA FROM SEEKTHERMAL
@author: jcard
"""

import pandas as pd
import numpy as np 
import cv2
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt

path = 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/thermal_img/data_trays/May-12-2023_1648_thermography-E2929D780611.csv'

def thermal_seek(filename, threshold, pseudo_min, pseudo_max):


    data = pd.read_csv(filename ,delimiter=" ", header=None)
    data.to_csv("thermal_data.csv",header=None,index=None)
    
    
    # def analyse_thermal(csv_img,) We will define a function after we are sure it runs 
    
    thermal_data,path,filename = pcv.readimage(filename= "thermal_data.csv" , mode='csv')
    pcv.plot_image(thermal_data)
    
    
    # Rescale the thermal data to a colorspace with range 0-255 rather than raw data 
        
    # Inputs:
    #   gray_img - Grayscale image data 
    #   min_value - New minimum value for range of interest. default = 0
    #   max_value - New maximum value for range of interest. default = 255
    scaled_thermal_img = pcv.transform.rescale(gray_img=thermal_data)
    pcv.plot_image(scaled_thermal_img)
    
    # Threshold the thermal data to make a binary mask
        
    # Inputs:
    #   gray_img - Grayscale image data 
    #   threshold- Threshold value (between 0-255)
    #   max_value - Value to apply above threshold (255 = white) 
    #   object_type - 'light' (default) or 'dark'. If the object is lighter than the background then standard 
    #                 threshold is done. If the object is darker than the background then inverse thresholding is done. 
    bin_mask = pcv.threshold.binary(gray_img=thermal_data, threshold=threshold, max_value=255, object_type='dark')
    pcv.plot_image(bin_mask)
    
    # Identify objects
    # Inputs: 
    #   img - RGB or grayscale image data for plotting 
    #   mask - Binary mask used for detecting contours 
    id_objects, obj_hierarchy = pcv.find_objects(img=scaled_thermal_img, mask=bin_mask)
    
    
    # Define the region of interest (ROI) 
    
    # Inputs: 
    #   img - RGB or grayscale image to plot the ROI on 
    #   x - The x-coordinate of the upper left corner of the rectangle 
    #   y - The y-coordinate of the upper left corner of the rectangle 
    #   h - The height of the rectangle 
    #   w - The width of the rectangle 
    roi, roi_hierarchy= pcv.roi.rectangle(img=scaled_thermal_img, x=0, y=0, h=220, w=310)
    
    
    # Decide which objects to keep
    
    # Inputs:
    #    img            = img to display kept objects
    #    roi_contour    = contour of roi, output from any ROI function
    #    roi_hierarchy  = contour of roi, output from any ROI function
    #    object_contour = contours of objects, output from pcv.find_objects function
    #    obj_hierarchy  = hierarchy of objects, output from pcv.find_objects function
    #    roi_type       = 'partial' (default, for partially inside the ROI), 'cutto', or 
    #                     'largest' (keep only largest contour)
    roi_objects, hierarchy, kept_mask, obj_area = pcv.roi_objects(img=scaled_thermal_img,roi_contour=roi,
                                                                  roi_hierarchy=roi_hierarchy,
                                                                  object_contour=id_objects,
                                                                  obj_hierarchy=obj_hierarchy, 
                                                                  roi_type='cutto')
    pcv.plot_image(kept_mask)
    
    # Inputs:
    #   img - Array of thermal values
    #   mask - Binary mask made from selected contours
    #   histplot - If True plots histogram of intensity values (default histplot = False)
    #   label - Optional label parameter, modifies the variable name of observations recorded. (default `label="default"`)filled_img = pcv.morphology.fill_segments(mask=cropped_mask, objects=edge_objects)
    
    analysis_img = pcv.analyze_thermal_values(thermal_array=thermal_data, mask=kept_mask, histplot=True, label="default")
    thermal_output = pcv.outputs.observations
    
    # Pseudocolor the thermal data 
    
    # Inputs:
    #     gray_img - Grayscale image data
    #     obj - Single or grouped contour object (optional), if provided the pseudocolored image gets 
    #           cropped down to the region of interest.
    #     mask - Binary mask (optional) 
    #     background - Background color/type. Options are "image" (gray_img, default), "white", or "black". A mask 
    #                  must be supplied.
    #     cmap - Colormap
    #     min_value - Minimum value for range of interest
    #     max_value - Maximum value for range of interest
    #     dpi - Dots per inch for image if printed out (optional, if dpi=None then the default is set to 100 dpi).
    #     axes - If False then the title, x-axis, and y-axis won't be displayed (default axes=True).
    #     colorbar - If False then the colorbar won't be displayed (default colorbar=True)
    pseudo_img = pcv.visualize.pseudocolor(gray_img = thermal_data, mask=kept_mask, cmap='jet', 
                                           min_value=pseudo_min, max_value=pseudo_max)

    return thermal_output,pseudo_img

output,image = thermal_seek(path, 19, 17,19 )
image

pcv.print_image(image,'C:/Users/jcard/OneDrive - University of Georgia/side_projects/thermal.jpg')


