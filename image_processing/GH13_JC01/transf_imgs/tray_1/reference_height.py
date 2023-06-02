
"""
Created on Thu Oct 27 22:45:31 2022

@author: jcard

Height reference for depth analysis. We will create a function that calculates the reference
from an empty tray. The picture of the tray will be taken at the beginning of the experiment. 

"""
import os
import matplotlib.pyplot as plt
import numpy as np 
# import glob
# from matplotlib import pyplot as plt
from plantcv import plantcv as pcv
import pandas as pd




# for now add img paths manually 
path_color = "C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/GH13_JC01/rgb_imgs/tray_1/raw_images/T01_GH13_JC01_Jan-25-2023_0832_rgb.jpg"
path_depth_csv = "C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/GH13_JC01/depth_csv/tray_1/T01_GH13_JC01_Jan-25-2023_0832_depth_values.csv"
path_result_imgs = 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/GH13_JC01/transf_imgs/tray_1/analyzed_imgs'
path_result_csv = 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/GH13_JC01/transf_imgs/tray_1/results_csv'

# Obtain raft height from picture. Alternatively, I could use the empty raft image to obtain that mean value of height


# tray_number = 1
# pcv.params.debug = "print"
# Let's develop the code for a single image first
# Here we will store a numpy with the distance values by retrieving the csv file. 
img, path, img_filename = pcv.readimage(filename=path_depth_csv, mode="csv")
pcv.plot_image(img)

histogram_depth, bin_edges = np.histogram(img, bins=256, range=(1150, 1300))
fig, ax = plt.subplots()
plt.plot(bin_edges[0:-1], histogram_depth)

# To improve colorspaces visualization, cut the image in such a way that just light blue background is visible. 
# (X and Y are starting X,Y coordinate respectively)
# h = y_axis total lenght , w = x_axis total lenght

# Since the original file is a csv we will need to split the array of floats according to the size we need 
crop_image = img[85:960,list(range(55,1880))]
#img = pcv.crop(img=depth_img, x=45, y=80, h=885, w=1850)
#pcv.plot_image(img)
pcv.plot_image(crop_image)

crop_image.mean()

histogram_depth, bin_edges = np.histogram(crop_image, bins=256, range=(1150, 1300))
fig, ax = plt.subplots()
plt.plot(bin_edges[0:-1], histogram_depth)
    
# Inputs:
#   gray_img - Grayscale image data 
#   min_value - New minimum value for range of interest. default = 0
#   max_value - New maximum value for range of interest. default = 255
scaled_distance_img = pcv.transform.rescale(gray_img=img)
pcv.plot_image(scaled_distance_img)

# Besides, I need to retrieve the rgb image in order to create the mask based on RGB values. 

color_img,path_color,filename_color =  pcv.readimage(filename=path_color, mode="rgb")
pcv.plot_image(color_img)

# From this point we will use the color_img for any operation that will lead us to the mask. 
    # Make sure you are analizing an RGB for this to work. 
    #   rbg_img      = original image
    #   original_img = whether to include the original RGB images in the display: True (default) or False
colorspace_img = pcv.visualize.colorspaces(rgb_img= color_img,original_img=False)
pcv.plot_image(colorspace_img)
#pcv.print_image(colorspace_img,'C:/Users/jcard/OneDrive - University of Georgia/side_projects/seedlings_Ferrarezi_Lab/process_imgs/colorspace_options.jpg')

#   channel = desired colorspace ('l', 'a', or 'b')
hsv_h = pcv.rgb2gray_lab(rgb_img=color_img, channel='b')
pcv.plot_image(hsv_h)
#pcv.print_image(lab_a,'C:/Users/jcard/OneDrive - University of Georgia/side_projects/seedlings_Ferrarezi_Lab/process_imgs/lab_a.jpg')

## To create a histrogram to obtain our threshold value using opencv
histogram, bin_edges = np.histogram(hsv_h, bins=256, range=(0, 255))
fig, ax = plt.subplots()
plt.plot(bin_edges[0:-1], histogram)


# Explain how the value in threshold works in the image. Everything over or under is discarded? 
# We use img color channels as gray_img. 
thresh = pcv.threshold.binary(gray_img=hsv_h, threshold=130, max_value=255, object_type='light')
pcv.plot_image(thresh)
#pcv.print_image(thresh,'C:/Users/jcard/OneDrive - University of Georgia/side_projects/seedlings_Ferrarezi_Lab/process_imgs/thresh_laba.jpg')
# =============================================================================
#  THIS IS PLANTCV histogram option. Creates a ggplot object. 
#hist2 = pcv.visualize.histogram(img=crop,hist_data=True)
# =============================================================================

# Remove small background noise
## The fill function removes "salt" noise from the background by filtering white regions by size.

# Inputs:
#   bin_img = Binary image data
#   size    = minimum object area size in pixels (integer), smaller objects will be filled
h_fill = pcv.fill(bin_img=thresh, size=1000)
pcv.plot_image(h_fill)


#### From this point we need to put together the mask defined as h_fill 


# Identify the outlines of all plants
## The binary mask (all values are either white or black) that resulted from thresholding and filtering 
## the thresholded image is used to identify the polygons that define the outlines of every connected white region. 
## Objects (or contours) can be nested, so a hierarchy that defines the relationship between objects is also calculated.

# Inputs:
#   img  = input image
#   mask = a binary mask used to detect objects
# Is there a possibility to print both the img and the mask on top? 
obj, obj_hierarchy = pcv.find_objects(img = scaled_distance_img, mask=h_fill)
pcv.plot_image(obj)

# ROIS will create a list with all the 98 position in the seeding tray. 
# Use len(rois) for the expected total seedling count. 
rois, roi_hierarchy = pcv.roi.multi(img=scaled_distance_img, coord=(168,205), radius=70, 
                                    spacing=(159, 161), nrows=5, ncols=11)


# Create a unique ID for each plant
## Create a sequence of values to label each plant within the image based on the ROI IDs.

# Inputs:
#   start = beginning value for range
#   stop  = ending value for range (exclusive)
plant_ids = range(0, len(rois))


#masked_depth2 = img[np.where(mask > 0)]
#masked_depth2[np.where(masked_depth2 == 0)] = masked_depth2.max()

##### Analysis #####
# Here we create a copy of the original crop image
# Do we need to create copy of color image or depth image or rescaled image

img_copy = np.copy(scaled_distance_img)

# Create a for loop to interate through every ROI (plant) in the image
for i in range(0, len(rois)):
    # The ith ROI, ROI hierarchy, and plant ID
    roi = rois[i]
    hierarchy = roi_hierarchy[i]
    plant_id = plant_ids[i]
    # Subset objects that overlap the ROI
    # Inputs:
    #   img            = input image
    #   roi_contour    = a single ROI contour
    #   roi_hierarchy  = a single ROI hierarchy
    #   object_contour = all objects detected in a binary mask
    #   obj_hierarchy  = all object hierarchies
    #   roi_type       = "partial" (default) keeps contours that overlap
    #                    or are contained in the ROI. "cutto" cuts off
    #                    contours that fall outside the ROI. "largest"
    #                    only keeps the largest object within the ROI
    plant_contours, plant_hierarchy, mask, area = pcv.roi_objects(img=scaled_distance_img, 
                                                                  roi_contour=roi, 
                                                                  roi_hierarchy=hierarchy, 
                                                                  object_contour=obj, 
                                                                  obj_hierarchy=obj_hierarchy, 
                                                                  roi_type="partial")

    # If the plant area is zero then no plant was detected for the ROI
    # and no measurements can be done
    if area > 0:
        # Combine contours together for each plant
        # Inputs:
        #   img       = input image
        #   contours  = contours that will be consolidated into a single object
        #   hierarchy = the relationship between contours
        plant_obj, plant_mask = pcv.object_composition(img=scaled_distance_img, 
                                                       contours=plant_contours, 
                                                       hierarchy=plant_hierarchy)       
        # Analyze the shape of each plant
        # Inputs:
        #   img   = input image
        #   obj   = composed object contours
        #   mask  = binary mask that contours were derived from
        #   label = a label for the group of measurements (default = "default")
        thermal = pcv.analyze_thermal_values(img, plant_mask, histplot=True, label=f"plant{plant_id}")
        #img_copy = pcv.analyze_object(img=img_copy, obj=plant_obj, 
                                      #mask=plant_mask, label=f"plant{plant_id}")
        

pcv.plot_image(img_copy)
# pcv.print_image(img_copy,os.path.join(path_imgs, f'{img_name}_analysis.jpg'))

# I'm storing all observations from the output as a dictionary to index values from that dictionary
plant_data = pcv.outputs.observations