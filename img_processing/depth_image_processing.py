# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 22:45:31 2022

@author: jcard

Details: From the RGB image retrieved before, this script will perform the image processing
required for obtaining all features for each individual plant inside a tray. 


"""
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np 
import open3d as o3d
import glob
from matplotlib import pyplot as plt
from plantcv import plantcv as pcv
from plotnine import ggplot, aes, geom_point
import pandas as pd
import seaborn as sns

from plantcv.plantcv import deprecation_warning, params
from plantcv.plantcv import outputs
from plantcv.plantcv._debug import _debug
from plantcv.plantcv.visualize import histogram
from plotnine import labs

# REMEMBER: We are plotting in console using PCV. Alternatives are: plt.imshow(img) or 
# TO SAVE IMAGES we are using plantcv.print_image(img, filename) however we could use Python: cv2.imwrite(filename, img[, params])

#img_name to avoid changing name in every path or when storing results  
# img_name = 'Oct-11-2022_1516_RGB_tray1'

# give img location use "/" instead of "\"
color_jpg = "C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/scripts_pykinectazure/my_scripts/rgb_imgs/2let_01Jan-17-2023_1722_rgb.jpg"
depth_jpg = "C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/scripts_pykinectazure/my_scripts/transf_imgs/2let_01Jan-17-2023_1722_transformed.jpg"
depth_csv = 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/scripts_pykinectazure/my_scripts/depth_csv/2let_01Jan-17-2023_1722_depth_values.csv'
path_result_imgs = 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/scripts_pykinectazure/my_scripts/analyzed_imgs'
path_result_csv = 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/scripts_pykinectazure/my_scripts/results_csv'

# tray_number = 1
# pcv.params.debug = "print"
# Let's develop the code for a single image first
# Here we will store a numpy with the distance values by retrieving the csv file. 
img, path, img_filename = pcv.readimage(filename=depth_csv, mode="csv")
pcv.plot_image(img)



# Rescale distance data to a colorspace with range 0-255 rather than raw data 
    
# Inputs:
#   gray_img - Grayscale image data 
#   min_value - New minimum value for range of interest. default = 0
#   max_value - New maximum value for range of interest. default = 255
scaled_distance_img = pcv.transform.rescale(gray_img=img)
pcv.plot_image(scaled_distance_img)

# Besides, I need to retrieve the rgb image in order to create the mask based on RGB values. 

color_img,path_color,filename_color =  pcv.readimage(filename=color_jpg, mode="rgb")
pcv.plot_image(color_img)

# From this point we will use the color_img for any operation that will lead us to the mask. 
    # Make sure you are analizing an RGB for this to work. 
    #   rbg_img      = original image
    #   original_img = whether to include the original RGB images in the display: True (default) or False
colorspace_img = pcv.visualize.colorspaces(rgb_img= color_img,original_img=False)
pcv.plot_image(colorspace_img)
#pcv.print_image(colorspace_img,'C:/Users/jcard/OneDrive - University of Georgia/side_projects/seedlings_Ferrarezi_Lab/process_imgs/colorspace_options.jpg')

#   channel = desired colorspace ('l', 'a', or 'b')
hsv_h = pcv.rgb2gray_hsv(rgb_img=color_img, channel='h')
pcv.plot_image(hsv_h)
#pcv.print_image(lab_a,'C:/Users/jcard/OneDrive - University of Georgia/side_projects/seedlings_Ferrarezi_Lab/process_imgs/lab_a.jpg')

## To create a histrogram to obtain our threshold value using opencv
histogram, bin_edges = np.histogram(hsv_h, bins=256, range=(0, 255))
fig, ax = plt.subplots()
plt.plot(bin_edges[0:-1], histogram)


# Explain how the value in threshold works in the image. Everything over or under is discarded? 
# We use img color channels as gray_img. 
thresh = pcv.threshold.binary(gray_img=hsv_h, threshold=50, max_value=255, object_type='dark')
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
h_fill = pcv.fill(bin_img=thresh, size=250)
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
roi, roi_hierarchy = pcv.roi.rectangle(img=scaled_distance_img, x=700, y=300, h=500, w=500)



plant_contours, plant_hierarchy, mask, area = pcv.roi_objects(img=scaled_distance_img, 
                                                               roi_contour=roi, 
                                                               roi_hierarchy=roi_hierarchy, 
                                                               object_contour=obj, 
                                                               obj_hierarchy=obj_hierarchy, 
                                                               roi_type="cutto")

masked_depth2 = img[np.where(mask > 0)]
masked_depth2[np.where(masked_depth2 == 0)] = masked_depth2.max()
##### Analysis #####

# Analyze thermal data 

# Inputs:
#   img - Array of thermal values
#   mask - Binary mask made from selected contours
#   histplot - If True plots histogram of intensity values (default histplot = False)
#   label - Optional label parameter, modifies the variable name of observations recorded. (default `label="default"`)filled_img = pcv.morphology.fill_segments(mask=cropped_mask, objects=edge_objects)

### THIS RESULTS ARE PROVIDED BY ANALYZE THERMAL VALUES:
def analyze_depth_values(depth_array, mask, histplot=None, label="default"):
    """This extracts the depth values of each pixel writes the values out to
       a file. It can also print out a histogram plot of pixel intensity
       and a pseudocolor image of the plant.
    Inputs:
    array        = numpy array of depth values
    mask         = Binary mask made from selected contours
    histplot     = if True plots histogram of depth values
    label        = optional label parameter, modifies the variable name of observations recorded
    Returns:
    analysis_image = output image
    :param depth_array: numpy.ndarray
    :param mask: numpy.ndarray
    :param histplot: bool
    :param label: str
    :return analysis_image: ggplot
    """
    if histplot is not None:
        deprecation_warning("'histplot' will be deprecated in a future version of PlantCV. "
                            "This function creates a histogram by default.")

    # Store debug mode
    debug = params.debug

    # apply plant shaped mask to image and calculate statistics based on the masked image
    # I'm replacing the depth values of 0 with the mean height value 
    masked_depth = depth_array[np.where(mask > 0)] 
    maxdistance = np.amax(masked_depth)
    avgdistance = np.average(masked_depth)
    masked_depth[np.where(masked_depth == 0)] =avgdistance
    mindistance = np.amin(masked_depth)
    mediandistance = np.median(masked_depth)

    # call the histogram function
    params.debug = None
    hist_fig, hist_data = histogram(depth_array, mask=mask, hist_data=True)
    bin_labels, hist_percent = hist_data['pixel intensity'].tolist(), hist_data['proportion of pixels (%)'].tolist()

    # Store data into outputs class
    outputs.add_observation(sample=label, variable='max_distance', trait='maximum distance',
                            method='plantcv.plantcv.analyze_depth_values', scale='degrees', datatype=float,
                            value=maxdistance, label='mm')
    outputs.add_observation(sample=label, variable='min_distance', trait='minimum distance',
                            method='plantcv.plantcv.analyze_depth_values', scale='degrees', datatype=float,
                            value=mindistance, label='mm')
    outputs.add_observation(sample=label, variable='mean_distance', trait='mean distance',
                            method='plantcv.plantcv.analyze_depth_values', scale='degrees', datatype=float,
                            value=avgdistance, label='mm')
    outputs.add_observation(sample=label, variable='median_distance', trait='median distance',
                            method='plantcv.plantcv.analyze_depth_values', scale='degrees', datatype=float,
                            value=mediandistance, label='mm')
    outputs.add_observation(sample=label, variable='depth_frequencies', trait='depth frequencies',
                            method='plantcv.plantcv.analyze_depth_values', scale='frequency', datatype=list,
                            value=hist_percent, label=bin_labels)
    # Restore user debug setting
    params.debug = debug

    # change column names of "hist_data"
    hist_fig = hist_fig + labs(x="distance (mm)", y="Proportion of pixels (%)")

    # Print or plot histogram
    _debug(visual=hist_fig, filename=os.path.join(params.debug_outdir, str(params.device) + "_therm_histogram.png"))

    analysis_image = hist_fig
    # Store images
    outputs.images.append(analysis_image)

    return analysis_image

depth_histogram = analyze_depth_values(img, mask, histplot=True, label="default")
depth_database = pcv.outputs.observations


pcv.outputs.clear()
########## THIS PART OF THE CODE WILL BE USEFUL FOR ANALYSING MORE THAN ONE SINGLE PLANT
# Here we create a copy of the original crop image
img_copy = np.copy(crop)

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
    plant_contours, plant_hierarchy, mask, area = pcv.roi_objects(img=crop, 
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
        plant_obj, plant_mask = pcv.object_composition(img=crop, 
                                                       contours=plant_contours, 
                                                       hierarchy=plant_hierarchy)       
        # Analyze the shape of each plant
        # Inputs:
        #   img   = input image
        #   obj   = composed object contours
        #   mask  = binary mask that contours were derived from
        #   label = a label for the group of measurements (default = "default")
        img_copy = pcv.analyze_object(img=img_copy, obj=plant_obj, 
                                      mask=plant_mask, label=f"plant{plant_id}")
        

pcv.plot_image(img_copy)
pcv.print_image(img_copy,os.path.join(path_imgs, f'{img_name}_analysis.jpg'))

# I'm storing all observations from the output as a dictionary to index values from that dictionary
plant_data = pcv.outputs.observations



# How do we append results once we have additional variables
# MAKE SURE THIS WORKS FIRST

data_parameters = {}
for key in plant_data['plant0'].keys():
  data_parameters['plant_'+ key] = []
  for plant in plant_data.keys():
     data_parameters['plant_'+ key].append(plant_data[plant][key]['value'])
  
# List containing the plant_ids generated by code above. Use this to create a column with plant identification.    
plant_observation = list(plant_data.keys())
 #calculation of germination rate
germination_rate = round(((len(plant_observation)/len(rois))*100),2) 
# Saving the new dictionary with each parameter value per plant as a dataframe. 
data_set = pd.DataFrame(data_parameters)

# I'm adding a new column with the identification for each plant. Insert plant id at the beginning
data_set.insert(0, 'plant_id', plant_observation)
data_set.insert(len(data_set.columns), 'total_plants', len(rois))
data_set.insert(len(data_set.columns), 'germinated_plants', len(plant_observation))
data_set.insert(len(data_set.columns), 'germination_rate', germination_rate)

data_set.to_csv(os.path.join(path_result,f'{img_name}_data.csv'))


# Find a way to include this variables and make the new dictionary functional. We could create a new database just wiht germination info.  
## Add results of seed count / germination rate. 
#pcv.outputs.add_observation(sample='default', variable='total_seeds', 
                            #trait='total amound of seeds sowed',
                            #method='ratio of pixels', scale='percent', datatype=int,
                            #value=len(rois), label='# seeds')
#pcv.outputs.add_observation(sample='default', variable='germination rate', 
                            #trait='seeds germinated per tray',
                            #method='ratio of pixels', scale='percent', datatype=int,
                            #value=germination_rate, label='% germination')


## Look a the germination before writting it into a file (we could display this for the user). 

# plant_area will store the results generated as a dictionaRy
# plant_data = pcv.outputs.observations


# Check how the results given by plantcv (from a dictionary file) look 
pcv.outputs.save_results(filename = f"C:/Users/jcard/OneDrive - University of Georgia/side_projects/seedlings_Ferrarezi_Lab/lan_2nd_trial/results_csv/pcv_{img_name}.csv", outformat="csv")
pcv.outputs.clear()




    

