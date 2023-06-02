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
from plantcv import plantcv as pcv
import pandas as pd



# REMEMBER: We are plotting in console using PCV. Alternatives are: plt.imshow(img) or 
# TO SAVE IMAGES we are using plantcv.print_image(img, filename) however we could use Python: cv2.imwrite(filename, img[, params])

#img_name to avoid changing name in every path or when storing results  
img_name = 'T02_GH13_JC01_Feb-14-2023_0747_rgb'

# give img location use "/" instead of "\"
path = f'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/GH13_JC01/rgb_imgs/tray_2/raw_images/{img_name}.jpg'
path_analyzed_imgs = 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/GH13_JC01/rgb_imgs/tray_2/analyzed_imgs'
path_result = 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/GH13_JC01/rgb_imgs/tray_2/results_csv'

# tray_number = 1
# pcv.params.debug = "print"
# Let's develop the code for a single image first
# PCV is reading this img as BGR instead of RGB
raw_img, path, img_filename = pcv.readimage(filename=path, mode="native")
pcv.plot_image(raw_img)


############# CROP TO RAFT PAVEL'S IMPLEMENTATION ##############                 # open a file
#black = np.zeros(raw_img.shape).astype(raw_img.dtype)

'''
#img = cv2.resize(img, (2000, 1400))            # Resize image
img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)  # change the colouring


# Definition of blue color around plate
lower = [0 , 0, 110]  # plates 60, 31, 165
higher = [255, 255, 255]
lower = np.array(lower, dtype = "uint8")
higher = np.array(higher, dtype = "uint8")

# Picture cropping
raft_mask = cv2.inRange(img,lower,higher)# Make a mask
pcv.plot_image(raft_mask)
cont, _ = cv2.findContours(raft_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # search for contours
#print(cont[1])
cont = sorted(cont,key=cv2.contourArea)
#print(cont[-1])
#cont_img = cv2.drawContours(img,cont,-1,(0,0,255),0)
#cont = np.array([[50, 50], [50, 150], [150, 150], [150, 50]])

cv2.fillPoly(black, pts = [cont[-1]], color=[255,255,255])
#cv2.drawContours(img, cont[-1], -1, color=(255, 255, 255), thickness=cv2.FILLED)
result = cv2.bitwise_and(img, black)
result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)  # change the colouring
pcv.plot_image(result)


#result = cv2.resize(result,(1000,500))

#cv2.imshow(file, result)
#cv2.waitKey(0)
#cv2.imwrite(mypath+"Raft_cropped/"+file, result)
'''


# (X and Y are starting X,Y coordinate respectively)
# h = y_axis total lenght , w = x_axis total lenght
#img = pcv.crop(img=raw_img, x=45, y=80, h=885, w=1850)
#pcv.plot_image(img)
#pcv.print_image(img,'C:/Users/jcard/OneDrive - University of Georgia/side_projects/seedlings_Ferrarezi_Lab/process_imgs/img.jpg')

    # Make sure you are analizing an RGB for this to work. 
    #   rbg_img      = original image
    #   original_img = whether to include the original RGB images in the display: True (default) or False
    
colorspace_img = pcv.visualize.colorspaces(rgb_img=raw_img,original_img=False)
pcv.plot_image(colorspace_img)
#pcv.print_image(colorspace_img,'C:/Users/jcard/OneDrive - University of Georgia/side_projects/seedlings_Ferrarezi_Lab/process_imgs/colorspace_options.jpg')

#   channel = desired colorspace ('l', 'a', or 'b')
hsv_s = pcv.rgb2gray_cmyk(rgb_img=raw_img, channel='c')
pcv.plot_image(hsv_s)
#pcv.print_image(lab_a,'C:/Users/jcard/OneDrive - University of Georgia/side_projects/seedlings_Ferrarezi_Lab/process_imgs/lab_a.jpg')

## To create a histrogram to obtain our threshold value using opencv
histogram, bin_edges = np.histogram(hsv_s, bins= 100, range=(0, 255))
fig, ax = plt.subplots()
plt.plot(bin_edges[0:-1], histogram)


# Explain how the value in threshold works in the image. Everything over or under is discarded? 
# We use img color channels as gray_img. 
thresh = pcv.threshold.binary(gray_img=hsv_s, threshold=125, max_value=255, object_type='light')
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
h_fill = pcv.fill(bin_img=thresh, size=200)
pcv.plot_image(h_fill)

# Identify the outlines of all plants
## The binary mask (all values are either white or black) that resulted from thresholding and filtering 
## the thresholded image is used to identify the polygons that define the outlines of every connected white region. 
## Objects (or contours) can be nested, so a hierarchy that defines the relationship between objects is also calculated.

# Inputs:
#   img  = input image
#   mask = a binary mask used to detect objects
# Is there a possibility to print both the img and the mask on top? 
obj, obj_hierarchy = pcv.find_objects(img=img, mask=h_fill)


# ROIS will create a list with all the 98 position in the seeding tray. 
# Use len(rois) for the expected total seedling count. 
rois, roi_hierarchy = pcv.roi.multi(img=img, coord=(125,125), radius=50, 
                                    spacing=(159, 161), nrows=5, ncols=11)

# this one as well will have a len of the expected total seedlings but in range format. 
plant_ids = range(0, len(rois))


# Here we create a copy of the original img image
img_copy = np.copy(img)

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
    plant_contours, plant_hierarchy, mask, area = pcv.roi_objects(img=img, 
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
        plant_obj, plant_mask = pcv.object_composition(img=img, 
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
pcv.print_image(img_copy,os.path.join(path_analyzed_imgs, f'{img_name}_analysis.jpg'))

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
pcv.outputs.save_results(filename = f"C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/GH13_JC01/rgb_imgs/tray_1/csv_plantcv/pcv_{img_name}.csv", outformat="csv")
pcv.outputs.clear()