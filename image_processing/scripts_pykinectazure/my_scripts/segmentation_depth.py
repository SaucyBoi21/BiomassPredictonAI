# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 12:30:08 2022

@author: jcard

Details: Besides retrieving and storing RGB and depth images. Includes image transformation.
The resulting transformed depth map provides a corresponding depth reading for every pixel of the color image.

The script runs but only stores the transformed image for some reason it doesnt stores the color image in the format needed
"""

import sys
import cv2
import time
import os
import open3d as o3d
import PIL as pil
import numpy as np
from matplotlib import pyplot as plt
from plantcv import plantcv as pcv
import pykinect_azure as pykinect
from pykinect_azure.k4a import _k4a
from pykinect_azure.k4a.calibration import Calibration
from pykinect_azure.k4a.image import Image
from pykinect_azure.k4a.transformation import Transformation
from pykinect_azure.k4a.capture import Capture


# Retrieving stored .jpg image without saving as a uint16 format. 
depth_csv_float = np.genfromtxt('C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/scripts_pykinectazure/my_scripts/depth_csv/2let_01Jan-17-2023_1722_depth_values.csv',delimiter =',')
plt.imshow(depth_csv_float, cmap = 'gray')
histogram2 = plt.hist(depth_csv_float.flat, bins = 100,range = (1000,1250))


# This will just store the location of the image. We need a function to retrieve it into variable explorer with channels values. 
# PCV.readimage is the same as retrieving the csv file as a numpy array, however it will turn distance values into 0-255 (brightness)
depth_csv = 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/scripts_pykinectazure/my_scripts/depth_csv/2let_01Jan-17-2023_1722_depth_values.csv'
depth_jpg = "C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/scripts_pykinectazure/my_scripts/transf_imgs/2let_01Jan-17-2023_1722_transformed.jpg"
img, path, img_filename = pcv.readimage(filename= depth_csv, mode="csv")

# This line of code will retrieve the csv file with depth values and transform it back to numpy. 
depth_data = np.array(np.genfromtxt('C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/scripts_pykinectazure/my_scripts/depth_csv/2let_01Jan-17-2023_1722_depth_values.csv',delimiter =',')).astype("uint16")
plt.imshow(depth_data, cmap = 'gray')

depth_data.min()
depth_data.max()

# Here we can transform the depth values numpy to an RGB image.  
#depth_alt = pil.Image.fromarray(np.uint8(depth_data)).convert('RGB')




## From here: ANALYSIS OF RGBD IMAGES
### Objective: Isolate the plant tissue by thresholding their distance.
# Probably the threshold is not the best approach since we are losing the distance information of each pixel.
# Or we could conver the 0 to the max valuea and segment. 
histogram, bin_edges = np.histogram(depth_data, bins=500, range=(0, depth_data.max()))
fig, ax = plt.subplots()
plt.plot(bin_edges[0:-1], histogram)

# I'm using this histogram to zoom in into the distances where my plant tissues are. 
histogram2 = plt.hist(depth_data.flat, bins = 100, range = (500,1200))


#I'm nor sure what this will do 
depth_data[np.where(depth_data == 0)] = depth_data.max()


ret2, thresh = cv2.threshold(depth_data, 1100,  depth_data.max(), cv2.THRESH_BINARY_INV)  
plt.imshow(thresh,cmap = 'gray')

thresh2 = pcv.threshold.binary(gray_img=transformed_numpy, threshold=400, max_value=transformed_numpy.max(), object_type='dark')
plt.imshow(thresh2,cmap = 'gray')



##### TEST FOR GETTING A COMBINED RGBD IMAGE #####
# I cannot print that image apparently. 
color_raw = o3d.io.read_image("C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/scripts_pykinectazure/my_scripts/rgb_imgs/2let_01Jan-18-2023_1615_rgb.jpg")
depth_raw = o3d.io.read_image("C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/scripts_pykinectazure/my_scripts/transf_imgs/2let_01Jan-18-2023_1615_transformed.jpg")
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw)

plt.subplot(1, 2, 1)
plt.title('RGB Kinect Sensor')
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title('Depth Kinect Sensor')
plt.imshow(rgbd_image.depth)
plt.show()
print(rgbd_image)

np.asarray(rgbd_image)
