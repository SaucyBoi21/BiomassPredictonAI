# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 12:30:08 2022

@author: jcard
"""
import sys
import cv2
import time
import os
import numpy as np
from matplotlib import pyplot as plt
import pykinect_azure as pykinect
from pykinect_azure.k4a import _k4a
from pykinect_azure.k4a.calibration import Calibration
from pykinect_azure.k4a.image import Image
from pykinect_azure.k4a.transformation import Transformation
from pykinect_azure.k4a.capture import Capture
timestamp = time.strftime("%b-%d-%Y_%H%M")

path_rgb = 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/scripts_pykinectazure/my_scripts/rgb_imgs'
path_trans = 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/scripts_pykinectazure/my_scripts/transf_imgs'


# Configure and start the device. 
pykinect.initialize_libraries()
device_config = pykinect.default_configuration
device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_15
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_UNBINNED
device_config.depth_format = pykinect.K4A_IMAGE_FORMAT_DEPTH16

device = pykinect.start_device(config=device_config)


# Retrieve calibration data: Requires deph_mode and color_resolution parameters as input. 
calibration = device.get_calibration(device_config.depth_mode, device_config.color_resolution)


# Obtain a transformation : depth image to color camera ()
# We are transforming from a depth camera 2D to a Color Camera 2D
# The resulting transformed depth map provides a corresponding depth reading for every pixel of the color image. 

## STEP 1: Create a transformation handle that is going to be used as an argument for k4a_transformation_depth_image_to_color_camera()
# IN this case Transformation class will give me a object using k4a_transformation_create()

trans_handle = Transformation(calibration._handle)

# Get a capture from the device 
capture = device.update()

ret, color_ref = capture.get_color_image()
plt.imshow(color_ref)
cv2.imwrite(os.path.join(path_rgb, timestamp + "_rgb.jpg"), color_ref)
# =============================================================================
# cv2.imshow('Transformed Depth to Color Image',color_ref)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# =============================================================================


depth_ref = capture.get_depth_image_object()
# =============================================================================
# cv2.imshow('Depth Image',depth_ref)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# =============================================================================

# Im using the function .depth image to color camera with the trans_handle object and that will return an image object. 
# Then I can use the function transformed image.tonumpy() to generate a numpy array of the transformed image. 
# The numpy array generated will have the same size as RGB image.
# Instead of color values it will hold distance values. 

transformed_image = trans_handle.depth_image_to_color_camera(depth_ref)
#plt.imshow(transformed_image, cmap = 'gray')
ret, transformed_numpy= transformed_image.to_numpy()
plt.imshow(transformed_numpy, cmap = 'gray')
cv2.imwrite(os.path.join(path_trans, timestamp + "_transformed.jpg"), transformed_numpy)
# =============================================================================
# 
# cv2.imshow('Transformed Image',transformed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# =============================================================================

np.savetxt(timestamp + "rgbd_values_comparison.csv", transformed_numpy, delimiter=",")

device_off = device.close()


## From here: ANALYSIS OF RGBD IMAGES
### Objective: Isolate the plant tissue by thresholding their distance.
# Probably the threshold is not the best approach since we are losing the distance information of each pixel.
# Or we could conver the 0 to the max valuea and segment. 
histogram, bin_edges = np.histogram(transformed_numpy, bins=500, range=(0, transformed_numpy.max()))
fig, ax = plt.subplots()
plt.plot(bin_edges[0:-1], histogram)

histogram2 = plt.hist(transformed_numpy.flat, bins = 100, range = (300,500))


transformed_numpy[np.where(transformed_numpy == 0)] = transformed_numpy.max()
transformed_numpy.min()
transformed_numpy.max()

ret2, thresh = cv2.threshold(transformed_numpy, 415,  transformed_numpy.max(), cv2.THRESH_BINARY_INV)  
plt.imshow(thresh,cmap = 'gray')
thresh2 = pcv.threshold.binary(gray_img=transformed_numpy, threshold=400, max_value=transformed_numpy.max(), object_type='dark')
plt.imshow(thresh2,cmap = 'gray')