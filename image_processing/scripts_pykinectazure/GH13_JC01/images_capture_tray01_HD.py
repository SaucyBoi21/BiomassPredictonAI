# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 12:30:08 2022

@author: jcard

Details: Obtains transformed depth image. It does not include image processing. 
It will store an image for RGB and depth plus a csv file with rgbd values. 

HD: High Definition, the aim of this script is to take an RGB image with a greater definition
"""
import sys
import cv2
import time
import open3d as o3d
import os
import PIL as pil
import numpy as np
from matplotlib import pyplot as plt
import pykinect_azure as pykinect
from pykinect_azure.k4a import _k4a
from pykinect_azure.k4a.calibration import Calibration
from pykinect_azure.k4a.image import Image
from pykinect_azure.k4a.transformation import Transformation
from pykinect_azure.k4a.capture import Capture

# Set timestamp for image file name metadata. Each picture contains date and time on its filename. 
# exp code: specie_cultivarcode
tray_ID = 'T01_'
exp_code = 'GH13_JC01_'
timestamp = time.strftime("%b-%d-%Y_%H%M")

# Folders where images and csv with features will be stored
path_rgb_depth= 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/scripts_pykinectazure/my_scripts/rgb_imgs/tray_1'
path_rgb = 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/scripts_pykinectazure/my_scripts/rgb_imgs_hd/tray_1'
path_trans = 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/scripts_pykinectazure/my_scripts/transf_imgs/tray_1'
path_csv_depth = 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/scripts_pykinectazure/my_scripts/depth_csv/tray_1' 

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
cv2.imwrite(os.path.join(path_trans, exp_code + tray_ID + timestamp + "_transformed.jpg"), transformed_numpy)
# =============================================================================
# 
# cv2.imshow('Transformed Image',transformed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# =============================================================================

np.savetxt(os.path.join(path_csv_depth,tray_ID + exp_code + timestamp + "_depth_values.csv"), transformed_numpy, delimiter=",")

# Here we update device capture to be able to take an RGB image simultaneously. 
capture = device.update()
ret, color_ref = capture.get_color_image()
plt.imshow(color_ref)
cv2.imwrite(os.path.join(path_rgb_depth,tray_ID + exp_code + timestamp + "_rgb_depth.jpg"), color_ref)

device_off = device.close()

####################################################################################
##### HIGH DEFINITION RGB IMAGE #####

pykinect.initialize_libraries()
device_config = pykinect.default_configuration
device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_2160P
device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_15
#device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_UNBINNED
#device_config.depth_format = pykinect.K4A_IMAGE_FORMAT_DEPTH16

device = pykinect.start_device(config=device_config)

capture = device.update()
ret, rgb = capture.get_color_image()
plt.imshow(rgb)
cv2.imwrite(os.path.join(path_rgb,tray_ID + exp_code + timestamp + "_rgb.jpg"), rgb)

device_off = device.close()
