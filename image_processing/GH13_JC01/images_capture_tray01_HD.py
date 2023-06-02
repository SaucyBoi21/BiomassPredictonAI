# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 12:30:08 2022

@author: jcard

Details: Obtains transformed depth image. It does not include image processing. 
It will store an image for RGB and depth plus a csv file with rgbd values. 

HD: High Definition, the aim of this script is to take an RGB image with a greater definition
"""

import cv2
import time
import os
import numpy as np
from time import sleep
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
path_rgb_depth= 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/scripts_pykinectazure/my_scripts/test_imgs'
path_rgb = 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/scripts_pykinectazure/my_scripts/test_hd'
path_trans = 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/scripts_pykinectazure/my_scripts/test_imgs'
path_csv_depth = 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/scripts_pykinectazure/my_scripts/test_imgs' 

# Configure and start the device. 


def config_RGB(c_format,c_resolution,c_fps):
    if c_format == "NV12":
        device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_NV12
    elif c_format == "MJPG":
        device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_MJPG
    else :
        device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        
    if c_resolution == 720:
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    elif c_resolution == 1080:
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
    elif c_resolution == 1440:
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1440P
    elif c_resolution == 1536:
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1536P
    elif c_resolution == 2160:
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_2160P
    elif c_resolution == 3072:
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_3072P
        
    if c_fps == 5:
        device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_5
    elif c_fps == 15:
        device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_15
    elif c_fps == 30:
        device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
    return device_config

# WFOV unbinned (resolution: 1024 x 1024 / FOI: 120 X 120)
# WFOV binned (resolution: 512 x 512 / FOI: 120 X 120)
# NFOV binned (resolution: 320 x 288 / FOI: 75 X 65)
# NFOV unbinned (resolution: 640 x 576 / FOI: 75 X 65)

def config_depth(d_mode,d_format):
    if d_mode == "NFOV_BINNED":
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
    elif d_mode == "NFOV_UNBINNED":
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
    elif d_mode == "WFOV_BINNED":
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
    elif d_mode == "WFOV_UNBINNED":
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_UNBINNED
    elif d_mode == "IR":
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_PASSIVE_IR
        
    if d_format == "DEPTH":
        device_config.depth_format = pykinect.K4A_IMAGE_FORMAT_DEPTH16
    elif d_mode == "IR":
        device_config.depth_format = pykinect.K4A_IMAGE_FORMAT_IR16
    return device_config




# If obtaining a RGB-D image

def capture_image(sensor):
    if sensor == "RGB_D":
        
        calibration = device.get_calibration(device_config.depth_mode, device_config.color_resolution)
        trans_handle = Transformation(calibration._handle)
        capture = device.update()
        depth_ref = capture.get_depth_image_object() #storing depth capture
        transformed_image = trans_handle.depth_image_to_color_camera(depth_ref) 
        ret, transformed_numpy= transformed_image.to_numpy()
        plt.imshow(transformed_numpy, cmap = 'gray')
        device.close()
        return transformed_numpy
        
    elif sensor == "RGB":
        # Get a capture from the device 
        #config_RGB("RGB",1080, 15)
        capture = device.update()
        ret, color_ref = capture.get_color_image()
        #plt.imshow(color_ref)
        device.close()
        return ret,color_ref
        
    elif sensor == "depth":
        capture = device.update()
        ret, depth_color = capture.get_smooth_colored_depth_image()
        plt.imshow(depth_color)
        device.close()
        return depth_color
        
# Initialize libraries
pykinect.initialize_libraries()

# Turn device on:
device_config = pykinect.default_configuration

config_RGB("RGB",1080, 15)
config_depth("WFOV_UNBINNED", "DEPTH")
device = pykinect.start_device(config=device_config)

    
depth_comb = capture_image("RGB_D")

### RGB Capture
pykinect.initialize_libraries()

# Turn device on:
device_config = pykinect.default_configuration
config_RGB("RGB",1080, 15)
device = pykinect.start_device(config=device_config)
rgb_img = capture_image("RGB")


            
    


