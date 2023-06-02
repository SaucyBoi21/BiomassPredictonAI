# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 16:25:49 2022

@author: jcard
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:03:49 2022
Script use for image transformation. 
@author: jcard
"""



# Step 1: Import all the folders we will use for: kinect manipulation, image analysis 
# I'm adding time module to include date and time in the images names
import sys
import cv2
import time
import os
import numpy as np
 

#paths where I will store my images (sabe each img type in individual folders):
path_rgb = 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/scripts_pykinectazure/my_scripts/rgb_imgs'
path_depth = 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/scripts_pykinectazure/my_scripts/depth_imgs'
path_transf = 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/scripts_pykinectazure/my_scripts/transf_imgs'

sys.path.insert(1, '../')
import pykinect_azure as pykinect
from pykinect_azure.k4a import _k4a
from pykinect_azure.k4a.configuration import Configuration
from pykinect_azure.k4arecord.record import Record
from pykinect_azure.k4a._k4atypes import K4A_WAIT_INFINITE
from pykinect_azure.k4a.calibration import Calibration
from pykinect_azure.k4a.image import Image
from pykinect_azure.k4a.transformation import Transformation
#this is the function we need to create the transformation. 
# Now we need to retrieve the calibration
#transf_handle = pykinect.Transformation()
#depth_to_color = transf_handle.depth_image_to_color_camera()
    
timestamp = time.strftime("%b-%d-%Y_%H%M")

if __name__ == "__main__":
    
	# Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

# THE FIRST STEP IS TO MAKE AN RGB COLOR PICTURE 
# Specify here the correct format for image allignment: 
	# Modify camera configuration
    
    
    
    
    
    device_config = pykinect.default_configuration
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
    device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_15
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_UNBINNED
    device_config.depth_format = pykinect.K4A_IMAGE_FORMAT_DEPTH16
	# print(device_config)	
    
    device = pykinect.start_device(config=device_config)

    
    
    
## Sensor Calibration. I'm still missing something here. 
    sensorcalibration  = device.get_calibration(device_config.depth_mode, device_config.color_resolution)
    
## Here I need to create my transformation handle using k4a_transpormation_create(), as parameter I should include the calibration 
## This one is not working: pykinect.k4a_transformation_create(sensorcalibration)
## I need to create the transformation_handle(). I have my sensor calibration already. 

##  
    depth_trans = Transformation(sensorcalibration._handle)
    ## trans_handle =  transf.handle()
    ## trans is the transformation_handle I need for k4a.transformation_depth_image_to_color_camera
    
    
## Transformed Image Retrieved


# =============================================================================
# =============================================================================
# # # =============================================================================
# # # k4a::calibration cali=k4a::transformation(cali)；
# # # NativeKinectDevice.start_cameras(&deviceConfig)；
# # # Two lines underneath are just capturing both color and depth
# # k4a::image depth = sensorCapture.get_depth_image();
# # k4a::image tansformedImage==trans.depth_image_to_color_camera(depth)；
# # 
# # # =============================================================================
# =============================================================================
# =============================================================================


    # Start device
    

    while True:
 		
 		# Get capture
         capture = device.update()

 		# Get the color image from the capture
         ret, color_ref = capture.get_color_image()

         if not ret:
             continue
# My depth image doesnt have the format object. 
 		# Get the colored depth // We probably need to get 
         ret, depth_ref = capture.get_depth_image()
         
         
         transformed_image = depth_trans.depth_image_to_color_camera(depth_ref)
         
         cv2.imshow('Transformed Depth to Color Image',depth_ref)

         if cv2.waitKey(1) == ord('s'):
             cv2.imwrite(os.path.join(path_rgb, timestamp + "raw_color.jpg"), color_ref)
         if cv2.waitKey(1) == ord('d'):
             cv2.imwrite(os.path.join(path_depth, timestamp + "raw_depth.jpg"), depth_ref)
         if cv2.waitKey(1) == ord('a'):
             cv2.imwrite(os.path.join(path_transf, timestamp + "transformed_combined.jpg"),transformed_image)
         if cv2.waitKey(1) == ord('q'):
             cv2.destroyAllWindows()
             break
    device_off = device.close()