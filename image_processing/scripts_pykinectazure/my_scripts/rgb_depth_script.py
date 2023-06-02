# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 12:35:53 2022

@author: jcard

Details: Retrieves and stores RGB and depth image. This script does not provide
the transformed depth image, just the raw depth image. 

For transformed deph image: retrieving_manual.py or segmentation_depth.py

Will be used mainly to check distances homogeneity. Preview.
"""

# Step 1: Import all the folders we will use for: kinect manipulation, image analysis 
# I'm adding time module to include date and time in the images names
import sys
import cv2
import time
import os
import numpy as np
 

#paths where I will store my images (sabe each img type in individual folders):
path_rgb = 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/scripts_pykinectazure/my_scripts/test_position_rgb'
path_depth = 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/scripts_pykinectazure/my_scripts/test_position_depth'

sys.path.insert(1, '../')
import pykinect_azure as pykinect

timestamp = time.strftime("%b-%d-%Y_%H%M")

if __name__ == "__main__":
    
	# Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

# THE FIRST STEP IS TO MAKE AN RGB COLOR PICTURE 
# Specify here the correct format for image allignment: 
	# Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    
    
	# print(device_config)

	# Start device
    device = pykinect.start_device(config=device_config)

    cv2.namedWindow('Color Image Preview',cv2.WINDOW_NORMAL)
    while True:

		# Get capture
        capture = device.update()

		# Get the color image from the captureq
        ret, color_image = capture.get_color_image()

        if not ret:
            continue
			
		# Plot the image
        cv2.imshow("Color Image Preview",color_image)
		
#Here we will store that image for future analysis and save it as a variable to perform the analysis in the script        
		# Press q key to stop
		# Press q key to stop
        if cv2.waitKey(1) == ord('s'):
            cv2.imwrite(os.path.join(path_rgb, timestamp + "_RGB_whiteraft_50inchWD.jpg"), color_image)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
        
    device_off = device.close()
    print("RGB Image Completed")
    print(device_config)
# RGB Retrieving COMPLETED

# DEPTH IMAGE CONFIGURATION STARTS HERE
    	# Modify camera configuration to retrieve DEPTH IMAGE 
    # Depth mode WFOV UNBINNED IS A 1024 X 1024 array of uint16 (it doesnt work with 30 FPS)
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
    device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_15
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_UNBINNED
    
    	# Start device
    device = pykinect.start_device(config=device_config)
    cv2.namedWindow('Depth Image Preview',cv2.WINDOW_NORMAL)
    while True:
    
    		# Get capture
            capture = device.update()
        # Format = array of uint16: Unsigned integer (0 to 65535). Size 512x512
    		# Get the color depth image from the capture
            ret, depth_image = capture.get_depth_image()

#if not ret:
#continue
			
		# Plot the image
            cv2.imshow('Depth Image Preview',depth_image)
		# Press q key to stop
            if cv2.waitKey(1) == ord('s'):
                cv2.imwrite(os.path.join(path_depth, timestamp + "_depth_whiteraft_50inchWD.jpg"), depth_image)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break
            
    #turn camera off after taking the imageqq
    device_off = device.close()
    print("Depth Image Completed")
    print(device_config)
    
    
# numpy.savetxt saves an array to a text file.
# np.savetxt(timestamp + "depth_values_comparison.csv", depth_image, delimiter=",")

# This function let us load data from a text or csv file, specify delimiter    
# depth_values = np.genfromtxt("./Oct-11-2022_1531depth_values_tray2_NFOV_UNBINNED.csv",delimiter = ",")
