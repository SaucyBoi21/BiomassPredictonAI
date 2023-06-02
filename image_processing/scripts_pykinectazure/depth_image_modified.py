# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:04:45 2022

@author: jcard
"""

import sys
import cv2
import matplotlib.pyplot as plt
import plantcv as pcv
import numpy as np


sys.path.insert(1, '../')
import pykinect_azure as pykinect



if __name__ == "__main__":

	# Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()
    
    	# Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
    
    	# Start device
    device = pykinect.start_device(config=device_config)
    
    cv2.namedWindow('Depth Image',cv2.WINDOW_NORMAL)
    while True:
    
    		# Get capture
            capture = device.update()
        # Format = array of uint16: Unsigned integer (0 to 65535). Size 512x512
    		# Get the color depth image from the capture
            ret, depth_image = capture.get_depth_image()

#if not ret:
#continue
			
		# Plot the image
            cv2.imshow('Depth Image',depth_image)
		
		# Press q key to stop
            if cv2.waitKey(1) == ord('s'):
                cv2.imwrite("raw_depth_tray2.jpg", depth_image)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break
# This will store the image in (512,512,3 size) UINT8 Format what we need to store is the raw picture (512,512) of UINT16 Format
raw_depth = cv2.imread("./raw_depth_tray2.jpg")

depth_image.ndim
depth_image.shape
depth_image.size

#here im fixing column 6 and moving through rows. 
print(depth_image[209:220,6])
print(np.max(depth_image))
print(np.min(depth_image))

#converting all values from mm to cm. Divide each pixel by 10
depth_cm = depth_image / 10
print(np.max(depth_cm))

# Creating plot
fig = plt.figure(figsize =(10, 7))
 
# Is this histogram considering all my pixels inside my 512x512 array
plt.hist(depth_image, bins = [0, 250, 500, 750,
                    1000, 1250, 1500, 1750,
                    2000, 2250, 2500])


# Creating histogram to check all the values being considered:

np.histogram(depth_image, bins = [0, 10, 20, 30, 40,
                        50, 60, 70, 80, 90,
                        100])