# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 12:39:48 2022
Test script for doing operations in depth image raw data (array of uint16)
@author: jcard
"""
import matplotlib.pyplot as plt
import plantcv as pcv
import cv2
import numpy as np
import sys


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
                cv2.imwrite("raw_depth_t1_11-10.jpg", depth_image)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break
            
    #turn camera off after taking the image
    device_off = device.close()
# This will store the image in (512,512,3 size) UINT8 Format what we need to store is the raw picture (512,512) of UINT16 Format
raw_depth = cv2.imread("./raw_depth_t1_11-10.jpg")


# numpy.savetxt saves an array to a text file.
np.savetxt("depth_info_tray1_11-10.csv", depth_image, delimiter=",")


#Open the csv file previously stored
with open("depth_info_tray1_11-10.csv") as file_name:
    array = np.loadtxt(file_name, dtype = "uint16",delimiter=",")
    
#new_type = array.astype('uint16')
    
#print()

# numpy.savetxt saves an array to a text file.
#np.savetxt("depth_info.csv", depth_image, delimiter=",")

#histo = numpy.histogram(depth_image, bins=512, range= (0,65535))
#plt.plot(histo)
#plt.hist(depth_image.flatten(),range=(0, 1000))



