# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 13:45:41 2022

@author: jcard

TAKING A PICTURE WITHOUT PREVIEW. 
"""

import sys
import cv2
import numpy

sys.path.insert(1, '../')
import pykinect_azure as pykinect

if __name__ == "__main__":

	# Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

	# Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_2160P
 

	# print(device_config)

	# Start device
    device = pykinect.start_device(config=device_config)
    cv2.namedWindow('RGB_Kinect',cv2.WINDOW_NORMAL)
    while True:
        capture=device.update()
        ret,color_image = capture.get_color_image()
        
     #Plot the image
     # Use command "s" to make a new frame from the video displayed
     # Use q to close the preview and store variables
        cv2.imshow('RGB_Kinect',color_image)
        if cv2.waitKey(1) == ord('s'):
             cv2.imwrite("RGB_2.jpg",color_image)
        if cv2.waitKey(1) == ord('q'):
             cv2.destroyAllWindows()
             break
    # print description of image properties (resolution, fps)
            

 
		
		
       
        
    