# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 14:02:33 2022

@author: jcard
"""

import sys
import cv2
import os
import numpy as np
import time

sys.path.insert(1, '../')
import pykinect_azure as pykinect

path_transf = 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/scripts_pykinectazure/my_scripts/transf_imgs'

timestamp = time.strftime("%b-%d-%Y_%H%M")

if __name__ == "__main__":

	# Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

	# Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
    device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_15
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_UNBINNED
	# print(device_config)

	# Start device
    device = pykinect.start_device(config=device_config)

    cv2.namedWindow('Transformed Color Image',cv2.WINDOW_NORMAL)
    while True:
		
		# Get capture
        capture = device.update()

		# Get the color image from the capture
        ret, color_ref = capture.get_transformed_color_image()

        if not ret:
            continue

		# Get the colored depth // We probably need to get 
        ret, depth_ref = capture.get_colored_depth_image()

		# Combine both images
        combined_image = cv2.addWeighted(color_ref[:,:,:3], 0.7, depth_ref, 0.3, 0)
	
		# Overlay body segmentation on depth image
        cv2.imshow('Transformed Color Image',combined_image)
		
		# Press q key to stop
        if cv2.waitKey(1) == ord('s'):
            cv2.imwrite(os.path.join(path_transf, timestamp + "transformed_color.jpg"), color_ref)
        if cv2.waitKey(1) == ord('d'):
            cv2.imwrite(os.path.join(path_transf, timestamp + "transformed_depth.jpg"), depth_ref)
        if cv2.waitKey(1) == ord('a'):
            cv2.imwrite(os.path.join(path_transf, timestamp + "transformed_combined.jpg"), combined_image)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
    device_off = device.close()

# Here we are extracting just one dimension of our dataframe        
np.savetxt("depth_ref_singledimension.csv", color_ref[4], delimiter=",")