# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 14:16:32 2022

@author: jcard
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 12:35:53 2022

@author: jcard
"""

# Step 1: Import all the folders we will use for: kinect manipulation, image analysis 
# I'm adding time module to include date and time in the images names
import sys
import cv2
import time
import numpy as np
from plantcv import plantcv as pcv  
from matplotlib import pyplot as plt


#Folder where I will store my images:
path = 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/scripts_pykinectazure/test_imgs'

sys.path.insert(1, '../')

import pykinect_azure as pykinect

timestamp = time.strftime("%b-%d-%Y_%H%M")


if __name__ == "__main__":
    
	# Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

# THE FIRST STEP IS TO MAKE AN RGB COLOR PICTURE 
# Dont forget that RGB FORMAT MATTERS to allign RGB and depth: BGRA32 is the correct format for allignment
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
        name_rgb = timestamp + "_RGB_tray2.jpg"
        if cv2.waitKey(1) == ord('s'):
            cv2.imwrite(name_rgb, color_image)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
        
    device_off = device.close()
    print("RGB Image Completed")
    print(device_config)
# RGB Retrieving COMPLETED
# FROM HERE WE START WITH THE IMAGE PROCESSING
# I need to upload the picture I just saved.


img, path, img_filename = pcv.readimage(filename = "./Oct-11-2022_1516_RGB_tray1.jpg", mode="rgb")
pcv.plot_image(img)

## Add pipeline from JUPYTER once is over. Pipeline is working but images are overlapping already and algae is 
## influencing the segmentation as well. 

crop = pcv.crop(img=img, x=200, y=75, h=720, w=1500)
pcv.plot_image(crop)

s_img = pcv.rgb2gray_hsv(crop, channel='s')
pcv.plot_image(s_img)

thresh = pcv.threshold.binary(gray_img=s_img, threshold=60, max_value=255, object_type='light')
pcv.plot_image(thresh)


    

