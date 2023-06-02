# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 22:45:31 2022
The purpose of this script is to generate a csv file with the 
geometrical description of each germination tray.
It will include each parameter per plant, a plant count and data summary.

GET GERMINATION RATE FROM THIS FILE. IT WILL BE SHOWN ON THE RESULTS CSV. 
GERMINATION RATE:
    - rois will hold all the positions available for seeds (according to the grid)/ 
    - plant id: Plant selection 
(plant_id /len(rois))

@author: jcard
"""
from os import chdir,remove,mkdir
import cv2
import matplotlib.pyplot as plt
import numpy as np 
import glob
from matplotlib import pyplot as plt
from plantcv import plantcv as pcv
#from plotnine import ggplot, aes, geom_point
import pandas as pd
#import seaborn as sns

# A required callback method that goes into the trackbar function.
def nothing(x):
    pass
# REMEMBER: We are plotting in console using PCV. Alternatives are: plt.imshow(img) or 
# TO SAVE IMAGES we are using plantcv.print_image(img, filename) however we could use Python: cv2.imwrite(filename, img[, params])

#Options
mypath = r"C:/Users/jsc00615/OneDrive - University of Georgia/image_processing/GH13_JC01/rgb_imgs/tray_3/raw_images"

mypath = mypath if mypath[-1] == "/" else mypath + "/"
chdir(mypath)

#files = [file for file in glob.glob("*.png")]
files = [file for file in glob.glob("*.jpg")]
l_h, l_s, l_v, u_h, u_s, u_v,size_ig = 0,0,0,255,255,255,0

for id,file in enumerate(files):
    #img_name to avoid changing name in every path or when storing results  
    img_name = '_'.join(file.split('.')[0:1])
    specie = '_'.join(file[1].split('_')[0:1])
    substrate = '_'.join(file[1].split('_')[1:2]).split('.')[0]
    total_seeds = '_'.join(file[1].split('_')[2:]).split('.')[0]
    
    # substrate = '_'.join(img_name.split('_')[1:])
    
   
    if 1==1:
        img,_,_ = pcv.readimage(mypath + file, mode="native")
   
  
    
    #if img.shape[0] < img.shape[1]:
        #img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    cap = img
    frame = cap
    
    
    # Create a window named trackbars.
    cv2.namedWindow("Trackbars")
    
    # Now create 6 trackbars that will control the lower and upper range of
    # H,S and V channels. The Arguments are like this: Name of trackbar,
    # window name, range,callback function. For Hue the range is 0-179 and
    # for S,V its 0-255.
    cv2.createTrackbar("L - H", "Trackbars", l_h, 255, nothing)
    cv2.createTrackbar("L - S", "Trackbars", l_s, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", l_v, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", u_h, 255, nothing)
    cv2.createTrackbar("U - S", "Trackbars", u_s, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", u_v, 255, nothing)
    cv2.createTrackbar("Size", "Trackbars", size_ig, 1000, nothing)
    
    while True:
    
    
        # Convert the BGR image to HSV image.
        hsv = cv2.cvtColor(cap, cv2.COLOR_BGR2LAB)
    
        # Get the new values of the trackbar in real time as the user changes
        # them
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")
        size_ig = cv2.getTrackbarPos("Size", "Trackbars")
    
        # Set the lower and upper HSV range according to the value selected
        # by the trackbar
        lower_range = np.array([l_h, l_s, l_v])
        upper_range = np.array([u_h, u_s, u_v])
    
        # Filter the image and get the binary mask, where white represents
        # your target color
        mask = cv2.inRange(hsv, lower_range, upper_range)
        cont, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in cont:
            if cv2.contourArea(contour) < size_ig:
                cv2.fillPoly(mask, pts=[contour], color=(0, 0, 0))
    
        # You can also visualize the real part of the target color (Optional)
        res = cv2.bitwise_and(frame, frame, mask=mask)
    
        # Converting the binary mask to 3 channel image, this is just so
        # we can stack it with the others
        mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
        # stack the mask, orginal frame and the filtered result
        stacked = np.hstack((mask_3, frame, res))
    
        # Show this stacked frame at 40% of the size.
        cv2.imshow('Trackbars', cv2.resize(stacked, None, fx=0.1, fy=0.1))
    
        # If the user presses ESC then exit the program
        key = cv2.waitKey(1)
        if key == 27:
            break
    
        # If the user presses `s` then print this array.
        if key == ord('s'):
            thearray = [[l_h, l_s, l_v], [u_h, u_s, u_v]]
            print(thearray)
            pcv.print_image(mask,f'C:/Users/jcard/OneDrive - University of Georgia/side_projects/seedlings_Ferrarezi_Lab/2nd_trial/original_imgs/cropped/mask_cropped/{img_name}.jpg')
    
            # Also save this array as penval.npy
            #np.save('hsv_value', thearray)
            break
    
    
    # Release the camera & destroy the windows.
    cv2.destroyAllWindows()

# =============================================================================
#  THIS IS PLANTCV histogram option. Creates a ggplot object. 
#hist2 = pcv.visualize.histogram(img=crop,hist_data=True)
# =============================================================================
pcv.print_image(mask,f'C:/Users/jcard/OneDrive - University of Georgia/side_projects/seedlings_Ferrarezi_Lab/2nd_trial/original_imgs/cropped/mask_cropped/{img_name}.jpg')

fig,(ax1,ax2) = plt.subplots(nrows =1,ncols=2, figsize =(7,7))
ax1.set_title('Original RGB')
ax1.set_axis_off()
ax1.imshow(img)
ax2.set_title('Mask: Colorspace Lab a')
ax2.set_axis_off()
ax2.imshow(mask)
plt.show()

