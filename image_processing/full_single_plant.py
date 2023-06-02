# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:52:53 2023

@author: jsc00615
"""
import os
import pickle
from sklearn.ensemble import RandomForestClassifier#,forest
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import glob
import numpy as np
from plantcv import plantcv as pcv
from scipy import stats

# A required callback method that goes into the trackbar function.
def nothing(x):
    pass
np.set_printoptions(threshold=np.inf)

# give img location use "/" instead of "\"
path_rgb = 'C:/Users/jsc00615/OneDrive - University of Georgia/image_processing/GH13_JC01/analysis_tray1/imgs_folder/T01_GH13_JC01_Feb-23-2023_0729_rgb.jpg'
path_csv = 'C:/Users/jsc00615/OneDrive - University of Georgia/image_processing/GH13_JC01/analysis_tray1/imgs_folder/T01_GH13_JC01_Feb-23-2023_0729_depth_values.csv'
#path_imgs = 'C:/Users/jcard/OneDrive - University of Georgia/side_projects/seedlings_Ferrarezi_Lab/2nd_trial/original_imgs/cropped/final_img'
#path_result = 'C:/Users/jcard/OneDrive - University of Georgia/side_projects/seedlings_Ferrarezi_Lab/2nd_trial/original_imgs/cropped/results_csv'

#DEBUG = False

def get_centeroid(cnt):
    length = len(cnt)
    sum_x = np.sum(cnt[..., 0])
    sum_y = np.sum(cnt[..., 1])
    return (int(sum_x / length), int(sum_y / length))

def get_y(coord):
    c_no,centerx,centery = coord
    return centerx

def generate_border(image, border_size=5, n_erosions=1):
    erosion_kernel = np.ones((3, 3), np.uint8)  ## Start by eroding edge pixels
    eroded_image = cv2.erode(image, erosion_kernel, iterations=n_erosions)

    ## Define the kernel size for dilation based on the desired border size (Add 1 to keep it odd)
    kernel_size = 2 * border_size + 1
    dilation_kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Kernel to be used for dilation
    dilated = cv2.dilate(eroded_image, dilation_kernel, iterations=1)
    # plt.imshow(dilated, cmap='gray')

    ## Replace 255 values to 127 for all pixels. Eventually we will only define border pixels with this value
    dilated_127 = np.where(dilated == 255, 127, dilated)

    # In the above dilated image, convert the eroded object parts to pixel value 255
    # What's remaining with a value of 127 would be the boundary pixels.
    original_with_border = np.where(eroded_image > 127, 255, dilated_127)

    # plt.imshow(original_with_border,cmap='gray')

    return original_with_border





depth_img = np.array(pd.read_csv(path_csv))
#mask = cv2.imread(r"height_harvest/"+file.split("\\")[-1])
img = cv2.imread(path_rgb)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



l_h, l_s, l_v, u_h, u_s, u_v,size_ig = 0,0,0,255,255,255,0
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
    cv2.imshow('Trackbars', cv2.resize(stacked, None, fx=0.3, fy=0.3))

    # If the user presses ESC then exit the program
    key = cv2.waitKey(1)
    if key == 27:
        break

    # If the user presses `s` then print this array.
    if key == ord('s'):
        thearray = [[l_h, l_s, l_v], [u_h, u_s, u_v]]
        print(thearray)

        # Also save this array as penval.npy
        #np.save('hsv_value', thearray)
        break


# Release the camera & destroy the windows.
cv2.destroyAllWindows()

pcv.plot_image(mask)



blur = cv2.GaussianBlur(mask,(11,11),0)

threshold = 100
blur[blur >= threshold] = 255
blur[blur < threshold] = 0
plt.imshow(blur)
#blur = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

cont, _ = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # search for contours

# sort the contours in the image
centers = []
for c in cont:
    centerx,centery = get_centeroid(c)
    centers.append([centery,centerx])
centers = sorted(centers)    

rows = []
px = 50
for c_no,(centery,centerx) in enumerate(centers):
    row = []
    row.append([c_no,centerx,centery])
    for c_no2,(centery2,centerx2) in enumerate(centers):
        if [centery,centerx] != [centery2,centerx2] and centery-px <= centery2 <= centery+px:
            if [c_no,centerx2,centery2] not in row:
                row.append([c_no2,centerx2,centery2])
    row = sorted(row,key=get_y)
    for i in row:
        if i not in rows:
            rows.append(i)

order = []
for no, c in enumerate(cont):
    for n, row in enumerate(rows):
        centerx,centery = get_centeroid(c)
        if [centerx,centery] == row[1:3]:
            order.append(n)
            
cont = [x for _,x in sorted(zip(order,cont))]
    

# Calculation of properties of the contours
areas, perimeters, heights, refs, maxes, avgs = [], [], [], [], [],[]
for c in cont:
    if cv2.contourArea(c) < 0.1*  cv2.contourArea(max(cont,key=cv2.contourArea)):
        blur = cv2.fillPoly(blur, pts=[c], color=(0, 0, 0))
    else:
        # Descriptors
        area, perimeter = cv2.contourArea(c), cv2.arcLength(c,True)
        areas.append(area)
        perimeters.append(perimeter)
        
        background = np.zeros(img.shape[0:2])
        
        # Height values
        #
        one_cont_img = cv2.fillPoly(background, pts=[c], color=255) 
        bordered_mask = generate_border(one_cont_img,10,0)
        plt.imshow(bordered_mask)
        ref_height = depth_img[np.where(bordered_mask == 127)]
        #max_height = ref_height.max()
        ref_height = [i for i in ref_height if i != 0]
        plant_height_max = max(ref_height)-min([i for i in depth_img[np.where(bordered_mask == 255)] if i != 0])
        plant_height_avg = (np.median(ref_height))-min([i for i in depth_img[np.where(bordered_mask == 255)] if i != 0])
        heights.append(plant_height_max)
        maxes.append(plant_height_max)
        avgs.append(plant_height_avg)
        
        refs.append(ref_height)
        
        # RGB values
        cv2.drawContours(background, [c], -1, 255, cv2.FILLED)  
        img_mask = rgb[np.where(background == 255)]
        RGB = np.mean(img_mask, axis=0).astype(int)
        total_plants = len(areas)
        
heights_mm = [num / 10 for num in heights]         
data_set = pd.DataFrame()
data_set.insert(0, 'area', areas)
data_set.insert(0, 'perimeter', perimeters)
data_set.insert(0, 'height_mm', heights_mm)

data_set.to_csv(f'csv_data/data_{name_img}.csv')
        #plt.imshow(one_cont_img)
        
        # plant = [i for i in depth_img[np.where(bordered_mask == 255)] if i != 0]
        # plt.hist(plant,bins = 50)            

# plt.hist(areas,5)
plt.imshow(blur)
    
    # overlap = np.zeros(rgb.shape)
    # blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
    # overlap = cv2.bitwise_and(blur, img) # removes the black dots, idk why 
    # cv2.imshow("",overlap)
    # cv2.waitKey(0)
