# -*- coding: utf-8 -*-
"""
Created on Tue May 30 21:58:48 2023

@author: jcard
"""

import cv2

def crop_image(image):
    height, width = image.shape[:2]
    section_width = width // 3
    section_height = height//4

    # Crop the first section
    section1 = image[0:section_height, 0:section_width]

    # Crop the second section
    section2 = image[section_height:2*section_height,section_width:2*section_width]

    # Crop the third section
    section3 = image[2*section_height:height, 2*section_width:width]

    return section1, section2, section3

# Load the image
image_path = 'Final/T01_GH13_JC01_Feb-01-2023_0749_rgb.jpg'
image = cv2.imread(image_path)

# Resize the image to 1920x1080 if needed
image = cv2.resize(image, (1920, 1080))

# Crop the image into three sections
section1, section2, section3 = crop_image(image)

# Display the cropped sections
cv2.imshow('Section 1', section1)
cv2.imshow('Section 2', section2)
cv2.imshow('Section 3', section3)
cv2.waitKey(0)
cv2.destroyAllWindows()


img = cv2.imread(image_path)
 
# cv2.imread() -> takes an image as an input
h, w, channels = img.shape
 
half = w//
 
 
# this will be the first column
left_part = img[:, :half]
 
# [:,:half] means all the rows and
# all the columns upto index half
 
# this will be the second column
right_part = img[:, half:] 
 
# [:,half:] means all the rows and all
# the columns from index half to the end
# cv2.imshow is used for displaying the image
cv2.imshow('Left part', left_part)
cv2.imshow('Right part', right_part)
 
# this is horizontal division
half2 = h//2
 
top = img[:half2, :]
bottom = img[half2:, :]
 
cv2.imshow('Top', top)
cv2.imshow('Bottom', bottom)