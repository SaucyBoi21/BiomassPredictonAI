# -*- coding: utf-8 -*-
"""
Created on Tue May 30 21:58:48 2023

@author: jcard
"""

import cv2

def crop_image(image):
    height, width = image.shape[:2]
    section_width = width // 3

    # Crop the first section
    section1 = image[:, 0:section_width]

    # Crop the second section
    section2 = image[:, section_width:2*section_width]

    # Crop the third section
    section3 = image[:, 2*section_width:width]

    return section1, section2, section3

# Load the image
image_path = 'path_to_your_image.jpg'
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