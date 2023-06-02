# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 22:46:20 2023
Script to perform image augmentation in a set of images. 

This script reads all the images in the input directory and applies random augmentations to each image.
The augmentations include random rotation, flipping, Gaussian noise, and brightness adjustment. 
The augmented images are saved to the output directory with a _augmented suffix added to the original filename. 

@author: jcard
"""

import os
from skimage import io, transform, util, exposure
from skimage.filters import gaussian
import numpy as np
import random

# Set the path to the directory containing the input images and the output directory
input_dir = 'path/to/input_directory'
output_dir = 'path/to/output_directory'

# Define the image augmentation functions
def random_rotation(image):
    # Generate a random angle between -10 and 10 degrees
    angle = random.uniform(-10, 10)

    # Rotate the image
    rotated_image = transform.rotate(image, angle, preserve_range=True)

    return rotated_image.astype(np.uint8)

def random_flip(image):
    # Flip the image horizontally or vertically with 50% probability
    if random.random() < 0.5:
        flipped_image = np.fliplr(image)
    else:
        flipped_image = np.flipud(image)

    return flipped_image.astype(np.uint8)

def random_gaussian(image):
    # Add random Gaussian noise to the image
    sigma = random.uniform(0, 0.05) * np.max(image)
    noisy_image = util.random_noise(image, mode='gaussian', var=sigma**2)

    return noisy_image.astype(np.uint8)

def random_brightness(image):
    # Adjust the brightness of the image
    gamma = random.uniform(0.5, 2.0)
    bright_image = exposure.adjust_gamma(image, gamma)

    return bright_image.astype(np.uint8)

# Loop through all the images in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load the image
        image = io.imread(os.path.join(input_dir, filename))

        # Perform random image augmentations
        image = random_rotation(image)
        image = random_flip(image)
        image = random_gaussian(image)
        image = random_brightness(image)

        # Save the augmented image to the output directory
        output_filename = os.path.splitext(filename)[0] + '_augmented.jpg'
        io.imsave(os.path.join(output_dir, output_filename), image)
