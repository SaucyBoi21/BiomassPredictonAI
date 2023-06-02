# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 21:22:17 2023
Random Forest for Segmentation
Based on: Tutorial by Digital Sreeni
Source: https://www.youtube.com/watch?v=6yW31TT6-wA
@author: jcard
"""

import pandas as pd
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
import glob
from os import chdir
from scipy import ndimage as nd
from skimage.filters import roberts, sobel, scharr, prewitt, meijering, sato, hessian
import sys
from plantcv import plantcv as pcv


# Here we need to loop through images. This function will return a dataframe with all the features extracted from one single image.  
def feature_extraction(img):

# FEATURE ENGINEERING
## Create an empty dataframe. 

   # [20:50]:
    df = pd.DataFrame()
    
    #img = cv2.imread(r"C:\Users\klimesp\Dropbox\Programovani\Python\USA\Images\done/Final/T01_GH13_JC01_Feb-17-2023_0743_rgb.jpg")

    R_0 = img[:,:,0].reshape(-1)
    df["R_0"] = R_0
    
    G_1 = img[:,:,1].reshape(-1)
    df["G_1"] = G_1
    
    B_2 = img[:,:,2].reshape(-1)
    df["B_2"] = B_2


    HSV_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H = HSV_img[:,:,0].reshape(-1)
    df["H"] = H
    S = HSV_img[:,:,1].reshape(-1)
    df["S"] = S


    LAB_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    A = LAB_img[:,:,1].reshape(-1)
    df["A"] = A
    Bb = LAB_img[:,:,2].reshape(-1)
    df["Bb"] = Bb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    flat_img = img.reshape(-1)


    edge_sato = sato(img)
    edge_sato = edge_sato.reshape(-1)
    df['sato'] = edge_sato
    
    edge_meijering = meijering(img)
    edge_meijering = edge_meijering.reshape(-1)
    df['meijering'] = edge_meijering
    
    # Feature 11 is Variance with size=3
    variance_img = nd.generic_filter(img, np.var, size=3)
    variance_img1 = variance_img.reshape(-1)
    df['Variance s3'] = variance_img1  # Add column to original dataframe

    # Add orifinal pixel value
    flat_img = img.reshape(-1)
    df["Original Image"] = flat_img
 
    return df
## So far we have just added columns for our predictors/ features. Now, add values for our labels. 
 
import glob
import pickle
from matplotlib import pyplot as plt

filename = 'Segmentation_model_pavel'
load_model = pickle.load(open(filename,'rb'))


path = 'images/imgs_predict/*.jpg'
for file in glob.glob(path):
    img = cv2.imread(file)
    #img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    print(img)
    
    X = feature_extraction(img)
    result = load_model.predict(X)
    segmented = result.reshape((img.shape[0:2]))
    #plt.imshow(segmented)
    name= file.split("T0")
    plt.imsave('C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/ML_segmentation/RF_segmentation/images/segmented_imgs/'+ name[1], segmented,cmap ="jet")

pcv.plot_image(segmented)
histogram, bin_edges = np.histogram(segmented, bins= 255, range=(250, 255))
fig, ax = plt.subplots()
plt.plot(bin_edges[0:-1], histogram)

thresh = pcv.threshold.binary(gray_img=segmented, threshold=100, max_value=255, object_type='light')
pcv.plot_image(thresh)
histogram, bin_edges = np.histogram(thresh, bins= 255, range=(250, 255))
fig, ax = plt.subplots()
plt.plot(bin_edges[0:-1], histogram)



size = 20
analysis_img,_,_ =  pcv.readimage(filename= 'C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/GH13_JC01/rgb_imgs/tray_1/analyzed_imgs/T01_GH13_JC01_Feb-23-2023_0729_rgb.jpg_analysis.jpg', mode="rgb")
depth_img = pd.read_csv('C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/GH13_JC01/depth_csv/harvest_reference/depth_csv/T01_GH13_JC01_Feb-23-2023_0729_depth_values.csv')
#### Compare original images with masks
fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows =2,ncols=2,figsize=(10,7))
ax1.set_title('Original RGB Image\n  (Pixel Intensity)',fontsize = size)
ax1.set_axis_off()
ax1.imshow(img)
ax2.set_title('Depth Image\n (Each Pixel in mm)',fontsize = size)
ax2.set_axis_off()
ax2.imshow(depth_img,cmap='gray')
ax3.set_title('Segmentation using RF',fontsize = size)
ax3.set_axis_off()
ax3.imshow(segmented)
ax4.set_title('Object Shape Analysis',fontsize = size)
ax4.set_axis_off()
ax4.imshow(analysis_img)
plt.savefig('C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/three_img.jpg', dpi = "200")
plt.show()


