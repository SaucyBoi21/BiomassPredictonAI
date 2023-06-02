# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 21:22:17 2023
Random Forest for Segmentation
Based on: Tutorial by Digital Sreeni
Source: https://www.youtube.com/watch?v=6yW31TT6-wA
@author: jcard
"""

import numpy as np
import cv2
import pandas as pd
from plantcv import plantcv as pcv


path = 'images/imgs_predict/*.jpg'
for file in glob.glob(path):
    
img = cv2.imread('new_images/T02_GH13_JC01_Feb-23-2023_1205_rgb.jpg')
pcv.plot_image(img)

# I need to convert the image into gray (SINGLE CHANNEL NOT 3)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# FEATURE ENGINEERING
## Create an empty dataframe. 
df = pd.DataFrame()

# Feature 1: Correspond to the intensity of our original pixels itself. 
## We haave a 2 dimensional image, so we need to unwrapped it. Img 2 dimenstion is the multiplication of my 2d image dimensions. 
img2 = img.reshape(-1)
df["Original_Pixel"] = img2

#  Add additional Features
## FIRST SET - Gabor Features: 
num =1
kernels = []
for theta in range(2):
    theta = theta/4. * np.pi
    for sigma in(1,3):
        for lamda in np.arange(0,np.pi,np.pi/4):  # Range of wavelenghts
            for gamma in (0.05,0.5):
                gabor_label = 'Gabor' + str(num) # Labelling Gabor column as Gabor
                ksize = 9
                kernel = cv2.getGaborKernel((ksize,ksize), sigma, theta, lamda, gamma,0,ktype = cv2.CV_32F)
                kernels.append(kernel)
                # Now
                fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                filtered_img = fimg.reshape(-1)
                df[gabor_label] = filtered_img
                num += 1
###########################################################################


## Additional Filter Extractions

# Canny edge --> Edge detection 
# I'm using RGB original image here. 
edges = cv2.Canny(img, 100, 200)
#cv2.imshow("image",edges)
#cv2.waitKey(0)
edges1 = edges.reshape(-1)

from skimage.filters import roberts,sobel, scharr, prewitt

#Feature 4 is Roberts edge
edge_roberts = roberts(img)
edge_roberts1 = edge_roberts.reshape(-1)
df['Roberts'] = edge_roberts1

#Feature 5 is Sobel
edge_sobel = sobel(img)
edge_sobel1 = edge_sobel.reshape(-1)
df['Sobel'] = edge_sobel1

#Feature 6 is Scharr
edge_scharr = scharr(img)
edge_scharr1 = edge_scharr.reshape(-1)
df['Scharr'] = edge_scharr1

#Feature 7 is Prewitt
edge_prewitt = prewitt(img)
edge_prewitt1 = edge_prewitt.reshape(-1)
df['Prewitt'] = edge_prewitt1

#Feature 8 is Gaussian with sigma=3
from scipy import ndimage as nd
gaussian_img = nd.gaussian_filter(img, sigma=3)
gaussian_img1 = gaussian_img.reshape(-1)
df['Gaussian s3'] = gaussian_img1

#Feature 9 is Gaussian with sigma=7
gaussian_img2 = nd.gaussian_filter(img, sigma=7)
gaussian_img3 = gaussian_img2.reshape(-1)
df['Gaussian s7'] = gaussian_img3

#Feature 10 is Median with sigma=3
median_img = nd.median_filter(img, size=3)
median_img1 = median_img.reshape(-1)
df['Median s3'] = median_img1

#Feature 11 is Variance with size=3
variance_img = nd.generic_filter(img, np.var, size=3)
variance_img1 = variance_img.reshape(-1)
df['Variance s3'] = variance_img1  #Add column to original dataframe


## So far we have just added columns for our predictors/ features. Now, add values for our labels. 
labeled_img = cv2.imread("new_masks/T02_GH13_JC01_Feb-23-2023_1205_rgb.jpg")
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
labeled_img1 = labeled_img.reshape(-1)
df['labels'] = labeled_img1

###################################################################

# Dependent variable --> Label (thats what we are trying to predict) in this case a class. 

Y= df['labels'].values


# Lets define X now. 

X = df.drop(labels = ['labels'],axis = 1)

# Split data into test and train:
    
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test =train_test_split(X,Y, test_size =0.4,random_state=20)

# Import ML algorithm and train the model

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10,random_state = 42)

model.fit(X_train,Y_train)

# Lets calculate the accuracy of this model. 
# Lets see how it works in our testing data firs

prediction_train = model.predict(X_test)

from sklearn import metrics
print("Accuracy = ", metrics.accuracy_score(Y_test, prediction_train))


##  WHICH FEATURES CONTRIBUTED THE MOST TOWARDS OUR MODEL

#importances = list(model.feature_importances_)
features_list = list(X.columns)
features_imp = pd.Series(model.feature_importances_,index = features_list).sort_values(ascending=False)
print(features_imp)



### IMPORTANT: Saving the trained model.Use it for future 
# Pickle it.
import pickle

filename = 'SEGMENTATION_model'
pickle.dump(model,open(filename,'wb'))


# Now lets load it and make some predictions.
## READ BINARY MODE (rb) 
load_model = pickle.load(open(filename,'rb'))
result = load_model.predict(X)

segmented = result.reshape((img.shape))

from matplotlib import pyplot as plt

plt.imshow(segmented)

