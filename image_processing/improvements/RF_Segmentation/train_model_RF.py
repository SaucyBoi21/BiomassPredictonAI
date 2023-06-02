import numpy as np
import cv2
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
import glob
from os import chdir
from scipy import ndimage as nd
from plantcv import plantcv as pcv
from skimage.filters import roberts, sobel, scharr, prewitt, meijering, sato, hessian
import sys




path = r"C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/improvements/RF_Segmentation/rgb/"
train_path = r"C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/improvements/RF_Segmentation/masks/"
path = path if path[-1] == "/" else path + "/"
chdir(path)

files = [file for file in glob.glob("*.jpg")]
#files.sort(key=getint)

dfs = pd.DataFrame()
for file in files:  # [20:50]:
    df = pd.DataFrame()
    # print(file)
    img = cv2.imread(path + file)
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
    labeled_img = cv2.imread(train_path + file)
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)

    flat_img = img.reshape(-1)
    flat_labeled_img = labeled_img.reshape(-1)

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

    """    
    # Gabor
    num = 1
    kernels = []
    for theta in range(2):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for lamda in np.arange(0, np.pi, np.pi / 4):
                for gamma in (0.05, 0.5):
                    gabor_label = "Gabor " + str(num)
                    ksize = 5
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                    kernels.append(kernel)
                    fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img
                    num += 1

    # GAUSSIAN
    gaus = nd.gaussian_filter(img, sigma=3)
    gaus = gaus.reshape(-1)
    df["gaus3"] = gaus

    # Feature 9 is Gaussian with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3

    # MEDIAN
    median = nd.median_filter(img, size=3)
    median = median.reshape(-1)
    df["median3"] = median

    # Feature 11 is Variance with size=3
    variance_img = nd.generic_filter(img, np.var, size=3)
    variance_img1 = variance_img.reshape(-1)
    df['Variance s3'] = variance_img1  # Add column to original dataframe

    edges = cv2.Canny(img, 100, 200)  # Image, min and max values
    edges1 = edges.reshape(-1)
    df['Canny Edge'] = edges1  # Add column to original dataframe

    # Feature 4 is Roberts edge
    edge_roberts = roberts(img)
    edge_roberts1 = edge_roberts.reshape(-1)
    df['Roberts'] = edge_roberts1

    # Feature 5 is Sobel
    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel1

    # Feature 6 is Scharr
    edge_scharr = scharr(img)
    edge_scharr1 = edge_scharr.reshape(-1)
    df['Scharr'] = edge_scharr1

    # Feature 7 is Prewitt
    edge_prewitt = prewitt(img)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['Prewitt'] = edge_prewitt1

    edge_meijering = meijering(img)
    edge_meijering = edge_meijering.reshape(-1)
    df['meijering'] = edge_meijering

    edge_sato = sato(img)
    edge_sato = edge_sato.reshape(-1)
    df['sato'] = edge_sato

    edge_hessian = hessian(img)
    edge_hessian = edge_hessian.reshape(-1)
    df['hessian'] = edge_hessian
    """
    # print(dfs)
    df["Labels"] = flat_labeled_img
    dfs = pd.concat([dfs, df], ignore_index=True)
    

# print(dfs)
# Dependent variable
y = dfs["Labels"].values
x = dfs.drop(labels=["Labels"], axis=1)

# spliting data to train and test set

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=20)

# Train the model

model = RandomForestClassifier(n_estimators = 25, random_state = 42)

model.fit(x_train,y_train)

prediction_test = model.predict(x_test)
print(metrics.accuracy_score(y_test,prediction_test))

feature_list = list(x.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp)

# Save model as a picle file
filename = 'Segmentation_model_pavel'
pickle.dump(model,open(filename,'wb'))

# Analysis of the performance of the semantic segmentation
# Add one of the images used for training to test the performance of the segmentation
test = cv2.imread('C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/ML_segmentation/RF_segmentation/pavel_imgs/T01_GH13_JC01_Feb-20-2023_1741_rgb.jpg')

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

test_feat = feature_extraction(test)
result = model.predict(test_feat)
segmented = result.reshape((img.shape[0:2]))

cv2.imshow("Segmented Test Image", segmented)
cv2.waitKey(0)