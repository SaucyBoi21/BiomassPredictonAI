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
from skimage.filters import roberts, sobel, scharr, prewitt, meijering, sato, hessian
import sys


def getint(name):
    num = name.split('.')
    return num[0]


path = r"C:\Users\klimesp\Dropbox\Programovani\Python\USA\Images\done\Final/"
train_path = r"C:\Users\klimesp\Dropbox\Programovani\Python\USA\Images\masks\Final/"
path = path if path[-1] == "/" else path + "/"
chdir(path)

files = [file for file in glob.glob("*.jpg")]
#files.sort(key=getint)

dfs = pd.DataFrame()
for file in files:  # [20:50]:
    # print(file)
    img = cv2.imread(path + file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    labeled_img = cv2.imread(train_path + file)
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)

    flat_img = img.reshape(-1)
    flat_labeled_img = labeled_img.reshape(-1)

    df = pd.DataFrame()

    # Add orifinal pixel value
    flat_img = img.reshape(-1)
    df["Original Image"] = flat_img

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
filename = "RF_model_230503"
pickle.dump(model,
            open(r"C:\Users\klimesp\Dropbox\Programovani\Python\USA/" + filename, "wb"))
