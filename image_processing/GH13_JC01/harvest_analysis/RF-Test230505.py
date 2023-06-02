import pickle
from sklearn.ensemble import RandomForestClassifier#,forest
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import glob
import numpy as np
from scipy import ndimage as nd
from skimage.filters import roberts, sobel, scharr, prewitt, meijering, sato, hessian
from skimage import io, img_as_float

np.set_printoptions(threshold=np.inf)

model = pickle.load(open(r"C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/ML_segmentation/RF_segmentation/Segmentation_model_pavel","rb"))
#model = pickle.load(open("RF_model_400x400_230307","rb"))

path = r"C:\Users\klimesp\Dropbox\Programovani\Python\USA\Images\done\Final\*.jpg"

def clahe(img):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Splitting the LAB image to L, A and B channels, respectively
    l, a, b = cv2.split(lab_img)

    # plt.hist(l.flat, bins=100, range=(0,255))
    ###########Histogram Equlization#############
    # Apply histogram equalization to the L channel
    equ = cv2.equalizeHist(l)

    # plt.hist(equ.flat, bins=100, range=(0,255))
    # Combine the Hist. equalized L-channel back with A and B channels
    updated_lab_img1 = cv2.merge((equ, a, b))

    # Convert LAB image back to color (RGB)
    hist_eq_img = cv2.cvtColor(updated_lab_img1, cv2.COLOR_LAB2BGR)

    ###########CLAHE#########################
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(l)
    # plt.hist(clahe_img.flat, bins=100, range=(0,255))

    # Combine the CLAHE enhanced L-channel back with A and B channels
    updated_lab_img2 = cv2.merge((clahe_img, a, b))

    # Convert LAB image back to color (RGB)
    CLAHE_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)

    return CLAHE_img


def feature_extraction(img):
    df = pd.DataFrame()
    # print(file)
    #img = cv2.imread(r"C:\Users\klimesp\Dropbox\Programovani\Python\USA\Images\done/Final/T01_GH13_JC01_Feb-17-2023_0743_rgb.jpg")

    R = img[:,:,0].reshape(-1)
    df["R"] = R


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
    #labeled_img = cv2.imread(train_path + file)
    #labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)

    flat_img = img.reshape(-1)
    #flat_labeled_img = labeled_img.reshape(-1)

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
    #df["Labels"] = flat_labeled_img
    #dfs = pd.concat([dfs, df], ignore_index=True)
    return df


#kernel = np.ones((3,3),np.float32)/8.8
kernel = np.ones((4,4),np.float32)/16.3
"""kernel = np.array([[0, -1, 0],   #3x3 kernel
                [-1, 5, -1],
                [0, -1, 0]])"""

for file in glob.glob(path)[1:2]:
    mask = cv2.imread(r"C:\Users\klimesp\Dropbox\Programovani\Python\USA\Images\masks\Final/"+file.split("\\")[-1])
    #print(file)
    img = cv2.imread(file)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #clahe_img = clahe(img)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img_filter = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    #flat_img = np.array(img.reshape(-1,1))
    flat_img = feature_extraction(img)
    #flat_img_filter = feature_extraction(img_filter)

    result = model.predict(flat_img)
    #result_filter = model.predict(flat_img_filter)

    result = result.reshape ((img.shape[:2]))
    #result_filter = result_filter.reshape ((img.shape))

    print(file)
    ax = plt.subplot(131)
    ax.set_title("Original")
    plt.axis('off')
    plt.imshow(rgb)
    ax2 = plt.subplot(132)
    ax2.set_title("FR  "+str(np.count_nonzero(result)))
    plt.axis('off')
    plt.imshow(result)
    #ax3 = plt.subplot(133)
    #ax3.set_title("RF+filter  "+str(np.count_nonzero(result_filter)))
    #plt.axis('off')
    #plt.imshow(result_filter)
    
    try:
        ax3 = plt.subplot(133)
        ax3.set_title("Mask  "+str(np.count_nonzero(mask)))
        plt.axis('off')
        plt.imshow(mask)
    except: pass
    

    #figManager = plt.get_current_fig_manager()
    #figManager.window.showMaximized()
    wm = plt.get_current_fig_manager()
    wm.window.state('zoomed')
    plt.show()
    #cv2.waitKey()
    
    plt.imshow(result)

