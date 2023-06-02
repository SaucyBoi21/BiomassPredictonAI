import os
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

model = pickle.load(open(r"Segmentation_model_pavel","rb"))
#model = pickle.load(open("RF_model_400x400_230307","rb"))

path_images = r"height_harvest/*.jpg"


DEBUG = False

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


#photos = glob.glob(path_images)
#img = cv2.imread(photos[0])
#flat_img = feature_extraction(img)

for id,file in enumerate( glob.glob(path_images)):
    base_name = os.path.basename(file)
    name_img = os.path.splitext(base_name)[0]
    #print(glob.glob(path_images)[id][:-4])
    depth_img = np.array(pd.read_csv(file.split(".")[0]+".csv"))
    #mask = cv2.imread(r"height_harvest/"+file.split("\\")[-1])
    img = cv2.imread(file)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    flat_img = feature_extraction(img)
    result = model.predict(flat_img)
    result = result.reshape ((img.shape[:2]))
    plt.imshow(result)
    cv2.imwrite(f"masks_data/mask_{name_img}.jpg", result)
    blur = cv2.GaussianBlur(result,(11,11),0)

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
            one_cont_img = cv2.fillPoly(background, pts=[c], color=255) 
            bordered_mask = generate_border(one_cont_img,10,0)
            ref_height = depth_img[np.where(bordered_mask == 127)]
            ref_height = [i for i in ref_height if i != 0]
            plant_height_max = (np.median(ref_height))-min([i for i in depth_img[np.where(bordered_mask == 255)] if i != 0])
            plant_height_avg = (np.median(ref_height))-np.mean([i for i in depth_img[np.where(bordered_mask == 255)] if i != 0])
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

    
  
    
    # cv2.imwrite(r"C:\Users\klimesp\Dropbox\Programovani\Python\USA\blur.jpg",blur)

