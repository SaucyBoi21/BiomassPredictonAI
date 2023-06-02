import cv2
import numpy as np
from os import listdir,mkdir,remove, chdir
from os.path import isfile, join
import pandas as pd
import sys
import itertools
import math
import glob, os


#Options
mypath = r"C:\Users\klimesp\Dropbox\Programovani\Python\USA\Images"
well_plate = 24

#
mypath = mypath if mypath[-1] == "/" else mypath + "/"
chdir(mypath)
files = [file for file in glob.glob("*.jpg")]



# For all pictures i the folder
for id,file in enumerate (files):
    if file[-4:] == ".jpg":
        print(f"evaluating {file}")
        img = cv2.imread(mypath+file)                   # open a file
        black = np.zeros(img.shape).astype(img.dtype)


        #img = cv2.resize(img, (2000, 1400))            # Resize image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # change the colouring


        # Definition of blue color around plate
        lower = [0 , 0, 110]  # plates 60, 31, 165
        higher = [255, 255, 255]
        lower = np.array(lower, dtype = "uint8")
        higher = np.array(higher, dtype = "uint8")

        # Picture cropping
        mask = cv2.inRange(img,lower,higher)# Make a mask
        cont, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # search for contours
        #print(cont[1])
        cont = sorted(cont,key=cv2.contourArea)
        #print(cont[-1])
        #cont_img = cv2.drawContours(img,cont,-1,(0,0,255),0)
        #cont = np.array([[50, 50], [50, 150], [150, 150], [150, 50]])

        cv2.fillPoly(black, pts = [cont[-1]], color=[255,255,255])
        #cv2.drawContours(img, cont[-1], -1, color=(255, 255, 255), thickness=cv2.FILLED)
        result = cv2.bitwise_and(img, black)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)  # change the colouring

        #result = cv2.resize(result,(1000,500))
        cv2.imshow(file, result)
        cv2.waitKey(0)
        cv2.imwrite(mypath+"Raft_cropped/"+file, result)
        #remove(mypath + file)  # remove original file

        """
        # Define colors for segmenting
        lower1 = [1,1,1]#[1, 20, 50]      # Can be violett
        higher1 = [1,1,1]#[180 , 255, 130]
        # Foils
        if foils == True:
            lower2 = [22, 30, 117]   # Green
            higher2 = [44, 148, 200]
        else:
            # No Foils
            lower2 = [22,30,85] # no foils
            higher2 = [61, 180, 165]

        lower1 = np.array(lower1, dtype="uint8")
        higher1 = np.array(higher1, dtype="uint8")
        lower2 = np.array(lower2, dtype="uint8")
        higher2 = np.array(higher2, dtype="uint8")


        area_no = 0     # Well area
        crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2HSV)  # change the colouring
        for row in range(rows):
            for column in range(columns):
                try:
                    # Definition of well segmenting
                    if well_plate == 96: sys.exit()
                    elif well_plate == 48: segmented_image = crop_image[47 + 200 * row:245 + 200 * row,
                                                             171 + 200 * column:369 + 200 * column]
                    elif well_plate == 24: segmented_image = crop_image[58 + 304 * row:362 + 304 * row,
                                                             58 + 308 * column:366 + 308 * column]
                    elif well_plate == 12: sys.exit()
                    elif well_plate == 6: sys.exit()

                    mask1 = cv2.inRange(segmented_image, lower1, higher1)
                    mask2 = cv2.inRange(segmented_image, lower2, higher2)
                    mask = cv2.bitwise_or(mask1, mask2)
                    #target = cv2.bitwise_and(img, img, mask=mask)



                    contours, hierarchyes = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                    contour_points = []


                    for contour in contours:
                        if cv2.contourArea(contour) < 30:
                            mask = cv2.fillPoly(mask, pts =[contour], color=(0,0,0))
                            #print(mask.shape)
                        else:
                            # The biggest contour is red
                            cv2.drawContours(segmented_image, contour, -1, (0, 255, 0), 0)
                            contour_points.append(contour)
                    contour_points = [j for i in contour_points for j in i]

                    points = []
                    for point in contour_points:
                        points.append(point[0])

                    max_dist = 0
                    for i,(point1,point2) in enumerate(itertools.combinations(points, 2)):
                        if i % 5 == 0:
                            dist = calculateDistance(point1[0], point1[1], point2[0], point2[1])
                            if dist > max_dist: max_dist = round(dist,1)
                    #print (max_dist)


                    #print(itertools.product(*contour_points))

                    BGR_image = cv2.cvtColor(segmented_image, cv2.COLOR_HSV2BGR)
                    img_mask = BGR_image[np.where(mask == 255)]
                    #cv2.imshow("mask", BGR_image)
                    #cv2.moveWindow("mask", 40, 30)  # Move it to (40,30)
                    #cv2.waitKey(0)
                    cv2.imwrite(mypath + "Test/" + str(area_no)+".png", BGR_image)
                    RGB = np.mean(img_mask, axis=0).astype(int)
                    area = cv2.countNonZero(mask)
                    #print(RGB)


                    if area > 10:
                        evaluation_file.write(filename +","+ "Area " + str(area_no) +","+ str(area)+"," + str(max_dist)
                                               +","+str(RGB[0])+","+str(RGB[1])
                                               +","+str(RGB[2])+"\n")
                    else:
                        evaluation_file.write(filename + "," + "Area " + str(area_no) + "," + "NaN" + "," + "NaN" + "," +
                                              "Nan" + "," + "Nan" + "," + "Nan" + "\n")

                except:
                    #print("empty well")
                    evaluation_file.write(filename + "," + "Area " + str(area_no) + ","  +
                                          "Nan" + "," + "Nan" + "," + "Nan"
                                          + "," + "Nan" + "," + "Nan" + "\n")

                area_no+=1
        segmented_image = cv2.cvtColor(crop_image, cv2.COLOR_HSV2BGR)  # change the colouring
        segmented_image = cv2.resize(segmented_image, (1500, 1000))

        if 1 == 1 and id % 1 == 0:
            cv2.imshow("mask", segmented_image)
            cv2.moveWindow("mask", 40, 30)  # Move it to (40,30)
            cv2.waitKey(0)

        """



