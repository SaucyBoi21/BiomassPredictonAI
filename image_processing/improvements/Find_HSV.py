# finding hsv range of target object(pen)
import cv2
import numpy as np
import time
import glob
from os import chdir,remove,mkdir

# A required callback method that goes into the trackbar function.
def nothing(x):
    pass

#Options
mypath = r"C:/Users/jcard/OneDrive - University of Georgia/kinect_imaging/GH13_JC01/rgb_imgs/tray_1/raw_images/" #r"C:\Users\klimesp\Dropbox\Programovani\Python\Seed counter"
#
mypath = mypath if mypath[-1] == "/" else mypath + "/"
chdir(mypath)

#files = [file for file in glob.glob("*.png")]
files = [file for file in glob.glob("*.jpg")]

l_h, l_s, l_v, u_h, u_s, u_v = 0,0,0,179,255,255
for id, file in enumerate (files):
    if 1==1:# file[0:2] == "24":
        img = cv2.imread(mypath+file)                  # open a file
        #img = cv2.resize(img,(1000,1000))
        #print(img.shape)
        if img.shape[0] < img.shape[1]:
            img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        cap = img
        frame = cap


        # Create a window named trackbars.
        cv2.namedWindow("Trackbars")

        # Now create 6 trackbars that will control the lower and upper range of
        # H,S and V channels. The Arguments are like this: Name of trackbar,
        # window name, range,callback function. For Hue the range is 0-179 and
        # for S,V its 0-255.
        cv2.createTrackbar("L - H", "Trackbars", l_h, 179, nothing)
        cv2.createTrackbar("L - S", "Trackbars", l_s, 255, nothing)
        cv2.createTrackbar("L - V", "Trackbars", l_v, 255, nothing)
        cv2.createTrackbar("U - H", "Trackbars", u_h, 179, nothing)
        cv2.createTrackbar("U - S", "Trackbars", u_s, 255, nothing)
        cv2.createTrackbar("U - V", "Trackbars", u_v, 255, nothing)


        while True:


            # Convert the BGR image to HSV image.
            hsv = cv2.cvtColor(cap, cv2.COLOR_BGR2LAB)

            # Get the new values of the trackbar in real time as the user changes
            # them
            l_h = cv2.getTrackbarPos("L - H", "Trackbars")
            l_s = cv2.getTrackbarPos("L - S", "Trackbars")
            l_v = cv2.getTrackbarPos("L - V", "Trackbars")
            u_h = cv2.getTrackbarPos("U - H", "Trackbars")
            u_s = cv2.getTrackbarPos("U - S", "Trackbars")
            u_v = cv2.getTrackbarPos("U - V", "Trackbars")

            # Set the lower and upper HSV range according to the value selected
            # by the trackbar
            lower_range = np.array([l_h, l_s, l_v])
            upper_range = np.array([u_h, u_s, u_v])

            # Filter the image and get the binary mask, where white represents
            # your target color
            mask = cv2.inRange(hsv, lower_range, upper_range)

            # You can also visualize the real part of the target color (Optional)
            res = cv2.bitwise_and(frame, frame, mask=mask)

            # Converting the binary mask to 3 channel image, this is just so
            # we can stack it with the others
            mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # stack the mask, orginal frame and the filtered result
            stacked = np.hstack((mask_3, frame, res))

            # Show this stacked frame at 40% of the size.
            cv2.imshow('Trackbars', cv2.resize(stacked, None, fx=0.3, fy=0.3))

            # If the user presses ESC then exit the program
            key = cv2.waitKey(1)
            if key == 27:
                break

            # If the user presses `s` then print this array.
            if key == ord('s'):
                thearray = [[l_h, l_s, l_v], [u_h, u_s, u_v]]
                print(thearray)

                # Also save this array as penval.npy
                #np.save('hsv_value', thearray)
                break


        # Release the camera & destroy the windows.
        cv2.destroyAllWindows()

