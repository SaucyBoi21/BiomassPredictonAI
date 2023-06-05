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
#mypath = r"T03_example/" #r"C:\Users\klimesp\Dropbox\Programovani\Python\Seed counter"
#
#mypath = mypath if mypath[-1] == "/" else mypath + "/"
#chdir(mypath)

#files = [file for file in glob.glob("*.png")]
#files = [file for file in glob.glob("*.jpg")]

filenames = sorted(glob.glob('T03_example/*.jpg'))


l_r, l_g, l_b, u_r, u_g, u_b, size_ig = 0,0,0,255,255,255,0
for id, file in enumerate (filenames):
    if 1==1:# file[0:2] == "24":
        img = cv2.imread(file)                  # open a file
        #img = cv2.resize(img,(1000,1000))
        #print(img.shape)
        #if img.shape[0] < img.shape[1]:
            #img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        cap = img
        frame = cap


        # Create a window named trackbars.
        cv2.namedWindow("Trackbars")

        # Now create 6 trackbars that will control the lower and upper range of
        # H,S and V channels. The Arguments are like this: Name of trackbar,
        # window name, range,callback function. For Hue the range is 0-179 and
        # for S,V its 0-255.
        cv2.createTrackbar("L - R", "Trackbars", l_r, 179, nothing)
        cv2.createTrackbar("L - G", "Trackbars", l_g, 255, nothing)
        cv2.createTrackbar("L - B", "Trackbars", l_b, 255, nothing)
        cv2.createTrackbar("U - R", "Trackbars", u_r, 179, nothing)
        cv2.createTrackbar("U - G", "Trackbars", u_g, 255, nothing)
        cv2.createTrackbar("U - B", "Trackbars", u_b, 255, nothing)
        cv2.createTrackbar("Size", "Trackbars", size_ig, 1500, nothing)


        while True:


            # Convert the BGR image to HSV image.
            rgb = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)

            # Get the new values of the trackbar in real time as the user changes
            # them
            l_r = cv2.getTrackbarPos("L - R", "Trackbars")
            l_g = cv2.getTrackbarPos("L - G", "Trackbars")
            l_b = cv2.getTrackbarPos("L - B", "Trackbars")
            u_r = cv2.getTrackbarPos("U - R", "Trackbars")
            u_g = cv2.getTrackbarPos("U - G", "Trackbars")
            u_b = cv2.getTrackbarPos("U - B", "Trackbars")
            size_ig = cv2.getTrackbarPos("Size", "Trackbars")

            # Set the lower and upper HSV range according to the value selected
            # by the trackbar
            lower_range = np.array([l_r, l_g, l_b])
            upper_range = np.array([u_r, u_g, u_b])

            # Filter the image and get the binary mask, where white represents
            # your target color
            mask = cv2.inRange(rgb, lower_range, upper_range)
            cont, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for contour in cont:
                if cv2.contourArea(contour) < size_ig:
                    cv2.fillPoly(mask, pts=[contour], color=(0, 0, 0))

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
                thearray = [[l_r, l_g, l_b], [u_r, u_g, u_b]]
                print(thearray)
                img_name = file.split('\\')[-1]
                cv2.imwrite(f'T03_mask/{img_name}', mask)

                # Also save this array as penval.npy
                #np.save('hsv_value', thearray)
                break


        # Release the camera & destroy the windows.
        cv2.destroyAllWindows()

