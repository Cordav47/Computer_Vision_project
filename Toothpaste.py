import math
import scipy as sp
import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from Additional_functions import Imp, dcpplot

#Visualization parameter
mpl.use('TkAgg')
plt.rcParams["figure.figsize"] = (12,6)
plt.rcParams.update({'font.size': 20})

#variable that define if we wan to visualize the intermediate procedure
vis = True

denb, DENB, fildenb, FILDENB = Imp(("Photos/Toothpaste/QueryD.jpg"))

#series of feature computed offline and valid for each iteration
sift = cv2.xfeatures2d.SIFT_create()
FLANN_INDEX_KDTREE = 1
MIN_MATCH_COUNT = 200
kp_query = sift.detect(FILDENB)
kp_query, des_query = sift.compute(FILDENB, kp_query)

def trova_den(link):
    den, DEN, filden, FILDEN = Imp(link)
    if vis:
        plt.imshow(DEN)
        plt.show()
    image_height, image_width = den.shape
    kp_train = sift.detect(FILDEN)
    kp_train, des_train = sift.compute(FILDEN, kp_train)
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_query,des_train,k=2) #k degree of freedom, number of nearest neighbours we have to evaluate
    flip = False
    good = []
    for m,n in matches:
       if m.distance < 0.8*n.distance: #0.8 degree of freedom, index of acceptability of a keypoint
        good.append(m)

    h,w = denb.shape

    print(len(good))
    if len(good)>MIN_MATCH_COUNT: #there's a match
        src_pts = np.float32([ kp_query[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp_train[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    # Using RANSAC to estimate a robust homography. 
    # It returns the homography M and a mask for the discarded points

        M1, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 16.0)
         # Projecting the corners into the train image
        
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M1)
        print(M1)
        dst = np.int32(dst.reshape(4,2))
        
        #Here we consider just the case upside down
        if M1[0,0] < 0 and M1[1,1] < 0:
            print("toothpaste flipped")
            dst = np.roll(dst,shift = 2, axis = 0) #Shift the array to keep the nh and nw math unchanged
            flip = True

        parallelogram_vertices = np.array([[dst[0,0], dst[0,1]], [dst[1,0], dst[1,1]], [dst[2,0], dst[2,1]], [dst[3,0], dst[3,1]]], dtype=np.float32)

        nh = np.int32((dst[1,1]-dst[3,1]+dst[2,1]-dst[0,1])/2)
        nw = np.int32((-dst[1,0]-dst[0,0]+dst[2,0]+dst[3,0])/2)
        # Define the vertices of the destination rectangle
        rectangle_vertices = np.array([[0, 0], [0, nh], [nw, nh], [nw,0]], dtype=np.float32)

        # Calculate the perspective transformation matrix
        perspective_matrix = cv2.getPerspectiveTransform(parallelogram_vertices, rectangle_vertices)

        # Apply the perspective transformation
        warped_image = cv2.warpPerspective(FILDEN, perspective_matrix, (nw,nh))
        if vis:
            plt.imshow(warped_image)
            plt.show()

        if flip: #rotate the image 
            warped_image = cv2.rotate(warped_image, cv2.ROTATE_180)
        #query image resize
        image_resized = cv2.resize(FILDENB, (warped_image.shape[1], warped_image.shape[0]), interpolation=cv2.INTER_AREA)

        diff = cv2.absdiff(warped_image, image_resized)

        gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

        binary_diff = cv2.threshold(gray_diff, 40, 255, cv2.THRESH_BINARY)[1]

        img_edge = cv2.Canny(warped_image, 80, 20, apertureSize = 3)

        masked_diff = binary_diff - img_edge
        #kernel construction
        size_k = 3
        kernel = np.zeros((size_k, size_k))
        kernel[:] = 1
        opened_diff = cv2.morphologyEx(masked_diff, cv2.MORPH_OPEN, kernel)
        if vis == True:
            plt.figure()
            plt.subplot(2,2,1)
            plt.imshow(gray_diff)
            plt.subplot(2,2,2)
            plt.imshow(binary_diff)
            plt.subplot(2,2,3)
            plt.imshow(masked_diff)
            plt.subplot(2,2,4)
            plt.imshow(opened_diff)
            #Evolution of difference
            plt.show()

        acceptability = 0.005 #The 99% of the image should have non defects
        ratio = np.count_nonzero(opened_diff)/(nh*nw)
        print(ratio)

        if ratio > acceptability:
            #Product damaged, draw red boundary
            SDEN = cv2.polylines(FILDEN,[np.int32(dst)],True, [255,0, 0],5, cv2.LINE_AA)
            SRvalue = '0010'
            SRout = False
            dcpplot(DEN, SDEN, SRvalue, SRout)
            quality = 3

        elif ratio <= acceptability:
            # Product ok, Drawing the green bounding box
            SDEN = cv2.polylines(FILDEN,[np.int32(dst)],True, [0,255, 0],5, cv2.LINE_AA)
            SRvalue = '0000'
            SRout = True
            dcpplot(DEN, SDEN, SRvalue, SRout, flip)
            quality = 1

    else: #Correspondence not found
        thresh = cv2.threshold(den, 150, 255, cv2.THRESH_BINARY )[1]
        num_black_pixels = np.sum(thresh == 0)
        num_white_pixels = np.sum(thresh == 255) 
        #plt.imshow(thresh)
        #plt.show()
        if num_white_pixels >= 0.1*h*w: #it basically means that there's at least a 10% of the screen white -> there's a product
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity = 8, ltype = cv2.CV_32S)
            max_area_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            max_area = stats[max_area_index, cv2.CC_STAT_AREA]
            x = stats[max_area_index, cv2.CC_STAT_LEFT]
            y = stats[max_area_index, cv2.CC_STAT_TOP]
            width = stats[max_area_index, cv2.CC_STAT_WIDTH]
            heigth = stats[max_area_index, cv2.CC_STAT_HEIGHT]
            Pic = cv2.rectangle(FILDEN, (stats[max_area_index, cv2.CC_STAT_LEFT], stats[max_area_index, cv2.CC_STAT_TOP]), (stats[max_area_index, cv2.CC_STAT_LEFT] +stats[max_area_index, cv2.CC_STAT_WIDTH], stats[max_area_index, cv2.CC_STAT_TOP] + stats[max_area_index, cv2.CC_STAT_HEIGHT]), (255, 255, 0), 5)
            plt.tight_layout()
            fig_manager = plt.get_current_fig_manager()
            fig_manager.window.wm_geometry("+50+50")
            plt.subplot(1,2,1)
            plt.imshow(DEN)
            plt.suptitle("Product must be turned of 180°", color = "orange")
            plt.subplot(1,2,2)
            plt.imshow(FILDEN)
            plt.figtext(0.8, 0.8, ("SR value : 0001"), fontsize=16, bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.5'))
            plt.figtext(0.8, 0.7, ("SR OUT : True"), fontsize=16, bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.5'))
            plt.show()
            quality = 2
        else: #There's no product
            print("error, not enough match found")
            plt.tight_layout()
            fig_manager = plt.get_current_fig_manager()
            fig_manager.window.wm_geometry("+50+50")
            plt.subplot(1,2,1)
            plt.imshow(FILDEN)
            plt.suptitle("Product not found", color = "red")
            plt.subplot(1,2,2)
            plt.imshow(DEN)
            plt.figtext(0.8, 0.8, ("SR value : 1000"), fontsize=16, bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.5'))
            plt.figtext(0.8, 0.7, ("SR OUT : False"), fontsize=16, bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5'))
            plt.show()
            quality = 4

    return quality

        
#The online procedure

trova_den("Photos/Toothpaste/D1.jpg") #OK
time.sleep(1)
trova_den("Photos/Toothpaste/D6.jpg") #Defective, some scratch
time.sleep(1)
trova_den("Photos/Toothpaste/D2.jpg") #OK
time.sleep(1)
trova_den("Photos/Toothpaste/D8.jpg") #Flipped upside down
time.sleep(1)
trova_den("Photos/Toothpaste/D3.jpg") #OK
time.sleep(1)
trova_den("Photos/Toothpaste/D9.jpg") #Defective, written eucerin reversed
time.sleep(1)
trova_den("Photos/Toothpaste/D4.jpg") #Flipped front back
time.sleep(1)
trova_den("Photos/Toothpaste/D5.jpg") #Defective, some scratch
time.sleep(1)
trova_den("Photos/Toothpaste/D7.jpg") #Defective, no written