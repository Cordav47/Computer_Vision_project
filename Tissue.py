#import
import math
import scipy as sp
import numpy as np
import time
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from Additional_functions import Imp, dcpplot

vis = True
#Q at the beginning stand for query 
qtis, QTIS, qfiltis, QFILTIS = Imp("Photos/Tissue/Query_tissue.jpg")

#Import the tissue write image
st, ST, filst, FILST = Imp("Photos/Tissue/scritta_tissue.png")

#the trademark created, of course there's no need to filter it
ttis, TTIS = Imp("Photos/Tissue/tissue.png")[:2]


sift = cv2.xfeatures2d.SIFT_create()
FLANN_INDEX_KDTREE = 1

#low value, we don't have so much features as in the other 2 products
MIN_MATCH_COUNT = 20
kp_query = sift.detect(QFILTIS)
kp_query, des_query = sift.compute(QFILTIS, kp_query)

def evaluate_tissue(link):
    tis, TIS, filtis, FILTIS = Imp(link)

    if vis:
        plt.imshow(TIS)
        plt.show()

    #Template match
    match = cv2.matchTemplate(FILTIS, TTIS, cv2.TM_SQDIFF_NORMED)

    #doing geometrical consideration we give the ok if the area of the match is within certain values
    #Sort of custom threshold
    match_area = np.where(match <= 0.2, 0, match)
    match_area = np.where(match > 0.2, 1, match_area)
    if vis:
        plt.imshow(match_area)
        plt.show()
    h,w = np.shape(match)
    pixel_area_sub = np.count_nonzero(match_area) 
    pixel_area = h*w - pixel_area_sub

    print(pixel_area)
    #min, max, minloc, maxloc = cv2.minMaxLoc(match)
    #Because we have a fixed place to take photos and products have a standard size (negligible distortion due to non pefectly allignment of the product)

    #if pixel area is between 400000 and 800000 is ok, threshold found in practice through
    #geometric consideration
    low_threshold = 400000
    high_threshold = 800000
    if pixel_area > low_threshold & pixel_area < high_threshold:
        print("Object detected")
        flag = True
    else:
        flag = False
        if pixel_area < low_threshold:
            print("object not detected")
        if pixel_area > high_threshold:
            print("More objects detected")

    if flag:
        kp_train = sift.detect(FILTIS)
        kp_train, des_train = sift.compute(FILTIS, kp_train)

        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des_query,des_train,k=2)

        good = []
        for m,n in matches:
            if m.distance < 0.8*n.distance: #0.8 degree of freedom, index of acceptability of a keypoint
                good.append(m)

        h,w = qfiltis.shape

        if len(good) >= MIN_MATCH_COUNT:
            #correspondence procedure
            src_pts = np.float32([ kp_query[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp_train[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            #Compute homography matrix
            M1, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 9.0)
            #Build a rectangle to draw it around the product
            pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M1)
            print(dst)

            #TIS_CUT = FILTIS[dst[0,1]:dst[2,1], dst[0,0]:dst[2,0]]
            
            dst = np.int32(dst.reshape(4,2))

            if M1[0,0]< 0 and M1[1,1] < 0: #if tissue is flipped
                dst = np.roll(dst, shift = 2, axis = 0) 

            parallelogram_vertices = np.array([[dst[0,0], dst[0,1]], [dst[1,0], dst[1,1]], [dst[2,0], dst[2,1]], [dst[3,0], dst[3,1]]], dtype=np.float32)

            nh = np.int32((dst[1,1]-dst[3,1]+dst[2,1]-dst[0,1])/2)
            nw = np.int32((-dst[1,0]-dst[0,0]+dst[2,0]+dst[3,0])/2)
            # Define the vertices of the destination rectangle
            rectangle_vertices = np.array([[0, 0], [0, nh], [nw, nh], [nw,0]], dtype=np.float32)

            # Calculate the perspective transformation matrix
            perspective_matrix = cv2.getPerspectiveTransform(parallelogram_vertices, rectangle_vertices)

            # Apply the perspective transformation
            warped_image = cv2.warpPerspective(FILTIS, perspective_matrix, (nw,nh))

            if M1[0,0]< 0 and M1[1,1] < 0: #rotate it if it's flipped
                warped_image = cv2.rotate(warped_image, cv2.ROTATE_180)

            image_resized = cv2.resize(QFILTIS, (warped_image.shape[1], warped_image.shape[0]), interpolation=cv2.INTER_AREA)

            diff = cv2.absdiff(image_resized, warped_image)

            #binarize
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
            #filter
            binary_diff = cv2.threshold(gray_diff, 50, 255, cv2.THRESH_BINARY)[1]
            #kernel construction
            size_k = 3
            kernel = np.zeros((size_k, size_k))
            kernel[:] = 1
            #open the difference
            opened_diff = cv2.morphologyEx(binary_diff, cv2.MORPH_OPEN, kernel)

            plt.imshow(opened_diff)
            plt.show()
            ht, wt = np.shape(opened_diff)
            area = ht*wt

            #count the final difference pixel
            difference_pixel = np.count_nonzero(opened_diff)

            ratio = difference_pixel/area
            acceptability = 0.01
            print(ratio)
            
            if ratio > acceptability: #product is bad
                
                PIC = cv2.polylines(FILTIS, [np.int32(dst)], True, (255, 0, 0), 5, cv2.LINE_AA)
                dcpplot(TIS, PIC, '0010', False)
            else: #product is good, check orientation
                """
                half_h = np.int32(ht/2)
                cut_f = 50
                cut_img = warped_image[cut_f:-cut_f, cut_f:-cut_f]
                cut_img = cv2.cvtColor(cut_img, cv2.COLOR_RGB2GRAY)
                thresh_white = cv2.threshold(cut_img, 150, 255, cv2.THRESH_BINARY)[1]
                #plt.imshow(thresh_white)
                #plt.show()
                thres_up = thresh_white[:half_h,:]
                thresh_down = thresh_white[half_h:,:]

                white_up = np.count_nonzero(thres_up)
                white_down = np.count_nonzero(thresh_down)
                """
                #Blue lines because the green would be too similar to the background
                if M1[0,0] > 0: #just use M1 as a discriminant
                    PIC = cv2.polylines(FILTIS, [np.int32(dst)], True, (0,0,255), 5, cv2.LINE_AA)
                    dcpplot(TIS, PIC, '0000', True)
                elif M1[0,0] < 0:
                    PIC = cv2.polylines(FILTIS, [np.int32(dst)], True, (200, 100, 100), 5, cv2.LINE_AA)
                    dcpplot(TIS, PIC, '0001', True, flip=True)

    else: #there's no product
        dcpplot(TIS, FILTIS, '1000', False)


        

evaluate_tissue("Photos/Tissue/S1.jpg") #OK
time.sleep(1)
evaluate_tissue("Photos/Tissue/S2.jpg") #Flipped upside down
time.sleep(1)
evaluate_tissue("Photos/Tissue/S9.jpg") #Error written misplaced
time.sleep(1)
evaluate_tissue("Photos/Tissue/S4.jpg") #OK
time.sleep(1)
evaluate_tissue("Photos/Tissue/S11.jpg") #No product
time.sleep(1)
evaluate_tissue("Photos/Tissue/S6.jpg") #OK
time.sleep(1)
evaluate_tissue("Photos/Tissue/S10.jpg") #Error, some scratch in the pic
time.sleep(1)
evaluate_tissue("Photos/Tissue/S8.jpg") #error, written reversed
time.sleep(1)
evaluate_tissue("Photos/Tissue/S3.jpg") #OK



