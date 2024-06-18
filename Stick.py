import math
import scipy as sp
import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from Additional_functions import dcsplot, Imp, Opener, cancel_borders


sift = []
query = []
FLANN_INDEX_KDTREE = 1
MIN_MATCH_COUNT = 100
des_query = []

front, FRONT, filfront, FILFRONT = Imp("Photos/Stick/Query_front.jpg")
back,BACK, filback, FILBACK = Imp("Photos/Stick/Query_back.jpg")
side1, SIDE1, filside1, FILSIDE1 = Imp("Photos/Stick/query_side_ccw.jpg")
side2, SIDE2, filside2, FILSIDE2 = Imp("Photos/Stick/query_side_cw.jpg")

images = [FILFRONT, FILBACK, FILSIDE1, FILSIDE2]
gray_images = [front, back, side1, side2]

for i in range(0, 4, 1):
    siftis = cv2.xfeatures2d.SIFT_create()
    sift.append(siftis)
    kp_query = siftis.detect(images[i])
    kp_query, desimg_query = siftis.compute(images[i], kp_query)
    query.append(kp_query)
    des_query.append(desimg_query)



def evaluate_stick(link):

    stick, STICK, filstick, FILSTICK = Imp(link)
    kp_train = []
    des_train = []
    matches = []
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    percentages = [0,0,0,0]

    for i in range (0, 4, 1):
        train = sift[i].detect(FILSTICK)
        train, d_train = sift[i].compute(FILSTICK, train)
        kp_train.append(train)
        des_train.append(d_train)
        match = flann.knnMatch(des_query[i], des_train[i], k = 2)
        matches.append(match)
        good = []
        
        for m,n in matches[i]:
            if m.distance < 0.8*n.distance:
                good.append(m)

        #Procedure to find the sides with most correspondences
        percentages[i] = len(good)/np.size(query[i])

        if i == 0:
            good_best = good
            index = 0
            print("Frontal correspondences are:", len(good))
        elif i > 0:
            if percentages[i] == np.max(percentages):
                good_best = good
                print("The side with most correspondences is the", i)
                index = i

        if len(good_best) >= 0.75*len(query[i]):
            break
        print(len(good))

    h,w = np.shape(gray_images[index]) 

    thresh_white = cv2.threshold(filstick, 0, 255, cv2.THRESH_OTSU)[1]

    if len(good_best) > MIN_MATCH_COUNT: #if the side with most matches is major than the match threshold, basically another way to say if there's the product
        ngood = []
        for m,n in matches[index]:
            if m.distance < 0.8*n.distance:
                ngood.append(m)
        src_pts = np.float32([ query[index][m.queryIdx].pt for m in ngood ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp_train[index][m.trainIdx].pt for m in ngood ]).reshape(-1,1,2)
        M1, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 9.0)
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M1)
        print(M1)

        #Compute the angle
        rotation_angle_rad = np.arctan2(M1[1, 0], M1[0, 0])
        #translate into degrees
        rotation_angle_deg = np.int32(np.degrees(rotation_angle_rad))
        print(rotation_angle_deg)

        dst = dst.reshape(4,2)
        dstint = np.int32(dst)

        #cut the lower part to see if there's white part
        basex = dstint[0,0]
        basey = dstint[0,1]
        stick_thresh = thresh_white[basey:dstint[1,1], basex:dstint[2,0]]

        #Take the last 300 pixels in height to verify if there's the white part
        lower_part = stick_thresh[-300:, :]
        #plt.imshow(lower_part)
        #plt.show()

        check_white = np.sum(lower_part == 255)
        tr_white = 10000
        if check_white >= tr_white:

            clockwise = True
            if index == 0: #front side
                rot_angle = np.abs(rotation_angle_deg)
                if rotation_angle_deg > 0:
                    clockwise = False
            elif index == 1: #back side
                rot_angle = 180 - np.abs(rotation_angle_deg)
                if rotation_angle_deg < 0:
                    clockwise = False
            elif index == 2:

                rot_angle = 90 - rotation_angle_deg 
            elif index == 3:
            
                rot_angle = 90 + rotation_angle_deg
                clockwise = False
                    
            if rot_angle < 10: #If the detected angle is small we don't operate the motor
                Srvalue = '0000'
            else:
                Srvalue = '0001'

            
            #Pick the lowest part of the stick and see if it it's white to check the integrity
            PIC = cv2.polylines(FILSTICK, [np.int32(dst)], True, (0,255, 0), 5, cv2.LINE_AA)
            dcsplot(STICK, PIC, Srvalue, True, rot_angle, clockwise)
        else:
            Srvalue = '0010'
            PIC = cv2.polylines(FILSTICK, [np.int32(dst)], True, (255,0, 0), 5, cv2.LINE_AA)
            dcsplot(STICK, PIC, Srvalue, False, 0)
    else:
        #Trademark not found, could be either defective or missing product
        thresh_white = cv2.threshold(filstick, 180, 255, cv2.THRESH_OTSU)[1]
        sum_white = np.sum(thresh_white == 255)
        if sum_white >= 400000:
            Srvalue = '0010'
            #We could compute here the connected component to highlight the white stick
            #In practice it should be a useless computation we don't perform it
    

        else:
            Srvalue = '1000'

        dcsplot(STICK, FILSTICK, Srvalue, False, 0)


#The online procedure

evaluate_stick("Photos/Stick/S1.jpg") 
evaluate_stick("Photos/Stick/S3.jpg") 
evaluate_stick("Photos/Stick/S13.jpg")
evaluate_stick("Photos/Stick/S11.jpg")
evaluate_stick("Photos/Stick/S5.jpg")
evaluate_stick("Photos/Stick/S6.jpg")
evaluate_stick("Photos/Stick/S7.jpg")
evaluate_stick("Photos/Stick/S8.jpg")
evaluate_stick("Photos/Stick/S9.jpg")
evaluate_stick("Photos/Stick/S10.jpg")
evaluate_stick("Photos/Stick/S16.jpg")
evaluate_stick("Photos/Stick/S12.jpg")
evaluate_stick("Photos/Stick/S20.jpg")
evaluate_stick("Photos/Stick/S14.jpg")
evaluate_stick("Photos/Stick/S17.jpg")
evaluate_stick("Photos/Stick/S18.jpg")
evaluate_stick("Photos/Stick/S2.jpg")
evaluate_stick("Photos/Stick/S4.jpg")
       
