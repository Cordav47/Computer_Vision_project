import math
import scipy as sp
import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from Additional_functions import Imp, Opener, cancel_borders, dcpplot, compute_displacement, enlarge_borders


mpl.use('TkAgg')
plt.rcParams["figure.figsize"] = (12,6)
plt.rcParams.update({'font.size': 20})

denb, DENB, fildenb, FILDENB = Imp(("Foto CV Coesia project/QueryD.jpg"))

#series of feature computed offline and valid for each iteration
sift = cv2.xfeatures2d.SIFT_create()
FLANN_INDEX_KDTREE = 1
MIN_MATCH_COUNT = 350
kp_query = sift.detect(FILDENB)
kp_query, des_query = sift.compute(FILDENB, kp_query)

def trova_den(link):
    den, DEN, filden, FILDEN = Imp(link)
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
       if m.distance < 0.8*n.distance: #0.9 degree of freedom, index of acceptability of a keypoint
        good.append(m)


    h,w = denb.shape

    print(len(good))
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp_query[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp_train[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    # Using RANSAC to estimate a robust homography. 
    # It returns the homography M and a mask for the discarded points

        M1, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 16.0)
         # Projecting the corners into the train image
        
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M1)
        print(M1)
        dst = dst.reshape(4,2)
        
        #Cut the image to match the size with the query one in order to provide the subtraction
        basex = np.int32(dst[0,0])
        basey = np.int32(dst[0,1])
        #Here we consider just the case upside down
        if M1[0,0] < -0.8 and M1[1,1] < -0.8:
            print("toothpaste flipped")
            basex = np.int32(dst[2,0])
            basey = np.int32(dst[2,1])
            flip = True

        cut_img = FILDEN[basey: basey+h, basex:basex+w]        

        if flip == True:
            cut_img = cv2.rotate(cut_img, cv2.ROTATE_180)

        diff = cv2.absdiff(cut_img, FILDENB)
        #Now color the differences to remark it, we choose a threshold to mark the difference
        diff_threshold = 30
        diff_img = np.where(diff > diff_threshold, [255,0,0], diff)
        
        color_mask = np.all( diff_img == ([255, 0, 0]), axis=-1)
        
        pixel_count = np.count_nonzero(color_mask)
        diffbn = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        #Binarized image
        binary_diff = cv2.threshold(diffbn, 30, 255, cv2.THRESH_BINARY)[1]
        plt.imshow(binary_diff)
        plt.show()
        dst_pts = dst_pts.reshape(-1,2)
        src_pts = src_pts.reshape(-1,2)
        #compute displacement
        src_adapted = src_pts+dst[0,:]
        dx, dy = compute_displacement(src_adapted, dst_pts)
        dx = np.int32(dx)
        dy = np.int32(dy)
        print(dx, dy)
        absx = np.abs(dx)+1
        absy = np.abs(dy)+1
        new_cut = np.copy(cut_img)
        new_fil = np.copy(FILDENB)
        if dx > 0:
            new_cut = cut_img[:, absx:]
            new_fil = FILDENB[:, :-absx]
        elif dx < 0:
            new_cut = cut_img[:, :-absx]
            new_fil = FILDENB[:, absx:]

        if dy > 0:
            new_cut = cut_img[absy:,:]
            new_fil = FILDENB[:-absy, :]
        elif dx < 0:
            new_cut = cut_img[:-absy, :]
            new_fil = FILDENB[absy:, :]

        #Now create the mask with the edge, enlarge them and further enlarge the lateral edges
        gray_cut_img = cv2.cvtColor(new_cut, cv2.COLOR_RGB2GRAY)
        gray_new_fil = cv2.cvtColor(new_fil, cv2.COLOR_RGB2GRAY)
        img_edge = cv2.Canny(gray_cut_img, 100, 50, apertureSize = 3)
        fil_edge = cv2.Canny(gray_new_fil, 100, 50, apertureSize = 3)
        k_edge = 3
        kernel_edge = np.zeros((k_edge, k_edge))
        kernel_edge[:] = 1
        #img_edge_dil = cv2.morphologyEx(img_edge, cv2.MORPH_DILATE, kernel_edge)
        #fil_edge_dil = cv2.morphologyEx(fil_edge, cv2.MORPH_DILATE, kernel_edge)

        enlarged_borders1 = enlarge_borders(img_edge, 13)
        enlarged_borders2 = enlarge_borders(fil_edge, 13)
        enlarged_borders = enlarged_borders1 + enlarged_borders2
        plt.imshow(enlarged_borders)
        plt.show()
        difference_wt_displacement = cv2.absdiff(new_cut, new_fil)
        plt.imshow(difference_wt_displacement)
        plt.show()
        binary_difference_wt_displacement = cv2.threshold(difference_wt_displacement, 80, 255, cv2.THRESH_BINARY)[1]
        Masked_difference = cancel_borders(binary_difference_wt_displacement, enlarged_borders)
        binary_difference_wt_displacement = cv2.cvtColor(binary_difference_wt_displacement, cv2.COLOR_RGB2GRAY)

        masked_binary_difference = np.where(enlarged_borders == 255, 0, binary_difference_wt_displacement )
        masked_binary_difference = cv2.cvtColor(Masked_difference, cv2.COLOR_RGB2GRAY)
        masked_binary_difference = cv2.threshold(masked_binary_difference, 20, 255, cv2.THRESH_BINARY)[1]
        size_k = 11
        kernel = np.zeros((size_k, size_k))
        kernel[:] = 1
        opened_diff = cv2.morphologyEx(masked_binary_difference, cv2.MORPH_OPEN, kernel )
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(Masked_difference)
        plt.subplot(1,2,2)
        plt.imshow(binary_difference_wt_displacement)
        plt.show()

        plt.figure()
        plt.imshow(opened_diff)
        plt.show()

        acceptability = 0.0075
        ratio = np.count_nonzero(opened_diff)/(h*w)
        print(ratio)

        if ratio > acceptability:
            #Product damaged, draw red boundary
            DEN = cv2.polylines(DEN,[np.int32(dst)],True, [255,0, 0],5, cv2.LINE_AA)
            SRvalue = '0010'
            SRout = False
            dcpplot(FILDEN, DEN, SRvalue, SRout)
            quality = 3

        elif ratio <= acceptability:
            # Drawing the bounding box
            DEN = cv2.polylines(DEN,[np.int32(dst)],True, [0,255, 0],5, cv2.LINE_AA)
            SRvalue = '0000'
            SRout = True
            dcpplot(FILDEN, DEN, SRvalue, SRout, flip)
            quality = 1       

    else:
        ret, thresh = cv2.threshold(den, 150, 255, cv2.THRESH_BINARY )
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
            Pic = cv2.rectangle(DEN, (stats[max_area_index, cv2.CC_STAT_LEFT], stats[max_area_index, cv2.CC_STAT_TOP]), (stats[max_area_index, cv2.CC_STAT_LEFT] +stats[max_area_index, cv2.CC_STAT_WIDTH], stats[max_area_index, cv2.CC_STAT_TOP] + stats[max_area_index, cv2.CC_STAT_HEIGHT]), (255, 255, 0), 5)
            plt.tight_layout()
            fig_manager = plt.get_current_fig_manager()
            fig_manager.window.wm_geometry("+50+50")
            plt.subplot(1,2,1)
            plt.imshow(FILDEN)
            plt.suptitle("Il prodotto va girato di 180 gradi", color = "orange")
            plt.subplot(1,2,2)
            plt.imshow(Pic)
            plt.figtext(0.8, 0.8, ("SR value : 0001"), fontsize=16, bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.5'))
            plt.figtext(0.8, 0.7, ("SR OUT : True"), fontsize=16, bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.5'))
            plt.show()
            quality = 2
        else:
            print("error, not enough match found")
            plt.tight_layout()
            fig_manager = plt.get_current_fig_manager()
            fig_manager.window.wm_geometry("+50+50")
            plt.subplot(1,2,1)
            plt.imshow(FILDEN)
            plt.suptitle("Prodotto non trovato", color = "red")
            plt.subplot(1,2,2)
            plt.imshow(DEN)
            plt.figtext(0.8, 0.8, ("SR value : 1000"), fontsize=16, bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.5'))
            plt.figtext(0.8, 0.7, ("SR OUT : False"), fontsize=16, bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5'))
            plt.show()
            quality = 4

    return quality

#trova_den("Foto CV Coesia project/D9.jpg")
trova_den("Foto CV Coesia project/D1.jpg")
time.sleep(1)
trova_den("Foto CV Coesia project/D2.jpg")
time.sleep(1)
trova_den("Foto CV Coesia project/D3.jpg")
time.sleep(1)
trova_den("Foto CV Coesia project/D4.jpg")
time.sleep(1)
trova_den("Foto CV Coesia project/D5.jpg")
time.sleep(1)
trova_den("Foto CV Coesia project/D6.jpg")
time.sleep(1)
#trova_den("Foto CV Coesia project/D8.jpg")
trova_den("Foto CV Coesia project/D7.jpg")