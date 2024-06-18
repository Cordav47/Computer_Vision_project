import math
import scipy as sp
import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import time 

#import tubetti
#import Dentifrici
#import salviette
#from tubetti import ruota_tub
#from Dentifrici import trova_den
#from salviette import match

#Now let's import the picture of the box and then apply the 3 functions to find the three objects
mpl.use('TkAgg')
plt.rcParams["figure.figsize"] = (12,6)


def Imp(link):
    img = cv2.imread(link, cv2.IMREAD_GRAYSCALE)
    IMG = cv2.imread(link, cv2.IMREAD_COLOR)
    IMG = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
    fil = cv2.bilateralFilter(img, 5,3,3, borderType = cv2.BORDER_CONSTANT)
    FIL = cv2.bilateralFilter(IMG, 5,3,3, borderType = cv2.BORDER_CONSTANT)
    return img, IMG, fil, FIL

den, DEN, filde, FILDEN = Imp(("Photos/Toothpaste/QueryD.jpg"))
sal, SAL, filsal, FILSAL = Imp(("Photos/Tissue/Query_tissue.jpg"))
tub, TUB, filtub, FILTUB = Imp(("Photos/Stick/Query_front.jpg"))
DEN = cv2.flip(DEN,-1)

#for toothpaste
#series of feature computed offline and valid for each iteration
siftd = cv2.xfeatures2d.SIFT_create()
FLANN_INDEX_KDTREE = 1
MIN_MATCH_COUNT = 100
kp_queryd = siftd.detect(FILDEN)
kp_queryd, des_queryd = siftd.compute(FILDEN, kp_queryd)
tot_matchesden = np.size(kp_queryd)
print(tot_matchesden)
#for tubetto
siftt = cv2.xfeatures2d.SIFT_create()

kp_queryt = siftt.detect(FILTUB)
kp_queryt, des_queryt = siftt.compute(FILTUB, kp_queryt)
tot_matchestub = np.size(kp_queryt)
print(tot_matchestub)
#for salvietta
sifts = cv2.xfeatures2d.SIFT_create()

kp_querys = sifts.detect(FILSAL)
kp_querys, des_querys = siftd.compute(FILSAL, kp_querys)
tot_matchessal = np.size(kp_querys)
print(tot_matchessal)
quality = 0

def valuta_scatola(link):
    quality = 0
    sc, SC, scfil, SCFIL = Imp(link)

    scatOTSU, threshOTSU = cv2.threshold(scfil, 100, 255, cv2.THRESH_OTSU )
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(threshOTSU, connectivity = 8, ltype = cv2.CV_32S)

    max_area_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    max_area = stats[max_area_index, cv2.CC_STAT_AREA]
    x = stats[max_area_index, cv2.CC_STAT_LEFT]
    y = stats[max_area_index, cv2.CC_STAT_TOP]
    width = stats[max_area_index, cv2.CC_STAT_WIDTH]
    heigth = stats[max_area_index, cv2.CC_STAT_HEIGHT]
    SC = cv2.rectangle(SC, (stats[max_area_index, cv2.CC_STAT_LEFT], stats[max_area_index, cv2.CC_STAT_TOP]), (stats[max_area_index, cv2.CC_STAT_LEFT] +stats[max_area_index, cv2.CC_STAT_WIDTH], stats[max_area_index, cv2.CC_STAT_TOP] + stats[max_area_index, cv2.CC_STAT_HEIGHT]), (0, 255, 0), 11)
    
    """
    plt.imshow(SC)
    plt.title("Scatola trovata", color = "Black", fontsize = 16)
    plt.show()
    """
    SCFILb = SCFIL[y:y-heigth, x:x+width]
    SCb = SC[y:y-heigth, x:x+width]
    image_height, image_width = den.shape

    #start detect dentifrici
    den_train = siftd.detect(SC)
    den_train, desden_train = siftd.compute(SC, den_train)
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flannd = cv2.FlannBasedMatcher(index_params, search_params)
    matchesd = flannd.knnMatch(des_queryd,desden_train,k=2)

    goodd = []
    for md,nd in matchesd:
       if md.distance < 0.8*nd.distance: #0.8 degree of freedom, index of acceptability of a keypoint
        goodd.append(md)
    
    print("Toothpaste matches:", len(goodd))
    if len(goodd)>tot_matchesden/10:
    # building the corrspondences arrays of good matches
        src_pts = np.float32([ kp_queryd[md.queryIdx].pt for md in goodd ]).reshape(-1,1,2)
        dst_pts = np.float32([ den_train[md.trainIdx].pt for md in goodd ]).reshape(-1,1,2)
    # Using RANSAC to estimate a robust homography. 
    # It returns the homography M and a mask for the discarded points

        Md, maskd = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 24.0)
         # Projecting the corners into the train image
        
        ptsd = np.float32([ [0,0],[0,image_height-1],[image_width-1,image_height-1],[image_width-1,0] ]).reshape(-1,1,2)
        dstd = cv2.perspectiveTransform(ptsd,Md)
        SC = cv2.polylines(SC,[np.int32(dstd)],True, [0,255,0],7, cv2.LINE_AA)
        quality = 1
    else:
        quality = 0

    
    #start detect tubetti
    
    tub_train = siftt.detect(SC)
    tub_train, tubden_train = siftt.compute(SC, tub_train)
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flannt = cv2.FlannBasedMatcher(index_params, search_params)
    matchest = flannt.knnMatch(des_queryt,tubden_train,k=2)
    image_height, image_width = tub.shape
    goodt = []
    for mt,nt in matchest:
       if mt.distance < 0.8*nt.distance: #0.8 degree of freedom, index of acceptability of a keypoint
        goodt.append(mt)
    
    print("Stick matches:", len(goodt))
    if len(goodt)> tot_matchestub/10:
    # building the corrspondences arrays of good matches
        src_ptst = np.float32([ kp_queryt[mt.queryIdx].pt for mt in goodt ]).reshape(-1,1,2)
        dst_ptst = np.float32([ tub_train[mt.trainIdx].pt for mt in goodt ]).reshape(-1,1,2)
    # Using RANSAC to estimate a robust homography. 
    # It returns the homography M and a mask for the discarded points

        Mt, maskt = cv2.findHomography(src_ptst, dst_ptst, cv2.RANSAC, 24.0)
         # Projecting the corners into the train image
        
        ptst = np.float32([ [0,0],[0,image_height-1],[image_width-1,image_height-1],[image_width-1,0] ]).reshape(-1,1,2)
        dstt = cv2.perspectiveTransform(ptst,Mt)
        SC = cv2.polylines(SC,[np.int32(dstt)],True, [0,255,0],7, cv2.LINE_AA)
        quality = quality +1
    else:
        quality = quality

    #now find the tissue
    
    sal_train = sifts.detect(SC)
    sal_train, salden_train = sifts.compute(SC, sal_train)
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flanns = cv2.FlannBasedMatcher(index_params, search_params)
    matchess = flanns.knnMatch(des_querys,salden_train,k=2)
    image_height, image_width = sal.shape
    goods = []
    for ms,ns in matchess:
       if ms.distance < 0.8*ns.distance: #0.9 degree of freedom, index of acceptability of a keypoint
        goods.append(ms)
    
    print("Tissue matches:", len(goods))
    if len(goods)>MIN_MATCH_COUNT:
    # building the corrspondences arrays of good matches
        src_ptss = np.float32([ kp_querys[ms.queryIdx].pt for ms in goods ]).reshape(-1,1,2)
        dst_ptss = np.float32([ sal_train[ms.trainIdx].pt for ms in goods ]).reshape(-1,1,2)
    # Using RANSAC to estimate a robust homography. 
    # It returns the homography M and a mask for the discarded points

        Ms, masks = cv2.findHomography(src_ptss, dst_ptss, cv2.RANSAC, 24.0)
         # Projecting the corners into the train image
        print(Ms)
        ptss = np.float32([ [0,0],[0,image_height-1],[image_width-1,image_height-1],[image_width-1,0] ]).reshape(-1,1,2)
        dsts = cv2.perspectiveTransform(ptss,Ms)
        SC = cv2.polylines(SC,[np.int32(dsts)],True, [0,255,0],7, cv2.LINE_AA)
        quality = quality +1
    else:
        quality = quality
    
    if quality == 3:
       color_write = "green"
       SRvalue = "SR value : 0000"
       Srout = "SR OUT : True"
       
    else:
       color_write = "red"
       Srout = "SR OUT : False"
       SRvalue = "SR value : 0001"

    plt.tight_layout()
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.wm_geometry("+50+50")
    plt.subplot(1,2,1)
    plt.imshow(SCFIL)
    plt.subplot(1,2,2)
    plt.suptitle("Scatola con {} prodotti all'interno".format(quality), color = color_write, fontsize = 20)
    plt.figtext(0.8, 0.8, SRvalue, fontsize=16, bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.5'))
    plt.figtext(0.8, 0.7, Srout, fontsize=16, bbox=dict(facecolor='white', edgecolor=color_write, boxstyle='round,pad=0.5'))
    plt.imshow(SC)

    plt.show()

    return quality



valuta_scatola("Photos/Box/Box1.jpg")
time.sleep(0.5)
valuta_scatola("Photos/Box/Box2.jpg")
time.sleep(0.5)
valuta_scatola("Photos/Box/Box3.jpg")
time.sleep(0.5)
valuta_scatola("Photos/Box/Box4.jpg")
time.sleep(0.5)
valuta_scatola("Photos/Box/Box5.jpg")
