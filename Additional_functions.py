# NOTE: some of those functions could be unused

import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import time 

def dcplot(img1, img2, SRvalue, SRout, box, pr):
    if SRout == True:
        color_write = "green"
    elif SRout == False:
        color_write = "red"

    plt.tight_layout()
    #fig_manager = plt.get_current_fig_manager()
    #fig_manager.window.wm_geometry("+50+50")
    plt.subplot(1,2,1)
    plt.imshow(img1)
    plt.subplot(1,2,2)
    if box == True:
        plt.suptitle("Scatola con {} prodotti all'interno".format(pr), color = color_write, fontsize = 20)
    elif box == False:
        if SRvalue == '0000':
            plt.suptitle("Prodotto ok", color = "green", fontsize = 20)
        if SRvalue == '0001':
            plt.suptitle("Prodotto da girare", color = "orange", fontsize = 20)
        if SRvalue == '0100' or SRvalue == '0010':
            plt.suptitle("Prodotto difettoso", color = "red")

    plt.figtext(0.8, 0.8, SRvalue, fontsize=16, bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.5'))
    plt.figtext(0.8, 0.7, SRout, fontsize=16, bbox=dict(facecolor='white', edgecolor=color_write, boxstyle='round,pad=0.5'))
    plt.imshow(img2)

    plt.show()



def Imp(link):
    img = cv2.imread(link, cv2.IMREAD_GRAYSCALE)
    IMG = cv2.imread(link, cv2.IMREAD_COLOR)
    IMG = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
    fil = cv2.bilateralFilter(img, 5,10,5, borderType = cv2.BORDER_CONSTANT)
    FIL = cv2.bilateralFilter(IMG, 5,10,5, borderType = cv2.BORDER_CONSTANT)
    return img, IMG, fil, FIL


def Opener(image, size):
    h,w = np.shape(image)
    opimg = np.copy(image)
    for i in range (size, h -size):
        for j in range(size, w -size):
            if image[i,j] == 255:
                opimg[i-size:i+size,j-size:j+size] = 255
                #i = i +size
                #j = j+size

    return opimg


def cancel_borders(image_to_correct, image_giving_points):
    h,w = np.shape(image_giving_points)
    new_img = np.copy(image_to_correct)
    for i in range(0, w,1):
        for j in range(0,h,1):
            if image_giving_points[j,i] >= 254:
                new_img[j,i] = [0,0,0]

    return new_img


def dcsplot(img1, img2, SRvalue, SRout, angle, clockwise = True):
    if SRout == True:
        color_write = "green"
    elif SRout == False:
        color_write = "red"

    plt.tight_layout()
    #fig_manager = plt.get_current_fig_manager()
    #fig_manager.window.wm_geometry("+50+50")
    plt.subplot(1,2,1)
    plt.imshow(img1)
    plt.subplot(1,2,2)
    if SRvalue == '0000':
        plt.suptitle("Product OK", color = 'green')
    elif SRvalue == '1000':
        plt.suptitle("No product", color = 'red')
    elif SRvalue == '0001':
        if clockwise == True:
            plt.suptitle("The product must be turned of {} degrees clockwise".format(angle), color = "orange", fontsize = 20)
        elif clockwise == False:
            plt.suptitle("The product must be turned of {} degrees counterclockwise".format(angle), color = "orange", fontsize = 20)
    #elif SRvalue == '0010':
    #    plt.suptitle("Il prodotto non è integro")
    elif SRvalue == '0100':
        plt.suptitle("Not intact product", color = 'red')
    elif SRvalue == '0010':
        plt.suptitle("Defective product", color = 'red')
    plt.figtext(0.8, 0.8, SRvalue, fontsize=16, bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.5'))
    plt.figtext(0.8, 0.7, SRout, fontsize=16, bbox=dict(facecolor='white', edgecolor=color_write, boxstyle='round,pad=0.5'))
    plt.imshow(img2)

    plt.show()


def dcblot(img1, img2, SRvalue, SRout, pr):
    if SRout == True:
        color_write = "green"
    elif SRout == False:
        color_write = "red"

    plt.tight_layout()
    #fig_manager = plt.get_current_fig_manager()
    #fig_manager.window.wm_geometry("+50+50")
    plt.subplot(1,2,1)
    plt.imshow(img1)
    plt.subplot(1,2,2)
    
    plt.suptitle("Scatola con {} prodotti all'interno".format(pr), color = color_write, fontsize = 20)
    

    plt.figtext(0.8, 0.8, SRvalue, fontsize=16, bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.5'))
    plt.figtext(0.8, 0.7, SRout, fontsize=16, bbox=dict(facecolor='white', edgecolor=color_write, boxstyle='round,pad=0.5'))
    plt.imshow(img2)

    plt.show()


def Isolate_Color(pic, int1, int2):
    out = np.copy(pic)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    h, w = np.shape(out)

    out = np.where(pic[:] > int1 and pic[:] < int2, 255, out)
    out = np.where(pic[:] < int1 or pic[:] > int2, 0, out)
    #for i in range (0, h, 1):
        #for j in range (0, j, 1):
            #if pic[i, j, 0] < int2[0] and pic[i, j, 0] > int2[0] and
    
    return out

def dcpplot(img1, img2, SRvalue, SRout, flip = False):
    if SRout == True:
        color_write = "green"
    elif SRout == False:
        color_write = "red"

    plt.tight_layout()
    #fig_manager = plt.get_current_fig_manager()
    #fig_manager.window.wm_geometry("+50+50")
    plt.subplot(1,2,1)
    plt.imshow(img1)
    plt.subplot(1,2,2)
    if SRvalue == '0000':
        if flip == True:
            plt.suptitle("Product must be rotated by 180 degrees", color = 'yellow')
        else:
            plt.suptitle("Product OK", color = 'green')
    elif SRvalue == '1000':
        plt.suptitle("NO product", color = 'red')
    elif SRvalue == '0001':
        plt.suptitle("The product must be turned of 180 degrees", color = 'orange')
    #elif SRvalue == '0010':
    #    plt.suptitle("Il prodotto non è integro")
    elif SRvalue == '0100':
        plt.suptitle("Product not compliant ", color = 'red')
    elif SRvalue == '0010':
        plt.suptitle("Defective product", color = 'red')
    plt.figtext(0.8, 0.8, SRvalue, fontsize=16, bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.5'))
    plt.figtext(0.8, 0.7, SRout, fontsize=16, bbox=dict(facecolor='white', edgecolor=color_write, boxstyle='round,pad=0.5'))
    plt.imshow(img2)

    plt.show()


def compute_displacement(points1, points2):
    
    # Compute the mean displacement
    displacement = np.mean(points2 - points1, axis=0)
    
    return displacement


def enlarge_borders(edge, size):
    borders = []
    output = np.copy(edge)
    h,w =  np.shape(edge)
    for i in range(0, h, 1):
        for j in range(size, w-size, 1):
            if edge[i,j] == 255:
                output[i,j-size:j+size] = 255
                break

        for j in range(w-size, size, -1):
            if edge[i,j] == 255:
                output[i,j-size:j+size] = 255
                break


    return output