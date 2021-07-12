import cv2
import imutils
from matplotlib import pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import skimage.io
from skimage.io import imsave, imread
from skimage.filters import threshold_sauvola

def drawCountours(path, id_):
    #Edge detection below
    image = cv2.imread(path)
    ratio = image.shape[0] / 500.0

    orig = image.copy()
    image = imutils.resize(image, height = 500)
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0) # this is the part where I tweaked the parameters to get to have contours over the entire micrograph
    edged = cv2.Canny(gray, 75, 200)
    orig=gray.copy()
    T_sauvola = threshold_sauvola(orig, 21)
    orig = (orig > T_sauvola).astype("uint8") * 255
    gray=orig

    kernel=np.ones((3,3),np.uint8)

    gray=cv2.erode(gray, kernel ,iterations=1)
    gray=cv2.dilate(gray, kernel ,iterations=1)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:]
    #looping over the entire contour
    gray = cv2.GaussianBlur(gray, (15, 15), 0) # this is the part where I tweaked the parameters to get to have contours over the entire micrograph
    edged = cv2.Canny(gray, 130, 140)

    for c in cnts:
        perimeter=cv2.arcLength(c,True)
        approx_curve = cv2.approxPolyDP(c,0.02*perimeter, True)
        cv2.drawContours(image, [approx_curve], -1, (0, 255, 0), 2)
    res = cv2.morphologyEx(edged,cv2.MORPH_CLOSE, kernel)
    invert_image=cv2.bitwise_not(res)

    #plt.imshow(image)
    dst = cv2.addWeighted(orig, 0.4, invert_image, 0.6, 0.0)
    dst=cv2.erode(dst, kernel ,iterations=1)
    dst=cv2.dilate(dst, kernel ,iterations=1)
    dst=cv2.erode(dst, kernel ,iterations=1)
    dst=cv2.dilate(dst, kernel ,iterations=2) # try putting iterations = 2 here, the edges become more prominent, however noise increases to some extent.
    dst = (dst>153).astype(np.uint8)
    imsave(id_+'.tif', dst)
    #print(dst)
    #plt.imshow(dst, cmap='gray', vmin=0, vmax=1)
    #plt.show()
