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
    dst=cv2.dilate(dst, kernel ,iterations=2) # @ Videh, try putting iterations = 2 here, the edges become more prominent, however noise increases to some extent.
    dst = (dst>153).astype(np.uint8)
    ###########  
    dst = dst*255    # New line - dont include for creating ground truths
    ############
    imsave(id_+'.tif', dst)
    #print(dst)
    #plt.imshow(dst, cmap='gray', vmin=0, vmax=1)
    #plt.show()

    

#drawCountours('test_microstructure.jpg','1.jpg')    

#path = 'D:/Videh_Acads/Machine_Learning/data/Sir_Micrographs/micrographs Inconel-20200827T122844Z-001/train_images/'



#Use below block for creating ground truths
'''
path = 'D:/Videh_Acads/Machine_Learning/data/Sir_Micrographs/micrographs Inconel-20200827T122844Z-001/train_images/'
train_ids = next(os.walk(path))[2]
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
    npath = path + id_
    drawCountours(npath, id_)'''


drawCountours("prediction_346.jpg", "prediction_346(1)") 
 
    ###############################################



########################################################
'''#Edge detection below
    image = cv2.imread(path)
    ratio = image.shape[0] / 500.0

    orig = image.copy()
    image = imutils.resize(image, height = 500)
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0) # this is the part where I tweaked the parameters to get to have contours over the entire micrograph
    edged = cv2.Canny(gray, 75, 200)
    #plt.imshow(edged)
    #plt.show()

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:]

    whiteframe = 255* np.ones((500,666,3), np.uint8)

    #looping over the entire contour
    for c in cnts:
        perimeter=cv2.arcLength(c,True)
        approx_curve = cv2.approxPolyDP(c,0.02*perimeter, True)
        cv2.drawContours(whiteframe, [approx_curve], -1, (0, 0, 0), 0, cv2.LINE_AA, 0, 0)

    #image1 = 255*((image!=0).astype(np.uint8))

    #print(image1)
    
    #image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY, cmap='gray', vmin=0, vmax=255)
    #image1 = image1[:,:,0]
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    imgplot = plt.imshow(orig)
    ax.set_title('Before')
    #plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
    ax = fig.add_subplot(1, 3, 2)
    imgplot = plt.imshow(image)
    #imgplot.set_clim(0.0, 0.7)
    ax.set_title('Countour lines')
    #plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
    ax = fig.add_subplot(1, 3, 3)
    imgplot = plt.imshow(image1, cmap='gray', vmin=0, vmax=255)
    ax.set_title('After')
    plt.show()'''    
   