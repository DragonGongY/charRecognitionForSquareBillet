import os
import cv2
import numpy as np
def Cut_Image(img,xmin,ymin,xmax,ymax,offset=3):
    CutImage = img[(ymin - offset):(ymax + offset),(xmin - offset):(xmax + offset)]
    return CutImage

img = cv2.imread('/home/gsy/1.jpg')
iimg = Cut_Image(img, 300, 21, 397, 107)
cv2.imwrite('aa.jpg', iimg)