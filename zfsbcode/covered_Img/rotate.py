import numpy as np
import cv2

img = cv2.imread('/home/gsy/cut40.jpg')
img = img.rotate(90)
cv2.imshow('ha1',img)
cv2.waitKey()