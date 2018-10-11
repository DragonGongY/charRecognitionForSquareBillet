import cv2
import numpy as np

img = cv2.imread('/home/gsy/155.jpg')
ret, th = cv2.threshold(img, 0, 127, cv2.THRESH_TOZERO)
cv2.imshow('thresh', th)
cv2.waitKey(0)
cv2.destroyAllWindows()