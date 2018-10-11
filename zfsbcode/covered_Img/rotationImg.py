import cv2
import numpy as np
def rotateImg(inputImg, angle):
    # center_x, center_y, channels = np.shape(inputImg)
    # X = float(center_x )/ 2
    # Y = float(center_y )/ 2
    # M = cv2.getRotationMatrix2D((X,Y), angle, 1)
    # rotate = cv2.warpAffine(inputImg, M, (center_x, center_y))
    # return rotate
    if angle == 90:
        rotate = np.rot90(inputImg)
    elif angle == 180:
        rotate = np.rot90(np.rot90(inputImg))
    elif angle == 270:
        rotate = np.rot90(np.rot90(np.rot90(inputImg)))
    else:
        pass

    return rotate

# img = cv2.imread('/home/gsy/cut48.jpg')
# img = rotateImg(img, 270)
# cv2.namedWindow('ssss', 0)
# cv2.imshow('ssss', img)
# cv2.waitKey()
# cv2.destroyAllWindows()