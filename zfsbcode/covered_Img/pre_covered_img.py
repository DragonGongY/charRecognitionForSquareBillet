import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
'''Adujust image gray'''
def scale_image_range(img,low,uper):
    ScaleImage = img
    MN = np.shape(img)
    for i in range(0, MN[0]):
        for j in range(0, MN[1]):
            if img[i][j] < low:
                ScaleImage[i][j] = 0
            elif img[i][j] > uper:
                ScaleImage[i][j] = 1
            else:
                ScaleImage[i][j] = (img[i][j] - low)/(uper - low)
    return ScaleImage

def Sub_Image(img1,img2,offset):
    SubImage = img1 - img2 + offset
    MN = np.shape(img1)
    for i in range(0,MN[0]-1,1):
        for j in range(0,MN[1] -1,1):
            if  SubImage[i][j] < 0:
                SubImage[i][j] = 0
    return SubImage

def Pre(Image,showflag=0):
    if showflag == 0:
        Image = cv2.imread(Image, 1)
        Image = cv2.cvtColor(Image, cv2.COLOR_RGB2BGR)
        G_gray = np.float32(Image[:, :, 2])
        G_Mean1 = cv2.blur(G_gray, (21, 21))
        G_Sub1 = Sub_Image(G_gray, G_Mean1, 15)
        G_Mean2 = cv2.blur(G_Sub1, (27, 27))
        G_Sub2 = Sub_Image(G_Sub1, G_Mean2, 15)
        G_Median = cv2.medianBlur(G_Sub2, 3)
        PreImage = scale_image_range(G_Median, 20, 60)
        PreImage = cv2.cvtColor(PreImage, cv2.COLOR_GRAY2BGR)
    if showflag == 1 :
        Image = cv2.cvtColor(Image, cv2.COLOR_RGB2BGR)
        G_gray = np.float32(Image[:, :, 2])
        G_Mean1 = cv2.blur(G_gray, (21, 21))
        G_Sub1 = Sub_Image(G_gray, G_Mean1, 15)
        G_Mean2 = cv2.blur(G_Sub1, (27, 27))
        G_Sub2 = Sub_Image(G_Sub1, G_Mean2, 15)
        G_Median = cv2.medianBlur(G_Sub2, 3)
        PreImage = scale_image_range(G_Median, 20, 60)
        PreImage = cv2.cvtColor(PreImage, cv2.COLOR_GRAY2BGR)
    return PreImage

# def Pre(Image,showflag=0):
#     # extension = os.path.splitext(ImageName)
#     # if extension == ".jpg":
#     Image = cv2.imread(Image, 1)
#     Image = cv2.cvtColor(Image, cv2.COLOR_RGB2BGR)
#     G_gray = np.float32(Image[:, :, 2])
#     G_Mean1 = cv2.blur(G_gray, (21, 21))
#     G_Sub1 = Sub_Image(G_gray, G_Mean1, 15)
#     G_Mean2 = cv2.blur(G_Sub1, (27, 27))
#     G_Sub2 = Sub_Image(G_Sub1, G_Mean2, 15)
#     G_Median = cv2.medianBlur(G_Sub2, 3)
#     PreImage = scale_image_range(G_Median, 20, 60)
#     PreImage = cv2.cvtColor(PreImage, cv2.COLOR_GRAY2BGR)
#     # else:
#     #     print 'There is no image!'
#     #     return -1
#     if showflag == 1 :
#         cv2.namedWindow("Image",0)
#         cv2.imshow("Image",PreImage)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     return PreImage
