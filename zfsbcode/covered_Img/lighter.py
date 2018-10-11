#-*- coding: utf-8 -*-
import cv2
import numpy as np
fn = "test1.jpg"
if __name__ =='__main__':
    print('loading... %s' % fn)
    print (u'显示原图')
    img = cv2.imread(fn)
    cv2.namedWindow('source')
    cv2.imshow('source',img)

    print (u'正在处理中')
    w = img.shape[1]
    h = img.shape[0]

    # 全部变暗
    for xi in range(0,w):
        for xj in range(0,h):
            #将像素值整体减少，设为原像素值的20%
            img[xj,xi,0]=int(img[xj,xi,0]*0.2)
            img[xj,xi,1]=int(img[xj,xi,1]*0.2)
            img[xj,xi,2]=int(img[xj,xi,2]*0.2)
        # 显示进度条
        if xi%10 ==0 :
            print('.')
    cv2.namedWindow('dark')
    cv2.imshow('dark', img)
    cv2.waitKey()
    cv2.destroyAllWindows()