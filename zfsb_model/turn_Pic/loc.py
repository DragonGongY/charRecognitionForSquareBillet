# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import cv2

import caffe

MODEL_FILE = '/media/gsy/D02497132496FC22/Users/Administrator/Desktop/zfsb/turn_Pic/deploy.prototxt/deploy.prototxt'
PRETRAINED = '/media/gsy/D02497132496FC22/Users/Administrator/Desktop/zfsb/turn_Pic/deploy.prototxt/snapshot_iter_8551.caffemodel'

# load the model
caffe.set_mode_cpu()
net = caffe.Net(MODEL_FILE,
                PRETRAINED,
                caffe.TEST)

mu = np.load('/media/gsy/D02497132496FC22/Users/Administrator/Desktop/zfsb/turn_Pic/deploy.prototxt/mean.npy')
mu = mu.mean(1).mean(1)
#print 'mean-subtracted values:', zip('BGR', mu)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', mu)
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

net.blobs['data'].reshape(50,
                          3,
                          128, 128)

#image = caffe.io.load_image('1.jpg')
image = cv2.imread('/home/gsy/pic/3.jpg')
transformed_image = transformer.preprocess('data', image)
#plt.imshow(image)

net.blobs['data'].data[...] = transformed_image

output = net.forward()

output_prob = output['softmax'][0]

# D(0) L(1) R(2) T(3)
print 'predicted class is:', output_prob.argmax()
# output_prob = output['prob'].data[0]
#
# # # load ImageNet labels
# labels_file = 'C:/Users/Administrator/Desktop/zfsb/turn_Pic/labels.txt'
# labels = np.loadtxt(labels_file, str, delimiter=',')
# print labels
# # print 'output label:', labels[output_prob.argmax()]

def rotateImg(inputImg, angle):
    center_x, center_y, channels = np.shape(inputImg)
    M = cv2.getRotationMatrix2D((center_x / 2, center_y / 2), angle, 1)
    rotate = cv2.warpAffine(inputImg, M, (center_y, center_x))
    return rotate

if int(output_prob.argmax()) == 0:
    img = rotateImg(image, 180)
    cv2.imwrite('D_T.jpg', img)
elif int(output_prob.argmax()) == 1:
    img = rotateImg(image, 270)
    cv2.imwrite('L_T.jpg', img)
elif int(output_prob.argmax()) == 2:
    img = rotateImg(image, 90)
    cv2.imwrite('R_T.jpg', img)
else:
    pass

