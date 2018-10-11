import cv2
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# Set caffe path and Load caffe into python
caffe_root = '/home/gsy/caffe-ssd/'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2
from covered_Img.pre_covered_img import *
from covered_Img.rotationImg import *

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames


# model_def = '/media/gsy/D02497132496FC22/Users/Administrator/Desktop/zfsb/location/deploy.prototxt'
# model_weights = '/media/gsy/D02497132496FC22/Users/Administrator/Desktop/zfsb/location/VGG_VOC0712_SSD_512x512_iter_116000.caffemodel'
#
# net = caffe.Net(model_def,  # The structure of the model
#                 model_weights,  # The trained weights
#                 caffe.TEST)  # Use test mode
#
# # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
# transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# transformer.set_transpose('data', (2, 0, 1))
# transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
# transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
# transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

labelmap_file = '/media/gsy/D02497132496FC22/Users/Administrator/Desktop/zfsb/location/labelmap_voc.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)
#
# def detect(image_file, conf_thresh=0.5, topn=5):
#     image_resize = 300
#     net.blobs['data'].reshape(1, 3, image_resize, image_resize)
#     image = caffe.io.load_image(image_file)
#
#     # Run the net and examine the top_k results
#     transformed_image = transformer.preprocess('data', image)
#     net.blobs['data'].data[...] = transformed_image
#
#     # Forward pass.
#     detections = net.forward()['detection_out']
#
#     # Parse the outputs.
#     det_label = detections[0, 0, :, 1]
#     det_conf = detections[0, 0, :, 2]
#     det_xmin = detections[0, 0, :, 3]
#     det_ymin = detections[0, 0, :, 4]
#     det_xmax = detections[0, 0, :, 5]
#     det_ymax = detections[0, 0, :, 6]
#
#     # Get detections with confidence higher than 0.6.
#     top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]
#
#     top_conf = det_conf[top_indices]
#     top_label_indices = det_label[top_indices].tolist()
#     top_labels = get_labelname(labelmap, top_label_indices)
#     top_xmin = det_xmin[top_indices]
#     top_ymin = det_ymin[top_indices]
#     top_xmax = det_xmax[top_indices]
#     top_ymax = det_ymax[top_indices]
#
#     result = []
#     for i in xrange(min(topn, top_conf.shape[0])):
#         xmin = top_xmin[i]  # xmin = int(round(top_xmin[i] * image.shape[1]))
#         ymin = top_ymin[i]  # ymin = int(round(top_ymin[i] * image.shape[0]))
#         xmax = top_xmax[i]  # xmax = int(round(top_xmax[i] * image.shape[1]))
#         ymax = top_ymax[i]  # ymax = int(round(top_ymax[i] * image.shape[0]))
#         score = top_conf[i]
#         label = int(top_label_indices[i])
#         label_name = top_labels[i]
#         result.append([xmin, ymin, xmax, ymax, label, score, label_name])
#     return result

# for i in range(5476, 23455):
	# image_file = '/home/gsy/JPEGImages/{}.jpg'.format(i)
	# result = detect(image_file)
	# image1 = cv2.imread(image_file)
	# print 'results',result
    #
	# height, width,_ = image1.shape
	# print width,height
	# for item in result:
    	# 	xmin = int(round(item[0] * width))
    	# 	ymin = int(round(item[1] * height))
    	# 	xmax = int(round(item[2] * width))
    	# 	ymax = int(round(item[3] * height))
    	# 	print item
    	# 	print [xmin, ymin, xmax, ymax]
    	# 	print [xmin, ymin], item[-1]
    #
	# def Cut_Image(img,xmin,ymin,xmax,ymax,offset=4):
	# 	CutImage = img[(ymin - offset):(ymax + offset),(xmin - offset):(xmax + offset)]
	# 	return CutImage
    #
    	# iimg = Cut_Image(image1, xmin, ymin, xmax, ymax)
    	# cv2.imwrite('/media/gsy/3FABB804EC984091/img/{}.jpg'.format(i), iimg)

    	# print 'the {}th pictures'.format(i)


##### turn pictures to the formal direction ######
MODEL_FILE = '/media/gsy/D02497132496FC22/Users/Administrator/Desktop/zfsb/turn_Pic/deploy.prototxt'
PRETRAINED = '/media/gsy/D02497132496FC22/Users/Administrator/Desktop/zfsb/turn_Pic/snapshot_iter_8551.caffemodel'

# # load the model
# caffe.set_device(0)
# caffe.set_mode_gpu()
net = caffe.Net(MODEL_FILE,
                PRETRAINED,
                caffe.TEST)

mu = np.load('/media/gsy/D02497132496FC22/Users/Administrator/Desktop/zfsb/turn_Pic/mean.npy')
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

picfiles = os.listdir('/home/gsy/pic/')
for p in picfiles:
    image1 = caffe.io.load_image('/home/gsy/pic/' + p)
    # image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    # image = image/255.
    transformed_image = transformer.preprocess('data', image1)

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

    if int(output_prob.argmax()) == 0:
        image = rotateImg(image1, 180)
        # cv2.imwrite('/home/gsy/pic/{}.jpg'.format(i), img)
    elif int(output_prob.argmax()) == 1:
        image = rotateImg(image1, 270)
        # cv2.imwrite('/home/gsy/pic/{}.jpg'.format(i), img)
    elif int(output_prob.argmax()) == 2:
        image = rotateImg(image1, 90)
        # cv2.imwrite('/home/gsy/pic/{}.jpg'.format(i), img)
    else:
        pass
        # cv2.imwrite('/home/gsy/pic/{}.jpg'.format(i),image1)

##### display numbers#####
    model_def1 = '/media/gsy/D02497132496FC22/Users/Administrator/Desktop/zfsb/recognition/deploy.prototxt'
    model_weights1 = '/media/gsy/D02497132496FC22/Users/Administrator/Desktop/zfsb/recognition/VGG_VOC0712_SSD_300x300_iter_81000.caffemodel'

# labelmap_file = '/media/gsy/D02497132496FC22/Users/Administrator/Desktop/zfsb/location/labelmap_voc.prototxt'
# file = open(labelmap_file, 'r')
# labelmap = caffe_pb2.LabelMap()
# text_format.Merge(str(file.read()), labelmap)

    net1 = caffe.Net(model_def1,      # defines the structure of the model
                model_weights1,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer1 = caffe.io.Transformer({'data': net1.blobs['data'].data.shape})
    transformer1.set_transpose('data', (2, 0, 1))
    transformer1.set_mean('data', np.array([104,117,123])) # mean pixel
    transformer1.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer1.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# set net to batch size of 1
    image_resize = 300
    net1.blobs['data'].reshape(1,3,image_resize,image_resize)

    transformed_image1 = transformer1.preprocess('data', image)
    net1.blobs['data'].data[...] = transformed_image1

# Forward pass.
    detections = net1.forward()['detection_out']

# Parse the outputs.
    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]

# Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = plt.cm.hsv(np.linspace(0, 1, 37)).tolist()

    plt.show(image)
    currentAxis = plt.gca()

    for i in xrange(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * image.shape[1]))
        ymin = int(round(top_ymin[i] * image.shape[0]))
        xmax = int(round(top_xmax[i] * image.shape[1]))
        ymax = int(round(top_ymax[i] * image.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = top_labels[i]
        display_txt = '%s: %.2f'%(label_name, score)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':1})

    plt.savefig('/home/gsy/rest/' + p)
    plt.close()