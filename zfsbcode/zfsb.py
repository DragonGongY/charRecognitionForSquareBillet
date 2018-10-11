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


loc_model_def = '/media/gsy/D02497132496FC22/Users/Administrator/Desktop/zfsb/location/deploy.prototxt'
loc_model_weights = '/media/gsy/D02497132496FC22/Users/Administrator/Desktop/zfsb/location/VGG_VOC0712_SSD_512x512_iter_116000.caffemodel'

net_loc = caffe.Net(loc_model_def,  # The structure of the model
                loc_model_weights,  # The trained weights
                caffe.TEST)  # Use test mode

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer_loc = caffe.io.Transformer({'data': net_loc.blobs['data'].data.shape})
transformer_loc.set_transpose('data', (2, 0, 1))
transformer_loc.set_mean('data', np.array([104, 117, 123]))  # mean pixel
transformer_loc.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer_loc.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

labelmap_file_loc = '/media/gsy/D02497132496FC22/Users/Administrator/Desktop/zfsb/location/labelmap_voc.prototxt'
file1 = open(labelmap_file_loc, 'r')
labelmap_loc = caffe_pb2.LabelMap()
text_format.Merge(str(file1.read()), labelmap_loc)

# load an image
image_file = '/home/gsy/aaaaaaa/1.jpg'

img_pre_covered = Pre(image_file, 0)
#img_pre_covered = img_pre_covered/255.
# cv2.imshow('pre_covered', img_pre_covered)
# cv2.waitKey()
# cv2.destroyAllWindows()

image_resize = 300
net_loc.blobs['data'].reshape(1, 3, image_resize, image_resize)
#image = caffe.io.load_image(image_file)

# Run the net and examine the top_k results
transformed_image_loc = transformer_loc.preprocess('data', img_pre_covered)
net_loc.blobs['data'].data[...] = transformed_image_loc

# Forward pass.
detections_loc = net_loc.forward()['detection_out']

# Parse the outputs.
det_label_loc = detections_loc[0, 0, :, 1]
det_conf_loc = detections_loc[0, 0, :, 2]
det_xmin_loc = detections_loc[0, 0, :, 3]
det_ymin_loc = detections_loc[0, 0, :, 4]
det_xmax_loc = detections_loc[0, 0, :, 5]
det_ymax_loc = detections_loc[0, 0, :, 6]

# Get detections with confidence higher than 0.6.
top_indices_loc = [i for i, conf in enumerate(det_conf_loc) if conf >= 0.6]

top_conf_loc = det_conf_loc[top_indices_loc]
top_label_indices_loc = det_label_loc[top_indices_loc].tolist()
top_labels_loc = get_labelname(labelmap_loc, top_label_indices_loc)
top_xmin_loc = det_xmin_loc[top_indices_loc]
top_ymin_loc = det_ymin_loc[top_indices_loc]
top_xmax_loc = det_xmax_loc[top_indices_loc]
top_ymax_loc = det_ymax_loc[top_indices_loc]

result_loc = []
for i in xrange(min(5, top_conf_loc.shape[0])):
    xmin_loc = top_xmin_loc[i]  # xmin = int(round(top_xmin[i] * image.shape[1]))
    ymin_loc = top_ymin_loc[i]  # ymin = int(round(top_ymin[i] * image.shape[0]))
    xmax_loc = top_xmax_loc[i]  # xmax = int(round(top_xmax[i] * image.shape[1]))
    ymax_loc = top_ymax_loc[i]  # ymax = int(round(top_ymax[i] * image.shape[0]))
    score_loc = top_conf_loc[i]
    label_loc = int(top_label_indices_loc[i])
    label_name_loc = top_labels_loc[i]
    result_loc.append([xmin_loc, ymin_loc, xmax_loc, ymax_loc, label_loc, score_loc, label_name_loc])

print 'results_loc',result_loc

height_loc, width_loc ,_ = img_pre_covered.shape
print width_loc,height_loc
for item in result_loc:
    xmin_loc = int(round(item[0] * width_loc))
    ymin_loc = int(round(item[1] * height_loc))
    xmax_loc = int(round(item[2] * width_loc))
    ymax_loc = int(round(item[3] * height_loc))
    print item
    print [xmin_loc, ymin_loc, xmax_loc, ymax_loc]
    print [xmin_loc, ymin_loc], item[-1]

def Cut_Image(img,xmin,ymin,xmax,ymax,offset=4):
    CutImage = img[(ymin - offset):(ymax + offset),(xmin - offset):(xmax + offset)]
    return CutImage

image = cv2.imread(image_file)
iimg = Cut_Image(image, xmin_loc, ymin_loc, xmax_loc, ymax_loc)
print iimg.shape
iimg = Pre(iimg, 1)
cv2.imshow('cut', iimg)
# cv2.waitKey()
# cv2.destroyAllWindows()

##### turn pictures to the formal direction ######
MODEL_FILE_TURN = '/media/gsy/D02497132496FC22/Users/Administrator/Desktop/zfsb/turn_Pic/deploy.prototxt'
PRETRAINED_TURN = '/media/gsy/D02497132496FC22/Users/Administrator/Desktop/zfsb/turn_Pic/snapshot_iter_8551.caffemodel'

net_turn = caffe.Net(MODEL_FILE_TURN,
                PRETRAINED_TURN,
                caffe.TEST)

mean_turn = np.load('/media/gsy/D02497132496FC22/Users/Administrator/Desktop/zfsb/turn_Pic/mean.npy')
mean_turn = mean_turn.mean(1).mean(1)
#print 'mean-subtracted values:', zip('BGR', mu)

transformer_turn = caffe.io.Transformer({'data': net_turn.blobs['data'].data.shape})

transformer_turn.set_transpose('data', (2,0,1))
transformer_turn.set_mean('data', mean_turn)
transformer_turn.set_raw_scale('data', 255)
transformer_turn.set_channel_swap('data', (2,1,0))

net_turn.blobs['data'].reshape(50,
                          3,
                          128, 128)

transformed_image_turn = transformer_turn.preprocess('data', iimg)

net_turn.blobs['data'].data[...] = transformed_image_turn

output = net_turn.forward()

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
    img_normal = rotateImg(iimg, 180)
    cv2.imshow('turn', img_normal)
elif int(output_prob.argmax()) == 1:
    img_normal = rotateImg(iimg, 270)
    cv2.imshow('turn', img_normal)
elif int(output_prob.argmax()) == 2:
    img_normal = rotateImg(iimg, 90)
    cv2.imshow('turn', img_normal)
else:
    img_normal = iimg
    cv2.imshow('turn', img_normal)

# cv2.waitKey()
# cv2.destroyAllWindows()

##### display numbers#####
model_def_rec = '/media/gsy/D02497132496FC22/Users/Administrator/Desktop/zfsb/recognition/deploy.prototxt'
model_weights_rec = '/media/gsy/D02497132496FC22/Users/Administrator/Desktop/zfsb/recognition/VGG_VOC0712_SSD_300x300_iter_81000.caffemodel'

labelmap_file_rec = '/media/gsy/D02497132496FC22/Users/Administrator/Desktop/zfsb/recognition/labelmap_voc.prototxt'
file2 = open(labelmap_file_rec, 'r')
labelmap_rec = caffe_pb2.LabelMap()
text_format.Merge(str(file2.read()), labelmap_rec)

net_rec = caffe.Net(model_def_rec,  # defines the structure of the model
                model_weights_rec,  # contains the trained weights
                caffe.TEST)  # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer_rec = caffe.io.Transformer({'data': net_rec.blobs['data'].data.shape})
transformer_rec.set_transpose('data', (2, 0, 1))
transformer_rec.set_mean('data', np.array([104,117,123]))  # mean pixel
transformer_rec.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer_rec.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

net_rec.blobs['data'].reshape(1, 3, image_resize, image_resize)

transformed_image_rec = transformer_rec.preprocess('data', img_normal)
net_rec.blobs['data'].data[...] = transformed_image_rec

# Forward pass.
detections_rec = net_rec.forward()['detection_out']

# Parse the outputs.
det_label_rec = detections_rec[0, 0, :, 1]
det_conf_rec = detections_rec[0, 0, :, 2]
det_xmin_rec = detections_rec[0, 0, :, 3]
det_ymin_rec = detections_rec[0, 0, :, 4]
det_xmax_rec = detections_rec[0, 0, :, 5]
det_ymax_rec = detections_rec[0, 0, :, 6]

# Get detections with confidence higher than 0.6.
top_indices_rec = [i for i, conf in enumerate(det_conf_rec) if conf >= 0.5]

top_conf_rec = det_conf_rec[top_indices_rec]
top_label_indices_rec = det_label_rec[top_indices_rec].tolist()
top_labels_rec = get_labelname(labelmap_rec, top_label_indices_rec)
top_xmin_rec = det_xmin_rec[top_indices_rec]
top_ymin_rec = det_ymin_rec[top_indices_rec]
top_xmax_rec = det_xmax_rec[top_indices_rec]
top_ymax_rec = det_ymax_rec[top_indices_rec]

colors = plt.cm.hsv(np.linspace(0, 1, 37)).tolist()

plt.imshow(img_normal)
currentAxis = plt.gca()


center = []
labels = []
for i in xrange(top_conf_rec.shape[0]):

    xmin = int(round(top_xmin_rec[i] * img_normal.shape[1]))
    ymin = int(round(top_ymin_rec[i] * img_normal.shape[0]))
    xmax = int(round(top_xmax_rec [i] * img_normal.shape[1]))
    ymax = int(round(top_ymax_rec[i] * img_normal.shape[0]))
    score_rec = top_conf_rec[i]
    label = int(top_label_indices_rec[i])
    label_name_rec = top_labels_rec[i]
    display_txt = '%s: %.2f'%(label_name_rec, score_rec)
    coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
    color = colors[label]
    currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
    currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':1})

    center7 = [ymin + (ymax - ymin) / 2., xmin + (xmax - xmin) / 2.]
    center.append(center7)
    labels.append(label_name_rec)

# Sorted the center point by y-axis
sorted_letters = sorted(center)

# The first four points is upper letters,another four points is lower letters
upper_letters = []
lower_letters = []

item = 0 # First point is upper
height, width, _ = image.shape
for num in xrange(len(sorted_letters)):
    if num == 0:
        upper_letters.append(sorted_letters[num])
    elif abs(sorted_letters[item][0]-sorted_letters[num][0]) < 20:
        upper_letters.append(sorted_letters[num])
    else:
        lower_letters.append(sorted_letters[num])

for num_upper in range(0, len(upper_letters)):
    upper_letters[num_upper].reverse()

for num_lower in range(0, len(lower_letters)):
    lower_letters[num_lower].reverse()

# Sort the upper letters by x-axis, so do the lower letters
upper_letters_sort = sorted(upper_letters)
lower_letters_sort = sorted(lower_letters)

upper_center_sum = [sum(upper_letters_sort[upper]) for upper in range(len(upper_letters_sort))]
lower_center_sum = [sum(lower_letters_sort[lower]) for lower in range(len(lower_letters_sort))]

center = [sum(center[sumation]) for sumation in range(len(center))]
center_labels_zipped_dict = dict(zip(center, labels))

# print the ordered letters
upper_ordered=[0 for i in range(len(upper_center_sum))]
lower_ordered=[0 for i in range(len(lower_center_sum))]
for keys in center_labels_zipped_dict.keys():
    if keys in upper_center_sum:
        for upper_val in range(len(upper_center_sum)):
            if keys == upper_center_sum[upper_val]:
               upper_ordered[upper_val] = center_labels_zipped_dict[keys]
    else:
        for lower_val in range(len(lower_center_sum)):
            if keys == lower_center_sum[lower_val]:
                lower_ordered[lower_val] = center_labels_zipped_dict[keys]

print 'upper letters:',upper_ordered
print 'lower letters:',lower_ordered



plt.savefig('/home/gsy/aaaaaaa/results.jpg')
plt.close()

cv2.waitKey()
cv2.destroyAllWindows()
