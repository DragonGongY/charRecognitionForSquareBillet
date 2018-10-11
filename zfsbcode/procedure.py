import numpy as np
import matplotlib.pyplot as plt
import os
import math

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
caffe_root = '/home/gsy/caffe-ssd/'  # this file is expected to be in {caffe_root}/examples
import os

os.chdir(caffe_root)
import sys

sys.path.insert(0, 'python')

import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load PASCAL VOC labels
labelmap_file = '/media/gsy/D02497132496FC22/Users/Administrator/Desktop/zfsb/recognition/labelmap_voc.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

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


model_def = '/media/gsy/D02497132496FC22/Users/Administrator/Desktop/zfsb/recognition/deploy.prototxt'
model_weights = '/media/gsy/D02497132496FC22/Users/Administrator/Desktop/zfsb/recognition/VGG_VOC0712_SSD_300x300_iter_81000.caffemodel'

net = caffe.Net(model_def,  # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)  # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([50.845, 50.845, 50.845]))  # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB


image_resize = 300
net.blobs['data'].reshape(1, 3, image_resize, image_resize)

image = caffe.io.load_image('/home/gsy/155.jpg')
plt.imshow(image)
transformed_image = transformer.preprocess('data', image)
net.blobs['data'].data[...] = transformed_image

# Forward pass.
detections = net.forward()['detection_out']

# Parse the outputs.
det_label = detections[0, 0, :, 1]
det_conf = detections[0, 0, :, 2]
det_xmin = detections[0, 0, :, 3]
det_ymin = detections[0, 0, :, 4]
det_xmax = detections[0, 0, :, 5]
det_ymax = detections[0, 0, :, 6]

# Get detections with confidence higher than 0.6.
top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.5]

top_conf = det_conf[top_indices]
top_label_indices = det_label[top_indices].tolist()
top_labels = get_labelname(labelmap, top_label_indices)
top_xmin = det_xmin[top_indices]
top_ymin = det_ymin[top_indices]
top_xmax = det_xmax[top_indices]
top_ymax = det_ymax[top_indices]
colors = plt.cm.hsv(np.linspace(0, 1, 37)).tolist()

plt.imshow(image)
currentAxis = plt.gca()


center = []
labels = []
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

    center7 = [ymin + (ymax - ymin) / 2., xmin + (xmax - xmin) / 2.]
    center.append(center7)
    labels.append(label_name)

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



plt.savefig('/home/gsy/recognition.jpg')
plt.close()
