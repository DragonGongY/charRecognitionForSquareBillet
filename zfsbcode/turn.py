# coding:utf-8
import numpy as np
import sys
import shutil
import os
caffe_root = '/home/gsy/caffe-ssd/'
sys.path.insert(0, caffe_root+'python')
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

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

picfiles = os.listdir('/home/gsy/pic/')
for p in picfiles:

    iimg = caffe.io.load_image('/home/gsy/pic/'+p)

    transformed_image_turn = transformer_turn.preprocess('data', iimg)

    net_turn.blobs['data'].data[...] = transformed_image_turn

    output = net_turn.forward()

    output_prob = output['softmax'][0]

    # D(0) L(1) R(2) T(3)
    print 'predicted class is:', output_prob.argmax()
    print '{}th picture'.format(p)

    if int(output_prob.argmax()) == 0:
        pass
        # shutil.move('/home/gsy/pic/'+p, '/home/gsy/pics/')
    elif int(output_prob.argmax()) == 1:
        pass
        # shutil.move('/home/gsy/pic/'+p, '/home/gsy/pics/')
    elif int(output_prob.argmax()) == 2:
        pass
        # shutil.move('/home/gsy/pic/'+p, '/home/gsy/pics/')
    else:
        shutil.move('/home/gsy/pic/'+p, '/home/gsy/pics/')


print 'all done'