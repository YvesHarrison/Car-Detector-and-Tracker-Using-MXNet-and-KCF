from __future__ import print_function
import argparse
import os
import cv2
import time
#import mxnet as mx
import numpy as np
import random
#from rcnn.config import config
#from rcnn.symbol import get_vggm_test, get_vggm_rpn_test
#from rcnn.symbol import get_vgg_test, get_vgg_rpn_test
#from rcnn.symbol import get_resnet_test
#from rcnn.io.image import resize, transform
#from rcnn.core.tester import Predictor, im_detect, im_proposal
#from rcnn.utils.load_model import load_param
#from rcnn.processing.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper


CLASSES = ('__background__',
        'car', 'coach','truck','person','tanker')

def parse_args():
    parser = argparse.ArgumentParser(description='Demonstrate a Faster R-CNN network')
    parser.add_argument('--image', help='custom image', default='', type=str)
    parser.add_argument('--prefix', help='saved model prefix', type=str)
    parser.add_argument('--epoch', help='epoch of pretrained model', type=int)
    parser.add_argument('--gpu', help='GPU device to use', default=0, type=int)
    parser.add_argument('--vis', help='display result', action='store_true')
    parser.add_argument('--network', help='display result', default='vgg', type=str)
    parser.add_argument('--in_dir', type=str, default='.')
    parser.add_argument('--test', type=str, default='.')
    parser.add_argument('--out_dir', type=str, default='.')
    parser.add_argument('--label_dir', type=str, default='.')
    args = parser.parse_args()
    return args

def tracking():
    n = 2
    trackers = [ cv2.Tracker_create('KCF') for _ in range(n) ]
    initBoxes = [ (385.5, 212.8, 137, 118), (553.7, 175.9, 63, 67)  ]
    tracker1 = cv2.Tracker_create('KCF')
    tracker2 = cv2.Tracker_create('KCF')
    initProbs = [ 0.99, 0.99 ]
    
    in_dir = 'data/hcar/tracking'
    first = True
    names = sorted([ os.path.join(in_dir, name) for name in os.listdir(in_dir) if name.endswith('.jpg')])

    #print('\n'.join(images[:100]))
    for name in names:
        im = cv2.imread(name)
        if first:
            first = False
            for tracker, initBox, initProb in zip(trackers, initBoxes, initProbs):
                ok = tracker.init(im, initBox)
            continue

        stop = True
        for i, (tracker, initBox, initProb) in enumerate(zip(trackers, initBoxes, initProbs)):
            ok, bbox = tracker.update(im)
            if not ok:
                print('{} Not found'.format(i))
                continue
              
            stop = False
            bbox = map(int, bbox)
            color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color=color, thickness=2)
            cv2.imshow('Tracking', im)
        
        if stop: break
        k = cv2.waitKey() & 0xff
        if k == 27: break

    print('last name = {}'.format(name))

if __name__ == '__main__':
    tracking()
