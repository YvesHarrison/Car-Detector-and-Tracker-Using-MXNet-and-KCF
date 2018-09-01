"""
Pascal VOC database
This class loads ground truth notations from standard Pascal VOC XML data formats
and transform them into IMDB format. Selective search is used for proposals, see roidb
function. Results are written as the Pascal VOC format. Evaluation is based on mAP
criterion.
"""

from __future__ import print_function
import cPickle
import cv2
import os
import numpy as np

from imdb import IMDB
#from hcar_eval import hcar_eval
from ds_utils import unique_boxes, filter_small_boxes


class hcar(IMDB):
    def __init__(self, image_set, root_path, devkit_path):
        """
        fill basic information to initialize imdb
        :param image_set: train or test
        :param root_path: intermediate result, e.g. 'cache'
        :param devkit_path: data and results
        :return: imdb object
        """
        super(hcar, self).__init__('HCar_{}'.format(image_set), image_set, root_path, devkit_path)  # set self.name
        self.root_path = root_path
        self.devkit_path = devkit_path
        self.data_path = devkit_path

        self.classes = ['__background__',  # always index 0
                        'car', 'coach','truck','person','tanker']
        self.num_classes = len(self.classes)
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        print('num_images', self.num_images)

        self.config = {'comp_id': 'comp4',
                       'use_diff': False,
                       'min_size': 2}

    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        image_set_index_file = os.path.join(self.data_path, 'ImageSets', self.image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            image_set_index = [x.strip() for x in f.readlines() if len(x.strip())]
        return image_set_index

    def image_path_from_index(self, index):
        """
        given image index, find out full path
        :param index: index of a specific image
        :return: full path of this image
        """
        image_file = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def gt_roidb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        print('cache_file = {}'.format(cache_file))
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self.load_pascal_annotation(index) for index in self.image_set_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def load_pascal_annotation(self, index):
        """
        for a given index, load image and bounding boxes info from XML file
        :param index: index of a specific image
        :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        roi_rec = dict()
        roi_rec['image'] = self.image_path_from_index(index)
        size = cv2.imread(roi_rec['image']).shape
        roi_rec['height'] = size[0]
        roi_rec['width'] = size[1]

        filename = os.path.join(self.data_path, 'Annotations', index + '.txt')
        with open(filename) as fd:
            objs = [ obj for obj in fd.readlines() if len(obj) ]
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        class_to_index = dict(zip(self.classes, range(self.num_classes)))
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            cls, offsets = obj.split(':')
            offsets = offsets.split(',')

            x1 = float(offsets[0]) - 1
            y1 = float(offsets[1]) - 1
            x2 = float(offsets[2]) - 1
            y2 = float(offsets[3]) - 1

            cls = class_to_index[cls]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        roi_rec.update({'boxes': boxes,
                        'gt_classes': gt_classes,
                        'gt_overlaps': overlaps,
                        'max_classes': overlaps.argmax(axis=1),
                        'max_overlaps': overlaps.max(axis=1),
                        'flipped': False})
        return roi_rec

