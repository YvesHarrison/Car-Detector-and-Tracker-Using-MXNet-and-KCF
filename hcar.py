from __future__ import print_function
import argparse
import os
import cv2
import time
import mxnet as mx
import numpy as np
from rcnn.config import config
from rcnn.symbol import get_vggm_test, get_vggm_rpn_test
from rcnn.symbol import get_vgg_test, get_vgg_rpn_test
from rcnn.symbol import get_resnet_test
from rcnn.io.image import resize, transform
from rcnn.core.tester import Predictor, im_detect, im_proposal, vis_all_detection, draw_all_detection
from rcnn.utils.load_model import load_param
from rcnn.processing.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper


CLASSES = ('__background__',
        'car', 'coach','truck','person','tanker')
config.TEST.HAS_RPN = True
SHORT_SIDE = config.SCALES[0][0]
LONG_SIDE = config.SCALES[0][1]
PIXEL_MEANS = config.PIXEL_MEANS
DATA_NAMES = ['data', 'im_info']
LABEL_NAMES = None
DATA_SHAPES = [('data', (1, 3, LONG_SIDE, SHORT_SIDE)), ('im_info', (1, 3))]
LABEL_SHAPES = None
# visualization
CONF_THRESH = 0.5
NMS_THRESH = 0.3
#NMS_THRESH = 0.3
nms = py_nms_wrapper(NMS_THRESH)


def get_net(symbol, prefix, epoch, ctx):
    arg_params, aux_params = load_param(prefix, epoch, convert=True, ctx=ctx, process=True)

    # infer shape
    data_shape_dict = dict(DATA_SHAPES)
    arg_names, aux_names = symbol.list_arguments(), symbol.list_auxiliary_states()
    arg_shape, _, aux_shape = symbol.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(arg_names, arg_shape))
    aux_shape_dict = dict(zip(aux_names, aux_shape))

    # check shapes
    for k in symbol.list_arguments():
        if k in data_shape_dict or 'label' in k:
            continue
        assert k in arg_params, k + ' not initialized'
        assert arg_params[k].shape == arg_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(arg_shape_dict[k]) + ' provided ' + str(arg_params[k].shape)
    for k in symbol.list_auxiliary_states():
        assert k in aux_params, k + ' not initialized'
        assert aux_params[k].shape == aux_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(aux_shape_dict[k]) + ' provided ' + str(aux_params[k].shape)

    predictor = Predictor(symbol, DATA_NAMES, LABEL_NAMES, context=ctx,
                          provide_data=DATA_SHAPES, provide_label=LABEL_SHAPES,
                          arg_params=arg_params, aux_params=aux_params)
    return predictor


def generate_batch(im):
    """
    preprocess image, return batch
    :param im: cv2.imread returns [height, width, channel] in BGR
    :return:
    data_batch: MXNet input batch
    data_names: names in data_batch
    im_scale: float number
    """
    im_array, im_scale = resize(im, SHORT_SIDE, LONG_SIDE)
    im_array = transform(im_array, PIXEL_MEANS)
    im_info = np.array([[im_array.shape[2], im_array.shape[3], im_scale]], dtype=np.float32)
    data = [mx.nd.array(im_array), mx.nd.array(im_info)]
    data_shapes = [('data', im_array.shape), ('im_info', im_info.shape)]
    data_batch = mx.io.DataBatch(data=data, label=None, provide_data=data_shapes, provide_label=None)
    #print (type(data_shapes))
    return data_batch, DATA_NAMES, im_scale

detect_num = { c : 0 for c in CLASSES }
tp, fp, fn = 0, 0, 0
gp, gr, gf1 = 0, 0, 0

def iou(rect1, rect2):
    iou = 0
    if rect1[0] < rect2[2] and rect1[2] > rect2[0] and rect1[1] < rect2[3] and rect1[3] > rect2[1]:
        i = (min(rect1[2], rect2[2]) - max(rect1[0], rect2[0])) * (min(rect1[3], rect2[3]), max(rect1[1], rect2[1]))
        o = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1]) + (rect2[2] - rect2[0]) * (rect2[3] - rect2[1]) - i
        iou = i / o
    return iou

def demo_net(predictor, image_name, args):
    """
    generate data_batch -> im_detect -> post process
    :param predictor: Predictor
    :param image_name: image name
    :param vis: will save as a new image if not visualized
    :return: None
    """
    global detect_num
    global tp, fp, fn
    global gp, gr, gf1
    assert os.path.exists(image_name), image_name + ' not found'
    im = cv2.imread(image_name)
    #im = cv2.flip(im, 0)
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #type(im.shape)
    #im.reshape((im.shape[0],im.shape[1],1))
    data_batch, data_names, im_scale = generate_batch(im)
    #for i in range(10):
    #    scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, im_scale)
    t0 = time.clock()
    for i in range(1):
        scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, im_scale)

    print(time.clock() - t0)

    all_boxes = [[] for _ in CLASSES]
    for cls in CLASSES:
        cls_ind = CLASSES.index(cls)
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind, np.newaxis]
        keep = np.where(cls_scores >= CONF_THRESH)[0]
        dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)[keep, :]
        keep = nms(dets)
        all_boxes[cls_ind] = dets[keep, :]

    boxes_this_image = [[]] + [all_boxes[j] for j in range(1, len(CLASSES))]
    #print(boxes_this_image)

    # print results
    rst = {}; lfn, lfp, ltp = 0, 0, 0
    print('class ---- [[x1, x2, y1, y2, confidence]]')
    for ind, boxes in enumerate(boxes_this_image):
        if len(boxes) > 0:
            print('---------', CLASSES[ind], '---------')
            print(boxes)
            rst[CLASSES[ind]] = [ box for box in boxes]
            #detect_num[CLASSES[ind]] += len(boxes)
            detect_num[CLASSES[ind]] += 1 #len(boxes)

    if args.image == '' and args.with_label:
        label_file = os.path.join(args.label_dir, os.path.split(image_name.replace('.jpg', '.txt'))[1])
        with open(label_file) as fd:
            for line in fd:
                cls, poss = line.split(':')
                x1, y1, x2, y2 = [ float(item) for item in poss.split(',') ]

                if cls not in rst:
                    lfn += 1
                    continue

                iou_thd = 0.5
                now_iou = 0
                now_idx = 0
                for ind, box in enumerate(rst[cls]):
                    #print('box = ', box, type(box))
                    #print('box = {}, true = {}'.format(box, (x1, y1, x2, y2)))
                    if (box[0] >= x2) or (box[2] <= x1) or (box[1] >= y2) or (box[3] <= y1):
                            continue
                    else:
                        #print('###############################################')
                        i = (min(x2, box[2]) - max(x1, box[0])) * (min(y2, box[3]) - max(y1, box[1]))
                        assert(i > 0)
                        u = (x2 - x1) * (y2 - y1) + (box[0] - box[2]) * (box[1] - box[3]) - i
                        if i / u > now_iou:
                            now_iou = i / u
                            now_idx = ind
                if now_iou > iou_thd:
                    ltp += 1
                    rst[cls].pop(now_idx)
                    if len(rst[cls]) == 0: rst.pop(cls)
                else:
                    lfn += 1
        for vs in rst.values():
            lfp += len(vs)

        p, r, f1 = 0, 0, 0
        if ltp != 0:
            p = 100.0 * ltp / (ltp + lfp)
            r = 100.0 * ltp / (ltp + lfn)
            f1 = 2 * p * r / (p + r)
        print('precision = {}%, recall = {}%, f1 score = {}%'.format(p, r, f1))

        tp += ltp
        fp += lfp
        fn += lfn
        gp += p
        gr += r
        gf1 += f1

    if args.vis:
        vis_all_detection(data_dict['data'].asnumpy(), boxes_this_image, CLASSES, im_scale)
    else:
        #print(os.path.join(args.out_dir, os.path.split(image_name.replace('.jpg', '_result.jpg'))[1]))
        result_file = os.path.join(args.out_dir, os.path.split(image_name.replace('.jpg', '_result.jpg'))[1])
        print('results saved to %s' % result_file)
        im = draw_all_detection(data_dict['data'].asnumpy(), boxes_this_image, CLASSES, im_scale)
        cv2.imwrite(result_file, im)


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
    parser.add_argument('--with_label', type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    global tp, fp, fn, detect_num
    args = parse_args()
    
    ctx = mx.gpu(args.gpu)
    #if args.network == 'vgg':
    #    symbol = get_vgg_test(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)
    #elif args.network == 'vggm':
    symbol = eval('get_'+ args.network + '_test')(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)
    predictor = get_net(symbol, args.prefix, args.epoch, ctx)

    if args.image != '':
        demo_net(predictor, args.image, args)
        return


    t0 = time.clock()
    #print(os.listdir(args.in_dir), args.in_dir)
    num = 0
    with open(args.test) as fd:
        test_imgs = set([ n.strip() + '.jpg' for n in fd.readlines() ])
    imgs = [ img for img in os.listdir(args.in_dir) if not os.path.isdir(img) and not img.count('_result.') and img in test_imgs ]
    
    #for image in [ img for img in os.listdir(args.in_dir) if not os.path.isdir(img) and not img.count('_result.')]:
    for image in imgs:
        print(os.path.join(args.in_dir, image))
        demo_net(predictor, os.path.join(args.in_dir, image), args)
        num += 1
    print(time.clock() - t0, 'time for rcnn')
    if args.with_label:
        p = 100.0 * tp / (tp + fp)
        r = 100.0 * tp / (tp + fn)
        f1 = 2.0 * p * r / (p + r)
        print('type 1#: avg precision = {}%, avg recall = {}%, avg f1 score = {}% (with tp = {}, fp = {}, fn = {})'.format(p, r, f1, tp, fp, fn))
        print('type 2#: avg precision = {}%, avg recall = {}%, avg f1 score = {}%'.format(gp / num, gr / num, gf1 / num))
    print(time.clock() - t0, '(with {} samples, detects {} objects)'.format(num, detect_num))


if __name__ == '__main__':
    main()
