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
import cv2
import sys
import multiprocessing
from time import time
import Queue

import kcftracker

CLASSES = ('__background__',
           'car', 'coach', 'truck', 'person', 'tanker')
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
            'shape inconsistent for ' + k + ' inferred ' + str(arg_shape_dict[k]) + ' provided ' + str(
                arg_params[k].shape)
    for k in symbol.list_auxiliary_states():
        assert k in aux_params, k + ' not initialized'
        assert aux_params[k].shape == aux_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(aux_shape_dict[k]) + ' provided ' + str(
                aux_params[k].shape)

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
    return data_batch, DATA_NAMES, im_scale


detect_num = {c: 0 for c in CLASSES}
tp, fp, fn = 0, 0, 0
gp, gr, gf1 = 0, 0, 0


def iou(rect1, rect2):
    iou = 0
    if rect1[0] < rect2[2] and rect1[2] > rect2[0] and rect1[1] < rect2[3] and rect1[3] > rect2[1]:
        i = (min(rect1[2], rect2[2]) - max(rect1[0], rect2[0])) * (min(rect1[3], rect2[3]), max(rect1[1], rect2[1]))
        o = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1]) + (rect2[2] - rect2[0]) * (rect2[3] - rect2[1]) - i
        iou = i / o
    return iou


def demo_net(predictor, image_name, image, with_label, vis, out_dir, label_dir):
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
    if (type(image_name)==str):
        assert os.path.exists(image_name), image_name + ' not found'
        im = cv2.imread(image_name)
    else:
        im = image
    # im = cv2.flip(im, 1)
    data_batch, data_names, im_scale = generate_batch(im)
    # for i in range(10):
    #    scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, im_scale)

    for i in range(1):
        scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, im_scale)
  
    
    xn = []
    yn = []
    wn = []
    hn = []
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
    # print(boxes_this_image)
    
    # print results
    rst = {};
    lfn, lfp, ltp = 0, 0, 0
    #print('class ---- [[x1, x2, y1, y2, confidence]]')
    for ind, boxes in enumerate(boxes_this_image):
        if len(boxes) > 0:
            #print('---------', CLASSES[ind], '---------')
            #print(boxes)
            for i in range(0, len(boxes)):
                xn.append(int(boxes[i][0] + 0))
                yn.append(int(boxes[i][1] + 0))
                wn.append(int(boxes[i][2] - boxes[i][0]))
                hn.append(int(boxes[i][3] - boxes[i][1]))

            #rst[CLASSES[ind]] = [box for box in boxes]
            # detect_num[CLASSES[ind]] += len(boxes)
            #detect_num[CLASSES[ind]] += 1  # len(boxes)

    """if image == '' and with_label:
        label_file = os.path.join(label_dir, os.path.split(image_name.replace('.jpg', '.txt'))[1])
        with open(label_file) as fd:
            for line in fd:
                cls, poss = line.split(':')
                x1, y1, x2, y2 = [float(item) for item in poss.split(',')]

                if cls not in rst:
                    lfn += 1
                    continue

                iou_thd = 0.5
                now_iou = 0
                now_idx = 0
                for ind, box in enumerate(rst[cls]):
                    # print('box = ', box, type(box))
                    # print('box = {}, true = {}'.format(box, (x1, y1, x2, y2)))
                    if (box[0] >= x2) or (box[2] <= x1) or (box[1] >= y2) or (box[3] <= y1):
                        continue
                    else:
                        # print('###############################################')
                        i = (min(x2, box[2]) - max(x1, box[0])) * (min(y2, box[3]) - max(y1, box[1]))
                        assert (i > 0)
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
        gf1 += f1"""

    """if vis:
        vis_all_detection(data_dict['data'].asnumpy(), boxes_this_image, CLASSES, im_scale)
    else:
        # print(os.path.join(args.out_dir, os.path.split(image_name.replace('.jpg', '_result.jpg'))[1]))
        # result_file = os.path.join(out_dir, os.path.split(image_name.replace('.jpg', '_result.jpg'))[1])
        result_file = os.path.join(out_dir, os.path.split('_result.jpg')[1])
        print('results saved to %s' % result_file)
        im = draw_all_detection(data_dict['data'].asnumpy(), boxes_this_image, CLASSES, im_scale)
        cv2.imwrite(result_file, im)"""
    # print(type(xn)) 
    return xn, yn, wn, hn


"""def parse_args():
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
    return args"""


def compare(x, y):
    DIR='video_output/'
    stat_x = os.stat(DIR + "/" + x)

    stat_y = os.stat(DIR + "/" + y)

    if stat_x.st_ctime < stat_y.st_ctime:
        return -1
    elif stat_x.st_ctime > stat_y.st_ctime:
        return 1
    else:
        return 0


def main(predictor,ctx,image, prefix, epoch, gpu, vis, network, in_dir, test, out_dir, label_dir, with_label):
  
    global tp, fp, fn, detect_num
    # args = parse_args()
    
    # if args.network == 'vgg':
    #    symbol = get_vgg_test(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)
    # elif args.network == 'vggm':
    #symbol = eval('get_' + network + '_test')(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)
    #predictor = get_net(symbol, prefix, epoch, ctx)
    if in_dir=='':
    
        x = []
        y = []
        w = []
        h = []
        """result = []
        pool = multiprocessing.Pool(processes=len(image))
        for i in range(0, len(image)):
            result.append(pool.apply_async(demo_net, (predictor, image[i], image[i], with_label, vis, out_dir,
                                     label_dir)))
        print(result)
       
        for i in range(0, len(result)):
            res = result[i].get()
            x.append(res[0])
            y.append(res[1])
            w.append(res[2])
            h.append(res[3])  
        pool.close()
        pool.join()"""
        a=image.qsize()
        for i in range (0,a):
            img=image.get()
            x1, y1, w1, h1 = demo_net(predictor,img, img, with_label, vis, out_dir,
                                     label_dir)
            x.append(x1)
            y.append(y1)
            w.append(w1)
            h.append(h1)
        
                    
    else:
        if image != '':
            return demo_net(predictor, image, image, with_label, vis, out_dir, label_dir)
        else:
            # t0 = time.clock()
            # print(os.listdir(in_dir), in_dir)
            num = 0
            # with open(test) as fd:
            # test_imgs = set([n.strip() + '.jpg' for n in fd.readlines()])
            iterms = os.listdir(in_dir)
            iterms.sort(compare)
            # for iterm in iterms:
            # print(iterm)

            imgs = [img for img in iterms]
            #print(imgs)
            # for image in [ img for img in os.listdir(in_dir) if not os.path.isdir(img) and not img.count('_result.')]:
            x = []
            y = []
            w = []
            h = []
            for image in imgs:
                print(os.path.join(in_dir, image))
                x1, y1, w1, h1 = demo_net(predictor, os.path.join(in_dir, image), image, with_label, vis, out_dir,
                                          label_dir)
                x.append(x1)
                y.append(y1)
                w.append(w1)
                h.append(h1)
                """num += 1
            if with_label:
                p = 100.0 * tp / (tp + fp)
                r = 100.0 * tp / (tp + fn)
                f1 = 2.0 * p * r / (p + r)
                print(
                    'type 1#: avg precision = {}%, avg recall = {}%, avg f1 score = {}% (with tp = {}, fp = {}, fn = {})'.format(
                        p, r, f1, tp, fp, fn))
                print(
                    'type 2#: avg precision = {}%, avg recall = {}%, avg f1 score = {}%'.format(gp / num, gr / num, gf1 / num))
            print(time.clock() - t0, '(with {} samples, detects {} objects)'.format(num, detect_num))"""


    return x,y,w,h

selectingObject = False
# initTracking = False
initTracking = True
onTracking = False

ix, iy, cx, cy = 0, 0, -1, -1
w, h = 0, 0

inteval = 1
duration = 0.04


# mouse callback function
def draw_boundingbox(event, x, y, flags, param):
    global selectingObject, initTracking, onTracking, ix, iy, cx, cy, w, h

    if event == cv2.EVENT_LBUTTONDOWN:
        selectingObject = True
        onTracking = False
        ix, iy = x, y
        cx, cy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        cx, cy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        selectingObject = False
        if (abs(x - ix) > 10 and abs(y - iy) > 10):
            w, h = abs(x - ix), abs(y - iy)
            ix, iy = min(x, ix), min(y, iy)
            initTracking = True
        else:
            onTracking = False

    elif event == cv2.EVENT_RBUTTONDOWN:
        onTracking = False
        if (w > 0):
            ix, iy = x - w / 2, y - h / 2
            initTracking = True


def Tracking(trackers, frame):
    boundingbox = trackers.update(frame)
    boundingbox = map(int, boundingbox)
    return boundingbox,trackers


def Renew(x,y,w,h,box):
    #print(x,y,w,h)
    #print(box)
    if len(x)!=len(box):
        #print ("renew")
        return True
    for i in range(len(x)):
        if(abs(x[i]-box[i][0])/float(box[i][0])>0.05 or abs(y[i]-box[i][1])/float(box[i][1])>0.05 ):
            #print("renew")
            return True
    #print("remain")
    return False
    
def video(name,ctx):
    inteval = 1
    duration = 0.01
    global selectingObject, initTracking, onTracking, ix, iy, cx, cy, w, h
    show_delay=24
    network_inteval=8
    start = True
    store = True
    cap = cv2.VideoCapture(name)
    t3=time()
    video = Queue.Queue(maxsize = show_delay)
    k = 0
    cn = 0
    ct = 0
    gpu = 0
    renew=True
    capture=Queue.Queue(maxsize=show_delay/network_inteval)
    cv2.namedWindow('tracking')
    cv2.setMouseCallback('tracking', draw_boundingbox)
    network='vggm'
    prefix='model/e2e'
    epoch=20
    symbol = eval('get_' + network + '_test')(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)
    predictor = get_net(symbol, prefix, epoch, ctx)
    #print (predictor)
    while (cap.isOpened()or ct!=cn):
        ret, frame = cap.read()
        if store:
            video.put(frame)
            if(k%network_inteval==0):
                capture.put(frame)
            k = k + 1
            cn = cn + 1
            if k==show_delay:
                store=False
            
        else:
            if start:
                timer = 0
                cnt=0
                start = False
            
            if not ret:
                cap.release()
                #break very slow if cap not released

            if (selectingObject):
                cv2.rectangle(frame, (ix, iy), (cx, cy), (0, 255, 255), 1)
            elif (initTracking):
                n=[]
                x=[]
                y=[]
                w=[]
                h=[]
                t5=time()
                x1, y1, w1, h1 = main(predictor,ctx,capture, prefix='model/e2e', epoch=20, gpu=0, vis=False, network='vggm',
                                  in_dir='', test='test.txt', out_dir='output/',
                                  label_dir='data/hcar/Annotations/', with_label=1)
                t6=time()
                print (t6-t5)
                for i in range (0,len(x1)):
                    x.append(x1[i])
                    y.append(y1[i])
                    w.append(w1[i])
                    h.append(h1[i])
                    n.append(len(x1[i]))
                initTracking = False
                onTracking = True
                
            elif (onTracking):
                ct += 1
                timer += 1
                t0 = time()
                show=video.get()
                #if (t%network_inteval==0 and Renew(x[t/network_inteval],y[t/network_inteval],w[t/network_inteval],h[t/network_inteval],box)):
                if (timer==1 and renew):
                    j=cnt
                    trackers={}
                    length=n[j]
                    box=[]

                    if(length!=0):
                        pool = multiprocessing.Pool(processes=length)
                    for i in range(0, length):
                        ix = x[j][i]
                        iy = y[j][i]
                        iw = w[j][i]
                        ih = h[j][i]
                        cv2.rectangle(show, (ix, iy), (ix + iw, iy + ih), (0, 255, 255), 2)
                        tracker = kcftracker.KCFTracker(True, True, True)  # hog, fixed_window, multiscale
                        tracker.init([ix, iy, iw, ih], show)
                        trackers[i] = tracker
                        box.append(0)
                #elif(t%network_inteval==0):
                    #pool = multiprocessing.Pool(processes=length)
                result = []

                for i in range(0, length):
                    result.append(pool.apply_async(Tracking, (trackers[i], show)))

                for i in range(0, length):
                    res = result[i].get()
                    trackers[i]=res[1]
                    #print(res[0][0],res[0][1],res[0][2],res[0][3])
                    box[i]=res[0]
                    cv2.rectangle(show, (res[0][0], res[0][1]),
                              (res[0][0] + res[0][2], res[0][1] + res[0][3]), (0, 255, 255), 1)
                t1 = time()
               
                duration = 0.8 * duration + 0.2 * (t1 - t0)
            # duration = t1-t0
                cv2.putText(show, 'FPS: ' + str(1 / duration)[:4].strip('.'), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)
                cv2.imshow('tracking', show)
                if (timer == network_inteval):
                    timer = 0
                    cnt+=1
                    if cnt<show_delay/network_inteval:
                        renew=Renew(x[cnt],y[cnt],w[cnt],h[cnt],box)
                    else:
                        renew=True
                    if renew:
                        pool.close()
                        pool.join()
                    
                if (cnt==show_delay/network_inteval):
                    initTracking = True
                    onTracking = False
                    cnt=0

                
                if(ret):
                    video.put(frame)
                    cn = cn + 1
                    if(timer==1):
                        capture.put(frame)
            cv2.waitKey(inteval)
            """c = cv2.waitKey(inteval) & 0xFF
        # break
            if c == 27 or c == ord('q'):
                break"""
    #print(k,cn,ct)
    
    t4=time()
    print (t4-t3)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    t=time()
    print ("java")
    if (len(sys.argv) == 1):
        cap = cv2.VideoCapture(0)
    else:
        if (sys.argv[1].isdigit()):  # True if sys.argv[1] is str of a nonnegative integer
            cap = cv2.VideoCapture(int(sys.argv[1]))
           
        else:
            name=[]
            for i in range (1,len(sys.argv)):
              name.append(sys.argv[i])
            print (name)
            inteval = 30
    gpu = 0
    ctx = mx.gpu(gpu)
    record=[]
    for i in range(0, len(sys.argv)-1):
        process = multiprocessing.Process(target=video, args=(str(name[i]),ctx))
        process.start()
        record.append(process)
                # print(boundingbox)
    for process in record:
        process.join() 
    print(time()-t)
    # if you use hog feature, there will be a short pause after you draw a first boundingbox, that is due to the use of Numba.
    



