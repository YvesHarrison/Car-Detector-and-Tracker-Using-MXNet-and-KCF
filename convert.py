#! /usr/bin/env python
#-*- coding: utf-8 -*-

from __future__ import print_function
import os
import cv2

def main():
    image_dir = 'data/hcar/JPEGImages'
    label_dir = 'data/hcar/Annotations'

    train = 'data/hcar/ImageSets/train.txt'

    train_ns = { s.strip() for s in open(train).readlines() }

    ns = []
    for n in os.listdir(image_dir):
        p = n.split('.jpg')[0]
        if p.endswith('gray'): continue
        if p not in train_ns: continue
        print(p)
        print(os.path.join(label_dir, p+'.txt'))
        if os.path.exists(os.path.join(label_dir, p+'.txt')):
            print(n)
            image = cv2.imread(os.path.join(image_dir, n))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            cv2.imwrite(os.path.join(image_dir, p+'.gray.jpg'), gray)
            with open(os.path.join(label_dir, p+'.gray.txt'), 'w') as fd:
                fd.write(''.join(open(os.path.join(label_dir, p+'.txt')).readlines()))
            ns.append(p+'.gray')
    with open('gray_train.txt', 'w') as fd:
        fd.write('\n'.join(ns))
    

if __name__ == '__main__':
    main()
