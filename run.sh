#!/usr/bin/env bash

python hcar.py --network vggm --prefix $1 --epoch $2 --in_dir data/hcar/rcnn-format/c4 --out_dir output/ --with_label 1  --label_dir data/hcar/rcnn-format/c4 --gpu 0 --test data/hcar/rcnn-format/c4/train.txt

