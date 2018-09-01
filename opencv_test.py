import cv2

cnt = 0
while True:
    cnt = (cnt + 1) % 100
    im = cv2.imread('data/hcar/tracking/seq_1_{:0>4d}.jpg'.format(cnt + 1))
    cv2.imshow('test', im)
    k = cv2.waitKey() & 0xff
    if k == 27 : break
