# -*- coding:utf-8 -*-
"""
create by gezhipeng
create on 18-12-25 上午9:05
func: 
"""
import os
import sys
sys.path.append('../')
import cv2
import numpy as np

def short_side_resize(img, short_size=256):
    """
    把图片等比缩放到短边为short_size
    """
    ratio = short_size / float(min(img.shape[:2]))
    new_h, new_w = int(img.shape[0] * ratio), int(img.shape[1] * ratio)

    res = cv2.resize(img, (new_w, new_h))
    return res

def equalize_hist(img):
    if len(img.shape) == 2:
        res = cv2.equalizeHist(img)
    elif len(img.shape) == 3:
        res = np.zeros_like(img)
        for ch in range(img.shape[2]):
            res[:, :, ch] = cv2.equalizeHist(img[:, :, ch])
    else:
        print("image error.")
        res = None
    return res


if __name__ == '__main__':
    image_dir = '/home/gezhipeng/datas/finger_occlusion/baidu/dark'
    for image in os.listdir(image_dir):
        path = os.path.join(image_dir, image)
        img = cv2.imread(path)
        res = equalize_hist(img)
        res = np.hstack((img, res))
        res = short_side_resize(res)
        cv2.imshow('', res)
        cv2.waitKey()
