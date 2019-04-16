# -*- coding:utf-8 -*-
"""
create by gezhipeng
create on 18-12-24 下午6:23
func: 
"""
import sys
sys.path.append('../../')
from Augmentation.poisson_image_editing import poisson_edit
from Visualization.image_label import imwrite_image_label
from Tools.utils import view_bar
import cv2
import os
import random
import math
import sys
import numpy as np
import pickle as pkl

import time
import platform
import matplotlib.pyplot as plt

# 手指遮挡位置
side_locations = ['t', 'r', 'b', 'l']                 # 边上
corner_locations = ['lt', 'rt', 'rb', 'lb']           # 角
corner3_locations = ['lt_3', 'rt_3', 'rb_3', 'lb_3']  # 大角
# 图片后缀
image_extensions = ['jpg', 'png', 'jpeg', 'bmp', 'tif', 'tiff']

def get_lc(img_mask):
    """
    Get the location of finger
    :param img_mask: none, 4 corners, 4 sides, 4 big corners, full
    :return:
    """
    record = [0, 0, 0, 0]
    if img_mask[0][0] == 255:
        record[0] = 1
    if img_mask[0][-1] == 255:
        record[1] = 1
    if img_mask[-1][0] == 255:
        record[2] = 1
    if img_mask[-1][-1] == 255:
        record[3] = 1
    s = sum(record)
    if s == 0:
        lc = "none"
    elif s == 1:
        # lc = id_lc[record.index(1)]
        if record[0] == 1:
            lc = 'lt'
        elif record[1] == 1:
            lc = 'rt'
        elif record[2] == 1:
            lc = 'lb'
        elif record[3] == 1:
            lc = 'rb'
    elif s == 2:
        if record[0] + record[1] == 2:
            lc = "t"
        elif record[1] + record[3] == 2:
            lc = "r"
        elif record[3] + record[2] == 2:
            lc = "b"
        else:
            lc = "l"
    elif s == 3:
        if record[0] == 0:
            lc = "rb_3"
        elif record[1] == 0:
            lc = "lb_3"
        elif record[2] == 0:
            lc = "rt_3"
        else:
            lc = "lt_3"
    else:
        lc = "full"
    return lc
# 根据目标图像、生成图像手指外接矩宽高占比获得合适的mask和src以及外接矩位置
def get_mask_and_src(img_mask, img_src, dst_img, lc0, w_mask, h_mask, w_ratio, h_ratio):
    """
    get img array of mask and src according to the dst img
    :param img_mask: mask
    :param img_src: src
    :param dst_img: dst
    :param lc0: location of finger in image
    :param w_mask: width of mask
    :param h_mask: height of mask
    :param w_ratio: the target width ratio of finger mask
    :param h_ratio: the target height ratio of finger mask
    :return: mask_img, src_img, rt, lb
    """
    h, w = dst_img.shape[0], dst_img.shape[1]      # 目标图片的尺度
    mask_img = np.zeros([h, w], dtype=np.uint8)    # 新建一个空的数组用于存放最后的mask 数组
    src_img = np.zeros([h, w, 3], dtype=np.uint8)  # 新建一个空的数组用于存放最后的src 数组

    w_mask2 = int(w_ratio * w)                     # 实际生成的手指外接矩的宽和高
    h_mask2 = int(h_ratio * h)
    # 生成手指的区域过大的警示
    if w_mask2 > 0.8 * w:
        raise Warning("生成的手指外接矩宽大于背景图片的宽的80%")
    if h_mask2 > 0.8 * h:
        raise Warning("生成的手指外接矩宽大于背景图片的宽的80%")

    new_mask_h = int(img_mask.shape[0] * h_mask2 / float(h_mask))  # 根据实际手指外接矩的宽高设计的mask宽高
    new_mask_w = int(img_mask.shape[1] * w_mask2 / float(w_mask))  #

    if lc0 == 'lt':
        # finger在左上角
        img_mask = cv2.resize(img_mask, (new_mask_w, new_mask_h))
        img_src = cv2.resize(img_src, (new_mask_w, new_mask_h))
        crop_h = min(img_src.shape[0], src_img.shape[0])
        crop_w = min(img_src.shape[1], src_img.shape[1])
        mask_img[: crop_h, :crop_w] = img_mask[: crop_h, : crop_w]
        src_img[: crop_h, :crop_w, :] = img_src[: crop_h, :crop_w, :]
        lt = (0, 0)
        rb = (crop_h, crop_h)
    elif lc0 == 'rt':
        img_mask = cv2.resize(img_mask, (new_mask_w, new_mask_h))
        img_src = cv2.resize(img_src, (new_mask_w, new_mask_h))
        crop_h = min(img_src.shape[0], src_img.shape[0])
        crop_w = min(img_src.shape[1], src_img.shape[1])
        mask_img[: crop_h, -crop_w:] = img_mask[: crop_h, -crop_w:]
        src_img[: crop_h, -crop_w:, :] = img_src[: crop_h, -crop_w:, :]
        lt = (w-crop_h, 0)
        rb = (w, crop_h)
    elif lc0 == 'lb':
        img_mask = cv2.resize(img_mask, (new_mask_w, new_mask_h))
        img_src = cv2.resize(img_src, (new_mask_w, new_mask_h))
        crop_h = min(img_src.shape[0], src_img.shape[0])
        crop_w = min(img_src.shape[1], src_img.shape[1])
        mask_img[-crop_h:, : crop_w] = img_mask[-crop_h:, : crop_w]
        src_img[-crop_h:, : crop_w, :] = img_src[-crop_h:, : crop_w, :]
        lt = (0, h-crop_h)
        rb = (crop_w, h)
    elif lc0 == 'rb':
        img_mask = cv2.resize(img_mask, (new_mask_w, new_mask_h))
        img_src = cv2.resize(img_src, (new_mask_w, new_mask_h))
        crop_h = min(img_src.shape[0], src_img.shape[0])
        crop_w = min(img_src.shape[1], src_img.shape[1])
        mask_img[-crop_h:, -crop_w:] = img_mask[-crop_h:, -crop_w:]
        src_img[-crop_h:, -crop_w:, :] = img_src[-crop_h:, -crop_w:, :]
        lt = (w-crop_w, h - crop_h)
        rb = (w, h)
    elif lc0 == 't' or lc0 == 'rt_3':
        img_mask = cv2.resize(img_mask, (w, new_mask_h))
        img_src = cv2.resize(img_src, (w, new_mask_h))
        crop_h = min(img_src.shape[0], src_img.shape[0])
        mask_img[: crop_h, :] = img_mask[: crop_h, :]
        src_img[: crop_h, :, :] = img_src[: crop_h, :, :]
        lt = (0, 0)
        rb = (w, crop_h)
    elif lc0 == 'r' or lc0 == 'rb_3':
        img_mask = cv2.resize(img_mask, (new_mask_w, h))
        img_src = cv2.resize(img_src, (new_mask_w, h))
        crop_w = min(img_src.shape[1], src_img.shape[1])
        mask_img[:, -crop_w:] = img_mask[:, -crop_w:]
        src_img[:, -crop_w:, :] = img_src[:, -crop_w:, :]
        lt = (w-crop_w, 0)
        rb = (w, h)
    elif lc0 == 'b' or lc0 == 'lb_3':
        img_mask = cv2.resize(img_mask, (w, new_mask_h))
        img_src = cv2.resize(img_src, (w, new_mask_h))
        crop_h = min(img_src.shape[0], src_img.shape[0])
        mask_img[-crop_h:, :] = img_mask[-crop_h:, :]
        src_img[-crop_h:, :, :] = img_src[-crop_h:, :, :]
        lt = (0, h - crop_h)
        rb = (w, h)
    elif lc0 == 'l' or lc0 == 'lt_3':
        img_mask = cv2.resize(img_mask, (new_mask_w, h))
        img_src = cv2.resize(img_src, (new_mask_w, h))
        crop_w = min(img_src.shape[1], src_img.shape[1])
        mask_img[:, : crop_w] = img_mask[:, : crop_w]
        src_img[:, : crop_w, :] = img_src[:, : crop_w, :]
        lt = (0, 0)
        rb = (crop_w, h)
    else:
        # 手指不经过角上或者四个角都有手指
        mask_img = src_img = lt = rb = None
        # print("4 corner all occluded or no occlusion in the corner")
    if mask_img is not None:
        ret, mask_img = cv2.threshold(mask_img, 127, 255, 0)
    return mask_img, src_img, lt, rb
# 获得手指在边上或者角上的mask，并返回一些其他的值
def get_mask_finger_side_corner(mask_path, new_size=256):
    """
    获得手指在角上或者边上的mask
    :param mask_path:
    :param new_size: mask将要缩放到的尺度， 如果是None，不需要进行缩放
    :return:
    """
    img_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Gray
    if new_size is not None:
        mask = cv2.resize(img_mask, (new_size, new_size))
    ret, mask = cv2.threshold(mask, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x_mask, y_mask, w_mask, h_mask = cv2.boundingRect(cnt)
    lc = get_lc(mask)
    if lc in ['lt', 'rt', 'lb', 'rb']:
        key = "corner"
    elif lc in ['lt_3', 'rt_3', 'lb_3', 'rb_3', 't', 'r', 'l', 'b']:
        key = 'side'
    else:
        key = "invalid"
    area = max(int(np.sum(mask/255) * 100 / float(mask.shape[0] * mask.shape[1])), 1)
    return mask, key, lc, w_mask, h_mask, area
# 把手指遮挡在边上或者角上mask与对应的src保存在pkl数据中
def save_data_to_pkl_finger_side_corner(src_path, mask_path, pkl_data, new_size=256):
    """
    save the template data to a dict, which will be stored in a pkl file
    :param src_path:
    :param mask_path:
    :param pkl_data:
    :param new_size:
    :return:
    """
    try:
        img_mask, key, lc, w_mask, h_mask, area = get_mask_finger_side_corner(mask_path, new_size=new_size)
        assert img_mask is not None, "mask is none"
        img_src = cv2.imread(src_path)
        img_src = cv2.resize(img_src, (new_size, new_size))
        assert img_src.shape[2] == 3, "{} is not 3 channel".format(src_path)
    except Exception:
        print("{} wrong. do not save".format(mask_path))
        return
    if key in pkl_data.keys():
        pkl_data[key]['mask'].append(img_mask)
        pkl_data[key]['src'].append(img_src)
        pkl_data[key]['location'].append(lc)
        pkl_data[key]['w_mask'].append(w_mask)
        pkl_data[key]['h_mask'].append(h_mask)
        pkl_data[key]['area'].append(area)
    else:
        pkl_data[key] = {}
        pkl_data[key]['mask'] = [img_mask]
        pkl_data[key]['src'] = [img_src]
        pkl_data[key]['location'] = [lc]
        pkl_data[key]['w_mask'] = [w_mask]
        pkl_data[key]['h_mask'] = [h_mask]
        pkl_data[key]['area'] = [area]
# 通过指定源文件夹和mask文件夹列表获得手指在角上或者边上的mask与src，并存储为pkl文件
def get_pkl_file_finger_side_corner(src_dirs, mask_dirs, pkl_path, new_size=256):
    """
    常规的手指遮挡模板，指定src文件夹列表和mask文件夹列表即可生成pkl文件
    :param src_dirs:
    :param mask_dirs:
    :param pkl_path:
    :param new_size:
    :return:
    """
    pkl_data = {}
    for src_dir, mask_dir in zip(src_dirs, mask_dirs):
        files = os.listdir(src_dir)
        files = [file_ for file_ in files if file_.split('.')[-1].lower() in image_extensions]
        for i, file_ in enumerate(files):
            view_bar("converting the pickle data from {}".format(src_dir), i+1, len(files))
            src_path = os.path.join(src_dir, file_)
            mask_path = os.path.join(mask_dir, file_.replace(file_.split('.')[-1], 'png'))
            save_data_to_pkl_finger_side_corner(src_path, mask_path, pkl_data, new_size=new_size)
    # 针对不同的python版本，pickle文件的存储形式不同
    if platform.python_version().startswith("3"):
        pkl.dump(pkl_data, open(pkl_path, 'wb'))
    else:
        pkl.dump(pkl_data, file(pkl_path, 'wb'))
# 获得一些困难手指位置的mask与src，并且存储为pkl文件
def get_hard_pkl_file(mask_dir, src_dir, pkl_path, new_size=256):
    images = os.listdir(src_dir)
    images = [image for image in images if image.split('.')[-1].lower() in image_extensions]
    locations = ['t', 'r', 'b', 'l']
    pkl_data = {}
    key = 'hard'
    pkl_data[key] = {}
    for i, image in enumerate(images):
        view_bar("converting the bad case images:", i+1, len(images))
        try:
            src_path = os.path.join(src_dir, image)
            mask_image = image.replace(image.split('.')[-1], 'png')
            mask_path = os.path.join(mask_dir, mask_image)
            img_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            img_src = cv2.imread(src_path)
            if new_size is not None:
                img_mask = cv2.resize(img_mask, (new_size, new_size))
                img_src = cv2.resize(img_src, (new_size, new_size))
            ret, img_mask = cv2.threshold(img_mask, 127, 255, 0)
            im2, contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            x_mask, y_mask, w_mask, h_mask = cv2.boundingRect(cnt)
            t_ = np.count_nonzero(img_mask[0] == 255)
            r_ = np.count_nonzero(img_mask[:, -1] == 255)
            b_ = np.count_nonzero(img_mask[-1] == 255)
            l_ = np.count_nonzero(img_mask[:, 0] == 255)
            sides_sum = [t_, r_, b_, l_]
            index = np.argmax(sides_sum)
            lc = locations[index]
            area = max(int(np.sum(img_mask / 255) * 100 / float(img_mask.shape[0] * img_mask.shape[1])), 1)
        except Exception:
            continue
        if len(pkl_data['hard']) != 0:
            pkl_data[key]['mask'].append(img_mask)
            pkl_data[key]['src'].append(img_src)
            pkl_data[key]['location'].append(lc)
            pkl_data[key]['w_mask'].append(w_mask)
            pkl_data[key]['h_mask'].append(h_mask)
            pkl_data[key]['area'].append(area)
        else:
            pkl_data[key] = {}
            pkl_data[key]['mask'] = [img_mask]
            pkl_data[key]['src'] = [img_src]
            pkl_data[key]['location'] = [lc]
            pkl_data[key]['w_mask'] = [w_mask]
            pkl_data[key]['h_mask'] = [h_mask]
            pkl_data[key]['area'] = [area]

    # 针对不同的python版本，pickle文件的存储形式不同
    if platform.python_version().startswith("3"):
        pkl.dump(pkl_data, open(pkl_path, 'wb'))
    else:
        pkl.dump(pkl_data, file(pkl_path, 'wb'))
# 合并pkl文件
def merge_pkl_files(pkl_path_list, merged_path):
    pkl_data = {}
    if platform.python_version().startswith("3"):
        for pkl_path in pkl_path_list:
            pkl_data_tmp = pkl.load(open(pkl_path, 'rb'))
            pkl_data.update(pkl_data_tmp)
    else:
        for pkl_path in pkl_path_list:
            pkl_data_tmp = pkl.load(file(pkl_path, 'rb'))
            pkl_data.update(pkl_data_tmp)
    # 针对不同的python版本，pickle文件的存储形式不同
    if platform.python_version().startswith("3"):
        pkl.dump(pkl_data, open(merged_path, 'wb'))
    else:
        pkl.dump(pkl_data, file(merged_path, 'wb'))
# 提取pkl文件
def get_pkl_data(pkl_path):
    if platform.python_version().startswith("3"):
        pkl_data = pkl.load(open(pkl_path, 'rb'))
    else:
        pkl_data = pkl.load(file(pkl_path, 'rb'))
    return pkl_data
# 泊松编辑，合成手指遮挡
def aug_poisson(pkl_data, w_ratio, h_ratio, dst_img, location="corner"):
    """
    在线泊松编辑
    :param pkl_data: 用于扩充的手指数据
    :param w_ratio: mask矩形宽占目标图片的宽的比重
    :param h_ratio:mask矩形高占目标图片的高的比重
    :param dst_img: 需要被混入手指的图片(read by cv2)
    :param location: 指定手指的位置(字符串)，可供选择的如下:corner,side,hard
    :return: 混合后的图像, 混合后的mask
    """
    dst_img = dst_img.copy()
    # choose a key that represents the ratio of finger area to the whole image
    key = location
    if key not in ['corner', 'side', 'hard']:
        raise RuntimeError("only support the corner, side and hard locations")
    mask_img = None
    while mask_img is None:
        #random.seed()
        index = random.choice(range(len(pkl_data[key]['mask'])))
        # choose a mask
        img_mask = pkl_data[key]['mask'][index].copy()
        # choose the related src img
        img_src = pkl_data[key]['src'][index].copy()
        # get the location of the finger image
        lc0 = pkl_data[key]['location'][index]
        w_mask = pkl_data[key]['w_mask'][index]
        h_mask = pkl_data[key]['h_mask'][index]

        # 随机左右翻转
        if random.randint(0, 100) < 50:
            img_mask = cv2.flip(img_mask, 1)
            img_src = cv2.flip(img_src, 1)
            if 'r' in lc0:
                lc0 = lc0.replace('r', 'l')
            elif 'l' in lc0:
                lc0 = lc0.replace('l', 'r')
        # 随机上下翻转
        if random.randint(0, 100) < 50:
            img_mask = cv2.flip(img_mask, 0)
            img_src = cv2.flip(img_src, 0)
            if 'b' in lc0:
                lc0 = lc0.replace('b', 't')
            elif 't' in lc0:
                lc0 = lc0.replace('t', 'b')

        # 随机顺时针90
        if random.randint(0, 100) < 50:
            img_mask = cv2.rotate(img_mask, cv2.ROTATE_90_CLOCKWISE)
            img_src = cv2.rotate(img_src, cv2.ROTATE_90_CLOCKWISE)
            w_mask, h_mask = h_mask, w_mask
            if lc0 in side_locations:
                index = side_locations.index(lc0)
                lc0 = side_locations[(index+1) % len(side_locations)]
            elif lc0 in corner_locations:
                index = corner_locations.index(lc0)
                lc0 = corner_locations[(index + 1) % len(corner_locations)]
            elif lc0 in corner3_locations:
                index = corner3_locations.index(lc0)
                lc0 = corner3_locations[(index + 1) % len(corner3_locations)]

        # 随机逆时针90
        if random.randint(0, 100) < 50:
            img_mask = cv2.rotate(img_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img_src = cv2.rotate(img_src, cv2.ROTATE_90_COUNTERCLOCKWISE)
            w_mask, h_mask = h_mask, w_mask
            if lc0 in side_locations:
                index = side_locations.index(lc0)
                lc0 = side_locations[(index - 1 + len(side_locations)) % len(side_locations)]
            elif lc0 in corner_locations:
                index = corner_locations.index(lc0)
                lc0 = corner_locations[(index - 1 + len(corner_locations)) % len(corner_locations)]
            elif lc0 in corner3_locations:
                index = corner3_locations.index(lc0)
                lc0 = corner3_locations[(index - 1 + len(corner3_locations)) % len(corner3_locations)]
        mask_img, src_img, lt, rb = get_mask_and_src(img_mask, img_src, dst_img, lc0, w_mask, h_mask, w_ratio, h_ratio)

    mask_img_copy = mask_img.copy()

    result = poisson_edit(src_img, mask_img, dst_img)
    return result, mask_img_copy, lt, rb
def test():
    pkl_path = '/home/gezhipeng/workspace/DataProcessing/Datas/Finger/finger_templates_merged.pkl'
    pkl_data = get_pkl_data(pkl_path)
    dst_image_path = '/home/gezhipeng/workspace/DataProcessing/0-demo_images/poisson_dst.jpg'
    dst_img = cv2.imread(dst_image_path)
    for i in range(10):
        result, mask_img, lt, rb = aug_poisson(pkl_data, w_ratio=0.5, h_ratio=0.5, dst_img=dst_img, location="hard")
        imwrite_image_label('result'+ str(i+1).zfill(3)+'.png'.format(i), [mask_img, dst_img, result], ['mask', 'dst', 'result'])
if __name__ == '__main__':
    test()