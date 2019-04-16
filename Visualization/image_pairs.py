# -*- coding:utf-8 -*-
"""
create by gezhipeng
create on 18-12-24 下午1:46
func: 
"""
import numpy as np
import cv2
import sys
import os
sys.path.append('../')
def imwrite_image_pairs(save_path, **kwargs):
    """
    针对图像处理，可以对多组图片同事可视化显示
    :param save_path:
    :param kwargs:
    :return:
    """
    block_row = len(list(kwargs.values())[0])
    block_col = len(kwargs.keys())
    size = 224
    res = np.zeros([block_row * size, block_col * size, 3])
    for i, k in enumerate(kwargs.keys()):
        imgs = kwargs[k]
        for row in range(block_row):
            img = imgs[row]
            img = cv2.resize(img, (size, size))
            if len(img.shape) != 3:
                if len(img.shape) == 2:
                    res[row * size: (row + 1) * size, i * size: (i + 1) * size, 0] = img
                    res[row * size: (row + 1) * size, i * size: (i + 1) * size, 1] = img
                    res[row * size: (row + 1) * size, i * size: (i + 1) * size, 2] = img
                else:
                    raise RuntimeError("The image is WRONG")

            else:
                res[row * size: (row + 1) * size, i * size: (i + 1) * size, :] = img[:, :, :3]
        cv2.putText(res, k, (i*size, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(save_path, res)

def test():
    import os
    image_dir = '../0-demo_images'
    imgs = []
    images = os.listdir(image_dir)
    for image in images:
        image_path = os.path.join(image_dir, image)
        img = cv2.imread(image_path)
        imgs.append(img)
    save_path = '../res.png'
    imwrite_image_pairs(save_path, imgs=imgs, imgs2=imgs, imgs3=imgs)

if __name__ == '__main__':
    # test()

    src_dir = '/home/gezhipeng/datas/OCR/DocUNet/baidu/data_pairs/scan/src'
    distorted_dir = '/home/gezhipeng/datas/OCR/DocUNet/baidu/data_pairs/scan/mask'
    mask_dir = '/home/gezhipeng/datas/OCR/DocUNet/baidu/data_pairs/scan/distorted'

    src_imgs = []
    distorted_imgs = []
    mask_imgs = []
    for i in range(1, 21):
        image_name = '2019010415002905777{}.png'.format(str(i).zfill(2))
        src_img = cv2.imread(os.path.join(src_dir, image_name))
        distorted_img = cv2.imread(os.path.join(distorted_dir, image_name))
        mask_img = cv2.imread(os.path.join(mask_dir, image_name), cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)

        src_imgs.append(src_img)
        distorted_imgs.append(distorted_img)
        mask_imgs.append(mask_img)

    imwrite_image_pairs(save_path='../res.png', src_imgs=src_imgs,distorted_imgs=mask_imgs,mask_imgs=distorted_imgs)
