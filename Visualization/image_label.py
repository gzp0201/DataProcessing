# -*- coding:utf-8 -*-
"""
create by gezhipeng
create on 18-12-24 下午1:45
func:
"""
import numpy as np
import cv2
import sys
sys.path.append('../')
def imwrite_image_label(save_path, imgs, labels):
    """
    针对分类模型，将图片和label可视化显示
    :param save_path:
    :param imgs:
    :param labels:
    :return:
    """
    n_imgs = len(labels)
    block_row = int(np.sqrt(n_imgs))
    if block_row ** 2 < n_imgs:
        block_row += 1
    size = 224
    res = np.zeros([block_row * size, block_row * size, 3])
    for i in range(n_imgs):
        img, label = imgs[i], labels[i]
        img = cv2.resize(img, (size, size))
        row = i // block_row
        col = i % block_row
        if len(img.shape) != 3:
            if len(img.shape) == 2:
                res[row * size: (row + 1) * size, col * size: (col + 1) * size, 0] = img
                res[row * size: (row + 1) * size, col * size: (col + 1) * size, 1] = img
                res[row * size: (row + 1) * size, col * size: (col + 1) * size, 2] = img
            else:
                raise RuntimeError("The image is WRONG")

        else:
            res[row * size: (row + 1) * size, col * size: (col + 1) * size, :] = img[:, :, :3]
        cv2.putText(res, str(label), (col * size, row * size+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
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
    imwrite_image_label(save_path, imgs=imgs, labels=[1, 2, 3])

if __name__ == '__main__':
    test()