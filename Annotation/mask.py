# -*- coding:utf-8 -*-
"""
create by gezhipeng
create on 18-12-26 下午1:12
func: 
"""

import os
import numpy as np
import cv2
import sys
sys.path.append('../')

class MaskPainter(object):
    def __init__(self, image_dir=None, mask_dir=None, painter_size=20, window_size=256):
        """
        :param image_dir:
        :param painter_size: 画笔大小
        :param window_size:
        """
        extensions = ['jpg', 'jpeg', 'png', 'tif', 'bmp']
        if image_dir is not None:
            try:
                self.image_dir = image_dir
                assert os.path.exists(self.image_dir), "image dir not exists"
                images = os.listdir(self.image_dir)
                self.image_paths = [os.path.join(image_dir, image) for image in images if
                                    image.split('.')[-1].lower() in extensions]
                assert len(self.image_paths) > 0, "no image found"
            except Exception:
                sys.exit()
        else:
            self.image_dir = None
        assert os.path.exists(mask_dir), "mask dir not exists"
        self.mask_dir = mask_dir
        self.painter_size = painter_size
        self.window_size = window_size
        self.to_draw = False
        self.window_name = "Draw mask. s:save; r:reset; q:quit"

    def _paint_mask_handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.to_draw = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.to_draw:
                # cv2.rectangle(self.image, (x - self.size, y - self.size),
                #               (x + self.size, y + self.size),
                #               (0, 255, 0), -1)
                # cv2.rectangle(self.mask, (x - self.size, y - self.size),
                #               (x + self.size, y + self.size),
                #               (255, 255, 255), -1)
                cv2.circle(self.image, (x, y), radius=self.painter_size, color=(0, 255, 0), thickness=-1)
                cv2.circle(self.mask, (x, y), radius=self.painter_size, color=(255, 255, 255), thickness=-1)
                cv2.imshow(self.window_name, self.image)
        elif event == cv2.EVENT_LBUTTONUP:
            self.to_draw = False

    def paint_mask_single(self, image_path):
        """
        paint one image
        :return:
        """
        self.image = cv2.imread(image_path)
        h, w = self.image.shape[: 2]
        self.image = cv2.resize(self.image, (self.window_size, self.window_size))
        self.mask = np.zeros(self.image.shape, dtype=np.uint8)
        image_copy = self.image.copy()
        mask_copy = self.mask.copy()

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name,
                             self._paint_mask_handler)
        image_name = os.path.split(image_path)[-1]
        image_ext = image_name.split('.')[1]
        mask_name = image_name.replace(image_ext, 'png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        if os.path.exists(mask_path):
            return
        while True:
            cv2.imshow(self.window_name, self.image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                self.image = image_copy.copy()
                self.mask = mask_copy.copy()

            elif key == ord("s"):
                break

            elif key == ord("q"):
                cv2.destroyAllWindows()
                exit()

        cv2.imshow("Press any key to save the mask", self.mask)
        cv2.waitKey(0)
        self.mask = cv2.resize(self.mask, (w, h))
        self.mask = self.mask[:, :, 1]
        rest, self.mask = cv2.threshold(self.mask, 127, 255, 0)
        cv2.imwrite(mask_path, self.mask)
        cv2.destroyAllWindows()

    def paint_mask_for_dir(self):
        for image_path in self.image_paths:
            self.paint_mask_single(image_path)


if __name__ == '__main__':
    mp = MaskPainter(image_dir='/home/gezhipeng/workspace/DataProcessing/0-demo_images', mask_dir='../')
    mp.paint_mask_for_dir()
