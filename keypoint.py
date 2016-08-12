import cv2
import math
import numpy as np

__author__ = "Dandi Chen"

def read_img_pair(img_path1, img_path2):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    return img1, img2

def grid_img(img, width, height, x_blk_size=100, y_blk_size=100):
    x_num = int(math.ceil((width - x_blk_size) / x_blk_size)) + 2
    y_num = int(math.ceil((height - y_blk_size) / y_blk_size)) + 2

    img_patch = []
    x_trans = []
    y_trans = []
    patch_x_idx = []
    patch_y_idx = []
    for i in range(y_num):
        for j in range(x_num):
            if i != y_num - 1 and j != x_num - 1:
                block = img[i*y_blk_size:(i+1)*y_blk_size, j*x_blk_size:(j+1)*x_blk_size]
                # print i*y_blk_size, (i+1)*y_blk_size, j*x_blk_size, (j+1)*x_blk_size
                # print block.shape
                img_patch.append(np.array(block))
                x_trans.append(j*x_blk_size)
                y_trans.append(i*y_blk_size)
                patch_x_idx.append(j)
                patch_y_idx.append(i)
            # elif i == y_num - 1 and j == x_num - 1:
            #     block = img[i*y_blk_size:height, j*x_blk_size:width]
            #     # print i*y_blk_size, height, j*x_blk_size, width
            #     print block.shape
            #     img_patch.append(block)
            # elif i != y_num - 1 and j == x_num - 1:
            #     block = img[i*y_blk_size:(i+1)*y_blk_size, j*x_blk_size:width]
            #     # print i*y_blk_size, (i+1)*y_blk_size, j*x_blk_size, width
            #     print block.shape
            #     img_patch.append(block)
            # else:
            #     block = img[i*y_blk_size:height, j*x_blk_size:(j+1)*x_blk_size]
            #     # print i*y_blk_size, height, j*x_blk_size, (j+1)*x_blk_size
            #     print block.shape
            #     img_patch.append(block)
    return img_patch, x_trans, y_trans, patch_x_idx, patch_y_idx, x_num, y_num

def grid2mask(kp, x_trans, y_trans):    # keypoints in each image patch
    x_mat = []
    y_mat = []
    for pt_idx in range(len(kp)):
        (x, y) = kp[pt_idx].pt
        x_mat.extend([int(x) + x_trans])
        y_mat.extend([int(y) + y_trans])
    return x_mat, y_mat

class Keypoint(object):
    def __init__(self, kp_num=200):
        self.kp_num = kp_num

        self.x = np.arange(self.kp_num)
        self.y = np.arange(self.kp_num)
        self.flow_X = np.arange(self.kp_num)
        self.flow_Y = np.arange(self.kp_num)
