import math
import numpy as np

__author__ = 'Dandi Chen'


def grid_img(img, width, height, x_blk_size=8, y_blk_size=8):
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
                block = img[i*y_blk_size:(i+1)*y_blk_size,
                        j*x_blk_size:(j+1)*x_blk_size]
                # print i*y_blk_size, (i+1)*y_blk_size, \
                #     j*x_blk_size, (j+1)*x_blk_size
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
