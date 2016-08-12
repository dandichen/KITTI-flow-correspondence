import cv2
import numpy as np

__author__ = "Dandi Chen"

def read_gt(gt_path, width, height):
    flow_gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

    # KITTI definition
    flow_X_gt = (np.float_(flow_gt[:, :, 2]) - 2 ** 15) / 64.0  # [-512..+512]
    flow_Y_gt = (np.float_(flow_gt[:, :, 1]) - 2 ** 15) / 64.0  # [-512..+512]
    flow_mask_gt = np.array(flow_gt[:, :, 0], dtype=bool)

    return flow_X_gt, flow_Y_gt, flow_mask_gt

def normaliztion(old_arr, start, end):
    old_min = np.amin(old_arr)
    old_range = np.amax(old_arr) - old_min

    new_range = end - start
    new_arr = [(n - old_min) / old_range * new_range + start for n in old_arr]

    return np.array(new_arr)

def convert(old_x, old_y, old_flow_X, old_flow_Y, flow_mask_gt):
    flow_X_norm = normaliztion(old_flow_X, -512, 512)
    flow_Y_norm = normaliztion(old_flow_Y, -512, 512)

    height, width = flow_mask_gt.shape
    flow_mask = np.zeros((height, width), dtype=bool)
    new_flow_X = np.zeros((height, width))
    new_flow_Y = np.zeros((height, width))

    for i in range(len(old_x)):
        flow_mask[old_y[i], old_x[i]] = True
        new_flow_X[old_y[i], old_x[i]] = flow_X_norm[i]
        new_flow_Y[old_y[i], old_x[i]] = flow_Y_norm[i]

    return new_flow_X, new_flow_Y, flow_mask

def convert_coordinate(old_x, old_y, flow_mask_gt):
    height, width = flow_mask_gt.shape
    flow_mask = np.zeros((height, width), dtype=bool)

    for i in range(len(old_x)):
        flow_mask[old_y[i], old_x[i]] = True

    return flow_mask

def get_flow_region(flow_X, flow_Y, width, height):
    mask_region = np.zeros((height, width), dtype=bool)
    flow_X_norm = normaliztion(flow_X, 0, 255)
    flow_Y_norm = normaliztion(flow_Y, 0, 255)

    for j in range(height):
        for i in range(width):
            # rectangle: (j, i), (j + int(round(flow_Y_norm[j, i])), i + int(round(flow_X_norm[j, i])))
            mask_region[j:j + int(round(flow_Y_norm[j, i]))][i:i + int(round(flow_X_norm[j, i]))] = True

    return mask_region
