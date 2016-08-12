import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


img_path = '/mnt/scratch/haoyiliang/KITTI/SceneFLow2015/data_scene_flow/training/image_2'
flow_gt_path = '/mnt/scratch/haoyiliang/KITTI/SceneFLow2015/data_scene_flow/training/flow_occ'

img_num = len([name for name in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, name))])
flow_num = img_num - 1  # continuous two frames

pair_num = img_num/2

def read_gt(gt_path):
    flow_gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

    # KITTI definition
    flow_X_gt = (np.float_(flow_gt[:, :, 2]) - 2 ** 15) / 64.0  # [-512..+512]
    flow_Y_gt = (np.float_(flow_gt[:, :, 1]) - 2 ** 15) / 64.0  # [-512..+512]
    flow_mask_gt = np.array(flow_gt[:, :, 0], dtype=bool)

    # flow_X_gt = flow_gt[:, :, 0]
    # flow_Y_gt = flow_gt[:, :, 1]
    # data = mpimg.imread(gt_path)[:, :, 2]
    # flow_X_gt = mpimg.imread(gt_path)[:, :, 0]
    # flow_Y_gt = mpimg.imread(gt_path)[:, :, 1]

    # flow_mask_gt = np.zeros((height, width), dtype=bool)

    # threshold = (np.amax(data) + np.amin(data)) / 2
    # flow_mask_gt[data >= threshold] = True

    # flow_mask_gt[data > 0.0] = True

    return flow_X_gt, flow_Y_gt, flow_mask_gt

max_flow_X = []
max_flow_Y = []

min_flow_X = []
min_flow_Y = []

for img in range(pair_num):
    print ''
    print 'img number: ', img

    img_path1 = os.path.join(img_path, str(img).zfill(6) + '_10.png')
    img_path2 = os.path.join(img_path, str(img).zfill(6) + '_11.png')

    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    height, width, _ = img1.shape

    flow_X_gt, flow_Y_gt, flow_mask_gt = read_gt(os.path.join(flow_gt_path, str(img).zfill(6) + '_10.png'))

    # gt = cv2.imread(os.path.join(flow_gt_path, str(img).zfill(6) + '_10.png'))
    # flow_X_gt = gt[:, :, 0]
    # flow_Y_gt = gt[:, :, 1]

    max_flow_X.append(np.amax(flow_X_gt))
    max_flow_Y.append(np.amax(flow_Y_gt))

    min_flow_X.append(np.amin(flow_X_gt))
    min_flow_Y.append(np.amin(flow_Y_gt))

    flow_gt = np.zeros((height, width, 2))
    flow_gt[:, :, 0] = flow_X_gt
    flow_gt[:, :, 1] = flow_Y_gt

    # plt.figure()
    # plt.imshow(flow_X_gt)
    # plt.colorbar()
    # plt.waitforbuttonpress()
    #
    # plt.figure()
    # plt.imshow(flow_Y_gt)
    # plt.colorbar()
    # plt.waitforbuttonpress()
    # plt.close("all")

plt.figure()
plt.plot(np.arange(pair_num), max_flow_X, 'r', label='max_flow_X')
plt.plot(np.arange(pair_num), min_flow_X, 'b', label='min_flow_X')
plt.legend()
plt.title('max/min flow X')
plt.show()


plt.figure()
plt.plot(np.arange(pair_num), max_flow_Y, 'r', label='max_flow_Y')
plt.plot(np.arange(pair_num), min_flow_Y, 'b', label='min_flow_Y')
plt.legend()
plt.title('max/min flow Y')
plt.show()