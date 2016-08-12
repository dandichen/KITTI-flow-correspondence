import os
import cv2
import numpy as np
import timeit

import keypoint
import BF_matcher
from ORB import ORB_point
from BF_matcher import BFMatcher

import visualization.keypoint_pairs as vis_kp
import visualization.matchers as vis_matchers

import evaluation.form as eval_form
import evaluation.percentage as eval_per

__author__ = "Dandi Chen"

print cv2.__version__

img_path = '/mnt/scratch/haoyiliang/KITTI/SceneFLow2015/data_scene_flow/training/image_2'
match_path = '/mnt/scratch/DandiChen/keypoint/KITTI/grid/ORB-BF'

flow_gt_path = '/mnt/scratch/haoyiliang/KITTI/SceneFLow2015/data_scene_flow/training/flow_occ'

img_num = len([name for name in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, name))])
flow_num = img_num - 1  # continuous two frames

pair_num = img_num/2
# pair_num = 2

t_mat = []
per_mat = []
ol_num_mat = []
gt_num_mat = []

corr_X_mat = []
corr_Y_mat = []
err_ratio_mat = []
ang_err_mat = []
end_pt_err_mat = []

x_blk_size = 150
y_blk_size = 150

for img in range(pair_num):
    print ''
    print 'img number: ', img

    img_x = []
    img_y = []

    img_path1 = os.path.join(img_path, str(img).zfill(6) + '_10.png')
    img_path2 = os.path.join(img_path, str(img).zfill(6) + '_11.png')

    start = timeit.default_timer()
    # ORB keypoint
    orb = ORB_point(200)
    img1, img2 = keypoint.read_img_pair(img_path1, img_path2)
    height, width, _ = img1.shape

    # grid: patch_idx1 = patch_y_idx * (x_num - 1) + patch_x_idx
    img_patch1, x_trans1, y_trans1, _, _, _, _ = keypoint.grid_img(img1, width, height, x_blk_size, y_blk_size)
    img_patch2, x_trans2, y_trans2, patch_x_idx, patch_y_idx, x_num, y_num = keypoint.grid_img(img2, width, height, \
                                                                                               x_blk_size, y_blk_size)
    des = np.zeros((1, 32), dtype=np.uint8)
    kp = []
    for patch_idx2 in range(len(img_patch2)):
        # print 'idx = ', patch_idx2
        kp2, des2 = orb.single_img_keypoint(img_patch2[patch_idx2])
        # print 'kp = ', len(kp2)
        # print np.array(des2).shape
        if des2 == None:
            continue
        des = np.concatenate((des, des2))
        kp += kp2
        # des.extend(des2)
        # print 'des = ', len(des)
    des = np.delete(des, 0, 0)


    for patch_idx1 in range(len(img_patch1)):
        # print ''
        # print 'patch index: ', patch_idx1

        # # patch-neighbor keypoint pairs
        # neighbor0 = (patch_y_idx - 1) * (x_num - 1) + (patch_x_idx - 1)
        # neighbor1 = (patch_y_idx - 1) * (x_num - 1) + patch_x_idx
        # neighbor2 = (patch_y_idx - 1) * (x_num - 1) + (patch_x_idx + 1)
        # neighbor3 = patch_y_idx * (x_num - 1) + (patch_x_idx - 1)
        # neighbor4 = patch_y_idx * (x_num - 1) + (patch_x_idx + 1)
        # neighbor5 = (patch_y_idx + 1) * (x_num - 1) + (patch_x_idx - 1)
        # neighbor6 = (patch_y_idx + 1) * (x_num - 1) + patch_x_idx
        # neighbor7 = (patch_y_idx + 1) * (x_num - 1) + (patch_x_idx + 1)

        # patch-patch keypoint pairs
        # kp1, des1, kp2, des2 = orb.get_keypoint(np.array(img_patch1[patch_idx1]), np.array(img_patch2[patch_idx1]))

        # patch-image keypoint pairs
        kp1, des1 = orb.single_img_keypoint(img_patch1[patch_idx1])
        # vis_kp.vis_pt_pairs(img1, kp1, x_trans1[patch_idx1], y_trans1[patch_idx1], \
        #                     img2, kp2, x_trans2[patch_idx1], y_trans2[patch_idx1])

        # BFMatcher
        bfm = BFMatcher()
        matches = bfm.get_matcher(des1, des)

        # # display matches.distance
        # dis = []
        # for mat_idx in range(len(matches)):
        #     dis.append(matches[mat_idx].distance)
        # plt.figure()
        # plt.plot(np.arange(0, len(dis)), dis)
        # plt.waitforbuttonpress()

        if matches == None:
            continue

        # x, y, flow_X, flow_Y = bfm.get_match_flow(kp1, kp2, matches, \
        #                                            os.path.join(match_path + '/' + str(img).zfill(6), \
        #                                            str(patch_idx1).zfill(4) + '.npz'))
        out_img = vis_matchers.vis_matches(img1, kp1, x_trans1[patch_idx1], y_trans1[patch_idx1], \
                                           img2, kp, x_trans2[patch_idx1], y_trans2[patch_idx1], matches)
        cv2.imwrite(os.path.join(match_path + '/' + str(img).zfill(6), str(patch_idx1).zfill(4) + '.png'), out_img)

        # evaluation

        # # keypoint before matching
        patch_x, patch_y = keypoint.grid2mask(kp1, x_trans1[patch_idx1], y_trans1[patch_idx1])

        img_x.extend(patch_x)
        img_y.extend(patch_y)

        # keypoint after matching
        # patch_x, patch_y = BF_matcher.grid2mask(kp1, x_trans1[patch_idx1], y_trans1[patch_idx1], matches)
        #
        # img_x.extend(patch_x)
        # img_y.extend(patch_y)

    end = timeit.default_timer()
    t = end - start
    print 'time = ', t
    t_mat.append([t])

    flow_X_gt, flow_Y_gt, flow_mask_gt = eval_form.read_gt(os.path.join(flow_gt_path, str(img).zfill(6) + '_10.png'), \
                                                           width, height)

    # # flow ground truth-mask
    # flow_mask_gt_region = eval_form.get_flow_region(flow_X_gt, flow_Y_gt, width, height)
    # flow_mask = eval_form.convert_coordinate(img_x, img_y, flow_mask_gt)
    # per, ol_num, gt_num = eval_per.get_overlap_per(flow_mask, flow_mask_gt_region)

    # flow ground truth-exact pixel
    flow_mask = eval_form.convert_coordinate(img_x, img_y, flow_mask_gt)
    per, ol_num, gt_num = eval_per.get_overlap_per(flow_mask, flow_mask_gt)

    print 'percentage = ', per
    per_mat.append([per])
    ol_num_mat.append([ol_num])
    gt_num_mat.append([gt_num])

print ''
print 'ave time = ', np.mean(t_mat)
print 'ave percentage = ', np.mean(per_mat)
print 'ave overlap num = ', np.mean(ol_num_mat)
print 'ave ground truth num = ', np.mean(gt_num_mat)

