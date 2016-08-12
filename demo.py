import sys
sys.path.insert(0, '/mnt/scratch/third-party-packages/libopencv_3.1.0/lib/python')
sys.path.insert(1, '/mnt/scratch/third-party-packages/libopencv_3.1.0/lib')

import os
import timeit
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from bbox.boundingbox import BoundingBox
from bbox.boundingbox import BoundingBoxList

from keypoint_detection import keypoint_pairs
from keypoint_detection.keypoint_pairs import KeypointList
from keypoint_detection.keypoint_pairs import KeypointPairList
from keypoint_detection.ORB import ORB_point
from keypoint_detection.FAST import FAST_point
from keypoint_detection.BRIEF import BRIEF_point
from keypoint_detection.SIFT import SIFT_point
from keypoint_detection.SURF import SURF_point
from keypoint_detection.HarrisCorner import Harris_point
from keypoint_detection.ShiTomasiCorner import ShiTomasi_point

from keypoint_matching.brute_force import BruteForceMatcherList
import keypoint_matching.brute_force as bf

from keypoint_matching.flow import Flow

__author__ = 'Dandi Chen'


img_path = '/mnt/scratch/haoyiliang/KITTI/SceneFLow2015/data_scene_flow/training/image_2/'
flow_gt_path = '/mnt/scratch/haoyiliang/KITTI/SceneFLow2015/data_scene_flow/training/flow_occ'
box_path = '/mnt/scratch/dandichen/KITTI/bbox/fill/test/'

eval_path = '/mnt/scratch/dandichen/KITTI/evaluation/'
kp_path = eval_path + 'keypoint/'
match_path = eval_path + 'matches/'
vel_path = eval_path + 'velocity/'
flow_path = eval_path + 'flow/'
overlap_path = match_path + 'overlap/'
non_overlap_path = match_path + 'non_overlap/'
voting_non_overlap_path = match_path + 'voting/non_overlap/'
voting_overlap_path = match_path + 'voting/params/ORB-200/overlap/'


img_num = len([name for name in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, name))])
flow_num = img_num - 1  # continuous two frames

pair_num = img_num / 2
# pair_num = 3

fix_height = 1080
fix_width = 1920

t = []
t_box_list = []
t_box_match_list = []
t_kp_list = []
t_matcher_list = []
t_good_matcher_list = []
t_wgt_matcher_list = []
diff_angs = []

for img in range(pair_num):
    print ''
    print 'img number: ', img

    img_path1 = os.path.join(img_path, str(img).zfill(6) + '_10.png')
    img_path2 = os.path.join(img_path, str(img).zfill(6) + '_11.png')

    img1, img2 = keypoint_pairs.read_img_pair(img_path1, img_path2)
    height, width, _ = img1.shape

    start = timeit.default_timer()

    # box_pair = BoundingBoxPairList()
    # box_pair.get_box_pair(img1, img2)
    # box_pair.vis_box_pair(img1, img2)
    #
    # # for each bounding box
    # for idx in range(box_pair.num):
    #     # keypoint detection
    #     pt1 = ORB_point(box_pair.list1[idx])
    #     pt2 = ORB_point(box_pair.list2[idx])
    #     pt1.get_keypoint(img1[int(box_pair.list1[idx].top_left_y):int(box_pair.list1[idx].bottom_right_y),
    #                       int(box_pair.list1[idx].top_left_x):int(box_pair.list1[idx].bottom_right_x)])
    #     pt2.get_keypoint(img2[int(box_pair.list2[idx].top_left_y):int(box_pair.list2[idx].bottom_right_y),
    #                       int(box_pair.list2[idx].top_left_x):int(box_pair.list2[idx].bottom_right_x)])
    #     t_kp = timeit.default_timer()
    #     print 'keypoint extraction time:', t_kp - start
    #     print 'keypoint number:', pt1.length, pt2.length
    #     print ''
    #     pt_pair = KeypointPairList(pt1, pt2)
    #     # pt_pair.get_neighbor_score()
    #     # pt_pair.get_grid_neighbor_score()
    #     # pt_pair.get_grid_kernel_neighbor_score()
    #     # pt_pair.vis_pt_pairs(img1, img2)
    #     # pt_pair.write_pt_pairs(img1, img2, os.path.join(kp_path, str(img).zfill(6) + '_10.png'),
    #     #                         os.path.join(kp_path, str(img).zfill(6) + '_11.png'))

    # bounding box
    res1 = np.ones((fix_height, fix_width, 3), dtype=np.uint8) * 255
    res2 = np.ones((fix_height, fix_width, 3), dtype=np.uint8) * 255

    res1[0: height, 0: width, :] = img1
    res2[0: height, 0: width, :] = img2

    bboxList1 = BoundingBoxList()
    bboxList2 = BoundingBoxList()

    bboxList1.get_box(res1, fix_width, fix_height)
    bboxList2.get_box(res2, fix_width, fix_height)
    t_box = timeit.default_timer()

    t_box_match_0 = timeit.default_timer()
    bboxList1.bbox_matching(bboxList2)
    t_box_match = timeit.default_timer()

    flow_gt = Flow(None, width, height)
    flow_gt.read(os.path.join(flow_gt_path, str(img).zfill(6) + '_10.png'))

    print 'bounding box extraction time:', t_box - start
    print 'bounding box matching time:', t_box_match - t_box_match_0
    t_box_list.append([t_box - start])
    t_box_match_list.append([t_box_match - t_box_match_0])

    for idx in range(bboxList1.num):
        print 'bbox ID = ', idx
        bbox1 = bboxList1[idx]
        bbox2 = bboxList2[idx]

        t_kp_0 = timeit.default_timer()

        # keypoint detection
        pt1 = ORB_point(bbox1)
        pt2 = ORB_point(bbox2)
        kp1, des1 = pt1.get_keypoint(res1[int(bbox1.top_left_y):
                         int(bbox1.bottom_right_y),
                         int(bbox1.top_left_x):
                         int(bbox1.bottom_right_x)],
                         fix_width, fix_height)
        kp2, des2 = pt2.get_keypoint(res2[int(bbox2.top_left_y):
                         int(bbox2.bottom_right_y),
                         int(bbox2.top_left_x):
                         int(bbox2.bottom_right_x)],
                         fix_width, fix_height)
        if kp1 == [] or kp2 == []:
            continue
        t_kp = timeit.default_timer()
        print 'keypoint extraction time:', t_kp - t_kp_0
        t_kp_list.append([t_kp - t_kp_0])
        print 'keypoint number:', pt1.length, pt2.length
        pt_pair = KeypointPairList(pt1, pt2)

        # t_matcher_0 = timeit.default_timer()
        # matchIdx = pt_pair.get_neighbor_score(1)
        # pt_pair.clip_val(matchIdx)
        #
        # bfm = BruteForceMatcherList(pt_pair)
        # bfm.get_matcher()
        # t_matcher = timeit.default_timer()
        # print 'matcher time = ', t_matcher - t_matcher_0
        # bfm.vis_matches(img1, img2, 1, 0, bfm.length)
        #
        # bfm.write_matches(img1, img2, os.path.join(match_path, str(img).zfill(6) +
        #                                            '_10_non_overlap_match.png'),
        #                   1, 0, bfm.length)
        #
        # bfm.write_matches_overlap(img1, img2, os.path.join(match_path,
        #                                                    str(img).zfill(6) +
        #                                                    '_10_overlap_match.png'),
        #                           1, 0, bfm.length)

        #
        # top 10 smallest distance
        # rows1, cols1, _ = img1.shape
        # rows2, cols2, _ = img2.shape
        # out_img = np.zeros((rows1 + rows2, max([cols1, cols2]), 3), dtype=np.uint8)
        # out_img[0:rows1, 0:cols1, :] = img1
        # out_img[rows1:rows1 + rows2,  0:cols2, :] = img2
        # out_img = img1.copy()
        #
        # start = 0
        # # end = 200
        # end = pt_pair.list1.length
        #
        # t_matcher_0 = timeit.default_timer()
        #
        # matchIdx = pt_pair.get_neighbor_score(1)
        #
        # for idx in range(start, end):
        #     # print 'idx = ', idx
        #     kp1 = []
        #     des1 = []
        #     kp2 = []
        #     des2 = []
        #     # print matchIdx[idx]
        #     for i in range(len(matchIdx[idx])):
        #         index = matchIdx[idx]
        #
        #         kp1.append(pt_pair.list1.kp[idx])
        #         des1.append(pt_pair.list2.des[idx])
        #
        #         kp2.append(pt_pair.list2.kp[index[i]])
        #         des2.append(pt_pair.list2.des[index[i]])
        #
        #     matches = bf.get_matcher_1vM(np.array(des1), np.array(des2))
        #     t_matcher = timeit.default_timer()
        #     print 'matcher time = ', t_matcher - t_matcher_0
        #     out_img = bf.vis_matches(out_img, kp1, kp2, bbox1, bbox2, matches, 0, len(matches))
        # # cv2.imwrite('matches.png', out_img)
        #
        # cv2.imshow('matches', out_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        # # smallest distance
        # out_img = img1.copy()
        #
        # start = 0
        # # end = 200
        # end = pt_pair.list1.length
        #
        # t_matcher_0 = timeit.default_timer()
        # matchIdx = pt_pair.get_neighbor_score(1)
        #
        #
        # matchesList = []
        # for idx in range(start, end):
        #     matches = bf.get_matcher_1v1(np.array([pt_pair.list1.des[idx]]),
        #                                  np.array(pt_pair.list2.des[matchIdx[idx]]))
        #     # out_img = bf.vis_matches(out_img, [pt_pair.list1.kp[idx]],
        #     #                          [pt_pair.list2.kp[matchIdx[idx]]],
        #     #                          bbox1, bbox2, matches, 0, len(matches))
        #     matchesList += matches
        #
        # kp2 = []
        # for i in np.asarray(matchIdx):
        #     kp2.append(pt_pair.list2.kp[i])
        # pt_pair.list2.kp = kp2
        #
        # bf.vis_matchesList(out_img, pt_pair.list1.kp, kp2, bbox1, bbox2, matchesList[start:end], start, end)
        #
        # t_matcher = timeit.default_timer()
        # print 'matcher time = ', t_matcher - t_matcher_0
        # cv2.imshow('matches', out_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # BFMatcher
        t_matcher_0 = timeit.default_timer()

        matchIdx = pt_pair.get_neighbor_score(1)
        print 'get_neighbor_score'
        print 'pre = ', pt_pair.neighbor_pre, 'score = ', pt_pair.neighbor_score
        # pt_pair.get_grid_neighbor_score()
        # print 'get_grid_neighbor_score'
        # print 'pre = ', pt_pair.neighbor_pre, 'score = ', pt_pair.neighbor_score
        # print ''
        # pt_pair.get_grid_kernel_neighbor_score()
        # print 'get_grid_kernel_neighbor_score'
        # print 'pre = ', pt_pair.neighbor_pre, 'score = ', pt_pair.neighbor_score
        # print ''
        # pt_pair.vis_mask()

        # pt_pair.vis_pt_pairs(img1, img2)
        # pt_pair.write_pt_pairs(img1, img2,
        #                        os.path.join(kp_path, str(img).zfill(6) + '_10.png'),
        #                        os.path.join(kp_path, str(img).zfill(6) + '_11.png'))

        pt_pair.clip_val(matchIdx)

        bfm = BruteForceMatcherList(pt_pair)
        bfm.get_matcher()
        t_matcher = timeit.default_timer()
        print 'matcher number:', bfm.length
        print 'matcher time :', t_matcher - t_matcher_0
        t_matcher_list.append([t_matcher - t_matcher_0])

        t_good_matcher_0 = timeit.default_timer()
        bfm.get_good_matcher()  # Lowe's good feature threshold criteria(feature similarity distance)
        t_good_matcher = timeit.default_timer()
        print 'good matcher number:', bfm.length
        print 'good matcher time:', t_good_matcher - t_good_matcher_0
        t_good_matcher_list.append([t_good_matcher - t_good_matcher_0])

        t_wgt_matcher_0 = timeit.default_timer()
        bfm.get_wgt_dis_matcher()
        t_wgt_matcher = timeit.default_timer()
        print 'good weighted matcher number:', bfm.length
        print 'good weighted matcher time:', t_wgt_matcher - t_wgt_matcher_0
        t_wgt_matcher_list.append([t_wgt_matcher - t_wgt_matcher_0])

        # # find homography
        # t_homography_0 = timeit.default_timer()
        # bfm.orientationVoting()
        # Mat, _ = bfm.get_homography()
        # t_homography = timeit.default_timer()
        # print 'homography time:', t_good_matcher - t_good_matcher_0
        # print ''

        print 'matcher length = ', bfm.length

        end = timeit.default_timer()
        print 'total time = ', end - t_kp_0
        t.append([end - t_kp_0])

        flow = Flow(bfm, width, height)
        flow.compute(flow_path)

        if flow != None and flow_gt != None:
            angle = np.arctan2(flow.val_y, flow.val_x)
            angle = angle[angle.nonzero()]

            ctr_x = bbox1.top_left_x + bbox1.width / 2
            ctr_y = bbox1.top_left_y + bbox1.height / 2

            trans_y = bbox2.bottom_right_y

            if np.sum(angle) != 0:
                angle_list = Counter(angle)
                val, count = angle_list.most_common(1)[0]
                trans_x = trans_y / val

                gt_val_x = flow_gt.val_x[int(bbox1.top_left_y):
                                         int(bbox1.bottom_right_y),
                                         int(bbox1.top_left_x):
                                         int(bbox1.bottom_right_x)]
                gt_val_y = flow_gt.val_y[int(bbox1.top_left_y):
                                         int(bbox1.bottom_right_y),
                                         int(bbox1.top_left_x):
                                         int(bbox1.bottom_right_x)]

                angle_gt = np.arctan2(gt_val_y, gt_val_x)
                angle_gt = angle_gt[angle_gt.nonzero()]

                trans_y_gt = bbox2.bottom_right_y

                if np.sum(angle_gt) != 0:
                    angle_gt_list = Counter(angle_gt)
                    val_gt, count_gt = angle_gt_list.most_common(1)[0]
                    trans_x_gt = trans_y_gt / val_gt

                    diff_angle = (val - val_gt) * 180 / np.pi
                    print 'difference angle =', diff_angle
                    diff_angs.append(diff_angle)

                    # bfm.vis_matches(img1, img2, ctr_x, ctr_y, trans_x, trans_y,
                    #                   trans_x_gt, trans_y_gt, 1, 0, bfm.length)
                    # bfm.write_matches(img1, img2,
                    #                   os.path.join(voting_non_overlap_path,
                    #                                str(img).zfill(6) + '_10_' +
                    #                                str(idx) + '.png'),
                    #                   ctr_x, ctr_y, trans_x, trans_y,
                    #                   trans_x_gt, trans_y_gt,
                    #                   1, 0, bfm.length)
                    bfm.write_matches_overlap(img1, img2,
                                              os.path.join(voting_overlap_path,
                                                           str(img).zfill(
                                                               6) + '_10_' + str(
                                                               idx) + '.png'),
                                              ctr_x, ctr_y, trans_x, trans_y,
                                              trans_x_gt, trans_y_gt, 1, 0,
                                              bfm.length)

print ''
print 'time: mean = ', np.mean(t), 'median = ', np.median(t), \
    'max = ', np.amax(t), 'min = ', np.amin(t)
print 'box time: mean = ', np.mean(t_box_list), \
    'median = ', np.median(t_box_list), \
    'max = ', np.amax(t_box_list), 'min = ', np.amin(t_box_list)
print 'keypoint time: mean = ', np.mean(t_kp_list), \
    'median = ', np.median(t_kp_list), \
    'max = ', np.amax(t_kp_list), 'min = ', np.amin(t_kp_list)
print 'matcher time: mean = ', np.mean(t_matcher_list), \
    'median = ', np.median(t_matcher_list), \
    'max = ', np.amax(t_matcher_list), 'min = ', np.amin(t_matcher_list)
print 'good matcher time: mean = ', np.mean(t_good_matcher_list), \
    'median = ', np.median(t_good_matcher_list), \
    'max = ', np.amax(t_good_matcher_list), \
    'min = ', np.amin(t_good_matcher_list)
print 'weighted matcher time: mean = ', np.mean(t_wgt_matcher_list), \
    'median = ', np.median(t_wgt_matcher_list), \
    'max = ', np.amax(t_wgt_matcher_list), \
    'min = ', np.amin(t_wgt_matcher_list)
print 'wdifference angle: mean = ', np.mean(diff_angs), \
    'median = ', np.median(diff_angs), \
    'max = ', np.amax(diff_angs), \
    'min = ', np.amin(diff_angs)

