import sys
sys.path.insert(0, '/mnt/scratch/third-party-packages/libopencv_3.1.0/lib/python')
sys.path.insert(1, '/mnt/scratch/third-party-packages/libopencv_3.1.0/lib')

import os
import timeit
import numpy as np
from collections import Counter

from bbox.boundingbox import BoundingBoxList
from keypoint_detection import keypoint_pairs
from keypoint_detection.keypoint_pairs import KeypointPairList
from keypoint_detection.ORB import ORB_point
from keypoint_matching.brute_force import BruteForceMatcherList
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
voting_overlap_path = match_path + 'voting/params/ORB-500/overlap/'
voting_non_overlap_path = match_path + 'voting/non_overlap/'

# def main():
#     img_num = len([name for name in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, name))])
#     pair_num = img_num / 2
#
#     fix_height = 1080
#     fix_width = 1920
#
#     t = []
#     t_box_list = []
#     t_kp_list = []
#     t_matcher_list = []
#     t_good_matcher_list = []
#     t_wgt_matcher_list = []
#     diff_angs = []
#
#     for img in range(pair_num):
#         print ''
#         print 'img number: ', img
#
#         img_path1 = os.path.join(img_path, str(img).zfill(6) + '_10.png')
#         img_path2 = os.path.join(img_path, str(img).zfill(6) + '_11.png')
#
#         img1, img2 = keypoint_pairs.read_img_pair(img_path1, img_path2)
#         height, width, _ = img1.shape
#
#         # bounding box
#         res1 = np.ones((fix_height, fix_width, 3), dtype=np.uint8) * 255
#         res2 = np.ones((fix_height, fix_width, 3), dtype=np.uint8) * 255
#
#         res1[0: height, 0: width, :] = img1
#         res2[0: height, 0: width, :] = img2
#
#         start = timeit.default_timer()
#
#         bboxList1 = BoundingBoxList()
#         bboxList2 = BoundingBoxList()
#
#         bboxList1.get_box(res1, fix_width, fix_height)
#         bboxList2.get_box(res2, fix_width, fix_height)
#         t_box = timeit.default_timer()
#
#         t_box_match_0 = timeit.default_timer()
#         bboxList1.bbox_matching(bboxList2)
#         t_box_match = timeit.default_timer()
#
#         print 'bounding box extraction time:', t_box - start
#         print 'bounding box matching time:', t_box_match - t_box_match_0
#         t_box_list.append([t_box - start])
#         t_box_match_list.append([t_box_match - t_box_match_0])
#
#         flow_gt = Flow(None, width, height)
#         flow_gt.read(os.path.join(flow_gt_path, str(img).zfill(6) + '_10.png'))
#
#         for idx in range(bboxList1.num):
#             print 'bbox ID = ', idx
#             bbox1 = bboxList1[idx]
#             bbox2 = bboxList2[idx]
#
#             t_kp_0 = timeit.default_timer()
#
#             # keypoint detection
#             pt1 = ORB_point(bbox1)
#             pt2 = ORB_point(bbox2)
#             kp1, des1 = pt1.get_keypoint(res1[int(bbox1.top_left_y):
#                              int(bbox1.bottom_right_y),
#                              int(bbox1.top_left_x):
#                              int(bbox1.bottom_right_x)],
#                              fix_width, fix_height)
#             kp2, des2 = pt2.get_keypoint(res2[int(bbox2.top_left_y):
#                              int(bbox2.bottom_right_y),
#                              int(bbox2.top_left_x):
#                              int(bbox2.bottom_right_x)],
#                              fix_width, fix_height)
#             if kp1 == [] or kp2 == []:
#                 continue
#             t_kp = timeit.default_timer()
#             print 'keypoint extraction time:', t_kp - t_kp_0
#             t_kp_list.append([t_kp - t_kp_0])
#             print 'keypoint number:', pt1.length, pt2.length
#             pt_pair = KeypointPairList(pt1, pt2)
#
#             # BFMatcher
#             t_matcher_0 = timeit.default_timer()
#
#             matchIdx = pt_pair.get_neighbor_score(1)
#             print 'get_neighbor_score'
#             print 'pre = ', pt_pair.neighbor_pre, 'score = ', pt_pair.neighbor_score
#
#             pt_pair.clip_val(matchIdx)
#
#             bfm = BruteForceMatcherList(pt_pair)
#             bfm.get_matcher()
#             t_matcher = timeit.default_timer()
#             print 'matcher number:', bfm.length
#             print 'matcher time :', t_matcher - t_matcher_0
#             t_matcher_list.append([t_matcher - t_matcher_0])
#
#             t_good_matcher_0 = timeit.default_timer()
#             bfm.get_good_matcher()  # Lowe's good feature threshold criteria(feature similarity distance)
#             t_good_matcher = timeit.default_timer()
#             print 'good matcher number:', bfm.length
#             print 'good matcher time:', t_good_matcher - t_good_matcher_0
#             t_good_matcher_list.append([t_good_matcher - t_good_matcher_0])
#
#             t_wgt_matcher_0 = timeit.default_timer()
#             bfm.get_wgt_dis_matcher()
#             t_wgt_matcher = timeit.default_timer()
#             print 'good weighted matcher number:', bfm.length
#             print 'good weighted matcher time:', t_wgt_matcher - t_wgt_matcher_0
#             t_wgt_matcher_list.append([t_wgt_matcher - t_wgt_matcher_0])
#
#             # # find homography
#             # t_homography_0 = timeit.default_timer()
#             # bfm.orientationVoting()
#             # Mat, _ = bfm.get_homography()
#             # t_homography = timeit.default_timer()
#             # print 'homography time:', t_good_matcher - t_good_matcher_0
#             # print ''
#
#             print 'matcher length = ', bfm.length
#
#             if bfm.length < 2:
#                 continue
#
#             end = timeit.default_timer()
#             print 'total time = ', end - t_kp_0
#             t.append([end - t_kp_0])
#
#             flow = Flow(bfm, width, height)
#             flow.compute(flow_path)
#
#             if flow != None and flow_gt != None:
#                 angle = np.arctan2(flow.val_y, flow.val_x)
#                 angle = angle[angle.nonzero()]
#
#                 ctr_x = bbox1.top_left_x + bbox1.width / 2
#                 ctr_y = bbox1.top_left_y + bbox1.height / 2
#
#                 trans_y = bbox2.bottom_right_y
#
#                 if np.sum(angle) != 0:
#                     # trans_x = trans_y / np.mean(angle)
#                     angle_list = Counter(angle)
#                     val, count = angle_list.most_common(1)[0]
#                     trans_x = trans_y / float(count)
#
#                     gt_val_x = flow_gt.val_x[int(bbox1.top_left_y):
#                                              int(bbox1.bottom_right_y),
#                                              int(bbox1.top_left_x):
#                                              int(bbox1.bottom_right_x)]
#                     gt_val_y = flow_gt.val_y[int(bbox1.top_left_y):
#                                              int(bbox1.bottom_right_y),
#                                              int(bbox1.top_left_x):
#                                              int(bbox1.bottom_right_x)]
#
#                     angle_gt = np.arctan2(gt_val_y, gt_val_x)
#                     angle_gt = angle_gt[angle_gt.nonzero()]
#
#                     trans_y_gt = bbox2.bottom_right_y
#
#                     if np.sum(angle_gt) != 0:
#                         # trans_x_gt = trans_y_gt / np.mean(angle_gt)
#                         angle_gt_list = Counter(angle_gt)
#                         val_gt, count_gt = angle_gt_list.most_common(1)[0]
#                         trans_x_gt = trans_y_gt / float(count_gt)
#
#                         diff_angle = (val - val_gt) * 180 / np.pi
#                         print 'difference angle =', diff_angle
#                         diff_angs.append(diff_angle)
#
#                         # bfm.vis_matches(img1, img2, ctr_x, ctr_y, trans_x, trans_y,
#                         #                   trans_x_gt, trans_y_gt, 1, 0, bfm.length)
#                         # bfm.write_matches(img1, img2,
#                         #                   os.path.join(voting_non_overlap_path,
#                         #                                str(img).zfill(6) + '_10_' +
#                         #                                str(idx) + '.png'),
#                         #                   ctr_x, ctr_y, trans_x, trans_y,
#                         #                   trans_x_gt, trans_y_gt,
#                         #                   1, 0, bfm.length)
#                         bfm.write_matches_overlap(img1, img2,
#                                                   os.path.join(voting_overlap_path,
#                                                                str(img).zfill(
#                                                                    6) + '_10_' + str(
#                                                                    idx) + '.png'),
#                                                   ctr_x, ctr_y, trans_x, trans_y,
#                                                   trans_x_gt, trans_y_gt, 1, 0,
#                                                   bfm.length)
#
#     print ''
#     print 'time: mean = ', np.mean(t), 'median = ', np.median(t), \
#         'max = ', np.amax(t), 'min = ', np.amin(t)
#     print 'box time: mean = ', np.mean(t_box_list), \
#         'median = ', np.median(t_box_list), \
#         'max = ', np.amax(t_box_list), 'min = ', np.amin(t_box_list)
#     print 'keypoint time: mean = ', np.mean(t_kp_list), \
#         'median = ', np.median(t_kp_list), \
#         'max = ', np.amax(t_kp_list), 'min = ', np.amin(t_kp_list)
#     print 'matcher time: mean = ', np.mean(t_matcher_list), \
#         'median = ', np.median(t_matcher_list), \
#         'max = ', np.amax(t_matcher_list), 'min = ', np.amin(t_matcher_list)
#     print 'good matcher time: mean = ', np.mean(t_good_matcher_list), \
#         'median = ', np.median(t_good_matcher_list), \
#         'max = ', np.amax(t_good_matcher_list), \
#         'min = ', np.amin(t_good_matcher_list)
#     print 'weighted matcher time: mean = ', np.mean(t_wgt_matcher_list), \
#         'median = ', np.median(t_wgt_matcher_list), \
#         'max = ', np.amax(t_wgt_matcher_list), \
#         'min = ', np.amin(t_wgt_matcher_list)
#     print 'wdifference angle: mean = ', np.mean(diff_angs), \
#         'median = ', np.median(diff_angs), \
#         'max = ', np.amax(diff_angs), \
#         'min = ', np.amin(diff_angs)

#
# if __name__ == '__main__':
#     main()
