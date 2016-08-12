import os
import timeit
import numpy as np

from bbox.boundingbox import BoundingBox
from bbox.boundingbox import BoundingBoxPairList
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


__author__ = 'Dandi Chen'


img_path = '/mnt/scratch/haoyiliang/KITTI/SceneFLow2015/data_scene_flow/training/image_2/'
flow_gt_path = '/mnt/scratch/haoyiliang/KITTI/SceneFLow2015/data_scene_flow/training/flow_occ'

eval_path = './result/'
kp_path = eval_path + 'keypoint/'


img_num = len([name for name in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, name))])
flow_num = img_num - 1  # continuous two frames

# pair_num = img_num/2
pair_num = 1

t = []

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


    # bounding box coordinates
    bbox1 = BoundingBox(154.07749939, 181.342102051, 405.574401855, 305.924407959)
    bbox2 = BoundingBox(0.0, 156.604873657, 353.453063965, 351.0)

    # keypoint detection
    pt1 = Harris_point(bbox1)
    pt2 = Harris_point(bbox2)
    pt1.get_keypoint(img1[int(bbox1.top_left_y):int(bbox1.bottom_right_y),
                      int(bbox1.top_left_x):int(bbox1.bottom_right_x)],
                     width, height)
    pt2.get_keypoint(img2[int(bbox2.top_left_y):int(bbox2.bottom_right_y),
                      int(bbox2.top_left_x):int(bbox2.bottom_right_x)],
                     width, height)
    t_kp = timeit.default_timer()
    print 'keypoint extraction time:', t_kp - start
    print 'keypoint number:', pt1.length, pt2.length
    print ''
    pt_pair = KeypointPairList(pt1, pt2)

    # pt_pair.get_neighbor_score()
    # print 'get_neighbor_score'
    # print 'pre = ', pt_pair.neighbor_pre, 'score = ', pt_pair.neighbor_score
    # print ''
    # pt_pair.get_grid_neighbor_score()
    # print 'get_grid_neighbor_score'
    # print 'pre = ', pt_pair.neighbor_pre, 'score = ', pt_pair.neighbor_score
    # print ''
    # pt_pair.get_grid_kernel_neighbor_score()
    # print 'get_grid_kernel_neighbor_score'
    # print 'pre = ', pt_pair.neighbor_pre, 'score = ', pt_pair.neighbor_score
    # print ''
    # pt_pair.vis_mask()

    pt_pair.vis_pt_pairs(img1, img2)
    # pt_pair.write_pt_pairs(img1, img2, os.path.join(kp_path, str(img).zfill(6) + '_10.png'),
    #                         os.path.join(kp_path, str(img).zfill(6) + '_11.png'))

    end = timeit.default_timer()
    print 'total time = ', end - start
    t.append([end - start])

print ''
print 'ave time = ', np.mean(t)
