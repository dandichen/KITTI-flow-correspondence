import sys
sys.path.insert(0, '/mnt/scratch/third-party-packages/libopencv_3.1.0/lib/python')
sys.path.insert(1, '/mnt/scratch/third-party-packages/libopencv_3.1.0/lib')
sys.path.append('/mnt/scratch/third-party-packages/mxnet/python')

import os
import cv2
import timeit
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from bbox.boundingbox import BoundingBoxList
from keypoint_detection import keypoint_pairs
from keypoint_detection.keypoint_pairs import KeypointPairList
from keypoint_detection.ORB import ORB_point
from keypoint_matching.brute_force import BruteForceMatcherList
from keypoint_matching.flow import Flow

video_path = 'sunny.mp4'
video2img_path = '/mnt/scratch/dandichen/ours_data/sunny/data/'
flow_path = '/mnt/scratch/dandichen/ours_data/sunny/flow/'
voting_overlap_path = '/mnt/scratch/dandichen/ours_data/sunny/visualization/voting/overlap/'

split_step = 3


def get_video_len(in_video):
    cap = cv2.VideoCapture(in_video)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def vidoe2img(in_video, step = 1):
    cap = cv2.VideoCapture(in_video)
    video_len = get_video_len(in_video)

    out_frames = []
    out_frames_ID = []
    for frame_ID in range(0, video_len, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_ID)
        print 'img_idx = ', int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        _, cur_frame = cap.read()
        rgb = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB)

        out_frames.append(rgb)
        out_frames_ID.append([frame_ID])
    return out_frames, out_frames_ID

def extract_frame(video_path, video2img_path):
    # extract frames from videos
    out_frames, out_frames_ID = vidoe2img(video_path, split_step)

    for img_idx in range(len(out_frames)):
        plt.imsave(os.path.join(video2img_path + '/' + str(img_idx).zfill(6) + '.png'), out_frames[img_idx])


def main():
    # # extract frames from videos
    # out_frames, out_frames_ID = vidoe2img(video_path, split_step)
    #
    # for img_idx in range(len(out_frames)):
    #     plt.imsave(os.path.join(video2img_path + '/' + str(img_idx).zfill(6) + '.png'), out_frames[img_idx])
    #
    img_num = len([name for name in os.listdir(video2img_path) if os.path.isfile(os.path.join(video2img_path, name))]) - 1
    flow_num = img_num - 1  # continuous two frames

    t = []
    t_box_list = []
    t_box_match_list = []
    t_kp_list = []
    t_matcher_list = []
    t_good_matcher_list = []
    t_wgt_matcher_list = []
    diff_angs = []

    for img in range(170, flow_num - 1):
        print ''
        print 'img number: ', img

        img_path1 = os.path.join(video2img_path, str(img).zfill(6) + '.png')
        img_path2 = os.path.join(video2img_path, str(img+1).zfill(6) + '.png')

        img1, img2 = keypoint_pairs.read_img_pair(img_path1, img_path2)
        height, width, _ = img1.shape

        # bounding box
        start = timeit.default_timer()

        bboxList1 = BoundingBoxList()
        bboxList2 = BoundingBoxList()

        bboxList1.get_box(img1, width, height)
        bboxList2.get_box(img2, width, height)
        t_box = timeit.default_timer()

        t_box_match_0 = timeit.default_timer()
        bboxList1.bbox_matching(bboxList2)
        t_box_match = timeit.default_timer()

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
            kp1, des1 = pt1.get_keypoint(img1[int(bbox1.top_left_y):
            int(bbox1.bottom_right_y),
                                         int(bbox1.top_left_x):
                                         int(bbox1.bottom_right_x)],
                                         width, height)
            kp2, des2 = pt2.get_keypoint(img2[int(bbox2.top_left_y):
            int(bbox2.bottom_right_y),
                                         int(bbox2.top_left_x):
                                         int(bbox2.bottom_right_x)],
                                         width, height)
            if kp1 == [] or kp2 == []:
                continue
            t_kp = timeit.default_timer()
            print 'keypoint extraction time:', t_kp - t_kp_0
            t_kp_list.append([t_kp - t_kp_0])
            print 'keypoint number:', pt1.length, pt2.length
            pt_pair = KeypointPairList(pt1, pt2)

            # BFMatcher
            t_matcher_0 = timeit.default_timer()

            matchIdx = pt_pair.get_neighbor_score(1)
            print 'get_neighbor_score'
            print 'pre = ', pt_pair.neighbor_pre, 'score = ', pt_pair.neighbor_score

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

            if bfm.length < 2:
                continue

            end = timeit.default_timer()
            print 'total time = ', end - t_kp_0
            t.append([end - t_kp_0])

            flow = Flow(bfm, width, height)
            flow.compute(flow_path)

            if flow != None:
                angle = np.arctan2(flow.val_y, flow.val_x)
                angle = angle[angle.nonzero()]

                ctr_x = bbox1.top_left_x + bbox1.width / 2
                ctr_y = bbox1.top_left_y + bbox1.height / 2

                trans_y = ctr_y + 100.0

                if np.sum(angle) != 0:
                    # trans_x = trans_y / np.mean(angle)
                    angle_list = Counter(angle)
                    val, count = angle_list.most_common(1)[0]
                    trans_x = trans_y / float(val)

                    # bfm.vis_matches(img1, img2, ctr_x, ctr_y, trans_x, trans_y,
                    #                 1, 0, bfm.length)
                    #
                    # bfm.write_matches(img1, img2,
                    #                   os.path.join(voting_non_overlap_path,
                    #                                str(img).zfill(6) + '_10_' +
                    #                                str(idx) + '.png'),
                    #                   ctr_x, ctr_y, trans_x, trans_y,
                    #                   1, 0, bfm.length)

                    bfm.write_matches_overlap(img1, img2,
                                              os.path.join(
                                                  voting_overlap_path,
                                                  str(img).zfill(
                                                      6) + '_10_' + str(
                                                      idx) + '.png'),
                                              ctr_x, ctr_y, trans_x,
                                              trans_y, 1, 0,
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


if __name__ == '__main__':
    main()

