import cv2
import numpy as np

from keypoint import Keypoint
import matplotlib.pyplot as plt

__author__ = "Dandi Chen"

def grid2mask(kp, x_trans, y_trans, matches):    # keypoints in each image patch
    x_mat = []
    y_mat = []
    for mat in matches:
        img_idx = mat.queryIdx
        (x, y) = kp[img_idx].pt
        x_mat.extend([int(x) + x_trans])
        y_mat.extend([int(y) + y_trans])
    return x_mat, y_mat

class BFMatcher(object):
    def __init__(self, match_num = 200):
        self.match_num = match_num

    def get_matcher(self, des1, des2):
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        if des1 != None and des2 != None:
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            return matches
        else:
            return None


    def get_match_flow(self, kp1, kp2, matches, match_path):
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        flow = dst_pts - src_pts
        flow_X = flow[:, 0, 0]
        flow_Y = flow[:, 0, 1]
        x = np.int_(src_pts[:, 0, 0])
        y = np.int_(src_pts[:, 0, 1])

        np.savez(match_path, x=x, y=y, flow_X=flow_X, flow_Y=flow_Y)

        return x, y, flow_X, flow_Y