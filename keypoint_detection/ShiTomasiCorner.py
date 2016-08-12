import cv2
import numpy as np

from keypoint_detection.keypoint_pairs import KeypointList

__author__ = 'Dandi Chen'


class ShiTomasi_point(KeypointList):
    def __init__(self, bbox):
        KeypointList.__init__(self, bbox)

    def get_keypoint(self, img_patch, img_width=1242, img_height=375):
        gray = cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
        corners = np.int0(corners)
        pt_vec = corners[:, 0].transpose()
        self.init_val(img_width, img_height, None, None, pt_vec)






