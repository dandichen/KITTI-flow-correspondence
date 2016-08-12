import cv2
import numpy as np

from keypoint_detection.keypoint_pairs import KeypointList

__author__ = 'Dandi Chen'


class Harris_point(KeypointList):
    def __init__(self, bbox):
        KeypointList.__init__(self, bbox)

    def get_keypoint(self, img_patch, img_width=1242, img_height=375,
                     threshold=0.01):
        gray = cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY)
        dst = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
        pt_idx = dst > threshold * dst.max()
        pt_vec = (np.where(pt_idx == True)[1], np.where(pt_idx == True)[0])
        self.init_val(img_width, img_height, None, None, pt_vec)






