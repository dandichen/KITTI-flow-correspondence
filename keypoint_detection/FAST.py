import cv2

from keypoint_detection.keypoint_pairs import KeypointList

__author__ = 'Dandi Chen'


class FAST_point(KeypointList):
    def __init__(self, bbox):
        KeypointList.__init__(self, bbox)

    def get_keypoint(self, img_patch, img_width=1242, img_height=375):
        # Initiate FAST object with default values
        fast = cv2.FastFeatureDetector_create()
        fast.setThreshold(9)
        kp = fast.detect(img_patch, None)
        self.init_val(img_width, img_height, kp, None, None)






