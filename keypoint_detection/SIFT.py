import cv2

from keypoint_detection.keypoint_pairs import KeypointList

__author__ = 'Dandi Chen'


class SIFT_point(KeypointList):
    def __init__(self, bbox):
        KeypointList.__init__(self, bbox)

    def get_keypoint(self, img_patch, img_width=1242, img_height=375):
        # Initiate SIFT detector
        # sift = cv2.xfeatures2d.SIFT_create()
        sift = cv2.xfeatures2d.SIFT_create(0, 3, 0.05, 15, 1.6)

        # find the keypoints and descriptors with SIFT
        kp, des = sift.detectAndCompute(img_patch, None)

        if kp != [] and des != None:
            self.init_val(img_width, img_height, kp, des, None)

        return kp, des



