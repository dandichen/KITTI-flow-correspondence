import cv2

from keypoint_detection.keypoint_pairs import KeypointList

__author__ = 'Dandi Chen'


class SURF_point(KeypointList):
    def __init__(self, bbox):
        KeypointList.__init__(self, bbox)

    def get_keypoint(self, img_patch, img_width=1242, img_height=375):
        # Initiate SURF detector
        surf = cv2.xfeatures2d.SURF_create()
        surf.setExtended(True)
        surf.setHessianThreshold(200)

        # find the keypoints and descriptors with SIFT
        kp, des = surf.detectAndCompute(img_patch, None)

        self.init_val(img_width, img_height, kp, des, None)





