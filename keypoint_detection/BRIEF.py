import cv2

from keypoint_detection.keypoint_pairs import KeypointList

__author__ = 'Dandi Chen'


class BRIEF_point(KeypointList):
    def __init__(self, bbox):
        KeypointList.__init__(self, bbox)

    def get_keypoint(self, img_patch, img_width=1242, img_height=375):
        # Initiate STAR detector
        star = cv2.xfeatures2d.StarDetector_create()
        # star = cv2.xfeatures2d.StarDetector_create(16)

        # Initiate BRIEF extractor
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

        # find the keypoints with STAR
        kp = star.detect(img_patch, None)

        # compute the descriptors with BRIEF
        kp, des = brief.compute(img_patch, kp)

        self.init_val(img_width, img_height, kp, des, None)





