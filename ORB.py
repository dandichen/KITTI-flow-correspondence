import cv2
import keypoint

__author__ = "Dandi Chen"

class ORB_point(keypoint.Keypoint):
    def __init__(self, kp_num):
        keypoint.Keypoint.__init__(self, kp_num)

    def get_keypoint(self, img1, img2):
        # Initiate ORB detector
        orb = cv2.ORB_create()

        # find the keypoints with ORB
        kp1 = orb.detect(img1, None)
        kp2 = orb.detect(img2, None)

        # compute the descriptors with ORB
        kp1, des1 = orb.compute(img1, kp1)
        kp2, des2 = orb.compute(img2, kp2)

        return kp1, des1, kp2, des2

    def single_img_keypoint(self, img):
        # Initiate ORB detector
        orb = cv2.ORB_create()

        # find the keypoints with ORB
        kp = orb.detect(img, None)

        # compute the descriptors with ORB
        kp, des = orb.compute(img, kp)
        return kp, des






