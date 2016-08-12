import cv2
import numpy as np

__author__ = "Dandi Chen"

def draw_Keypoints(img, kp):
    out_img = img

    for idx in range(len(kp)):
        (x, y) = kp[idx].pt

        # radius 4
        # colour blue
        # thickness = 2
        cv2.circle(out_img, (int(x), int(y)), 4, (255, 0, 0), 2)
    return out_img

def draw_Keypoints_trans(img, kp, x_trans, y_trans):
    out_img = img

    for idx in range(len(kp)):
        (x, y) = kp[idx].pt

        # radius 4
        # colour blue
        # thickness = 2
        cv2.circle(out_img, (int(x) + x_trans, int(y) + y_trans), 4, (255, 0, 0), 2)
    return out_img

def vis_pt_pairs(img1, kp1, x_trans1, y_trans1, img2, kp2, x_trans2, y_trans2,):
    # # patch-patch
    # shown_img1 = draw_Keypoints_trans(img1, kp1, x_trans1, y_trans1)
    # cv2.imshow('image1', shown_img1)
    # cv2.waitKey(0)
    #
    # shown_img2 = draw_Keypoints_trans(img2, kp2, x_trans2, y_trans2)
    # cv2.imshow('image2', shown_img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # patch-image
    shown_img1 = draw_Keypoints_trans(img1, kp1, x_trans1, y_trans1)
    cv2.imshow('image1', shown_img1)
    cv2.waitKey(0)

    shown_img2 = draw_Keypoints(img2, kp2)
    cv2.imshow('image2', shown_img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



