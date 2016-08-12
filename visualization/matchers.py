import cv2
import numpy as np

def draw_Matches(img1, kp1, x_trans1, y_trans1, img2, kp2, matches):
    rows1, cols1, _ = img1.shape
    rows2, cols2, _ = img2.shape

    out_img = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
    out_img[:, 0:cols1, :] = img1
    out_img[:, cols1:cols1 + cols2, :] = img2

    i = 0
    for mat in matches:

        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # radius 4
        # colour blue
        # thickness = 2
        cv2.circle(out_img, (int(x1) + x_trans1, int(y1) + y_trans1), 2, (255, 0, 0), 1)
        cv2.circle(out_img, (int(x2) + cols1, int(y2)), 2, (255, 0, 0), 1)

        # thickness = 2
        # Create some random colors
        color = np.random.randint(0, 255, (100, 3))
        cv2.line(out_img, (int(x1) + x_trans1, int(y1) + y_trans1), (int(x2) + cols1, int(y2)), \
                 color[i].tolist(), 1)
        i += 1
    return out_img

def draw_Matches_trans(img1, kp1, x_trans1, y_trans1, img2, kp2, x_trans2, y_trans2, matches):
    rows1, cols1, _ = img1.shape
    rows2, cols2, _ = img2.shape

    out_img = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
    out_img[:, 0:cols1, :] = img1
    out_img[:, cols1:cols1 + cols2, :] = img2

    i = 0
    for mat in matches:

        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # radius 4
        # colour blue
        # thickness = 2
        cv2.circle(out_img, (int(x1) + x_trans1, int(y1) + y_trans1), 2, (255, 0, 0), 1)
        cv2.circle(out_img, (int(x2) + cols1 + x_trans2, int(y2) + y_trans2), 2, (255, 0, 0), 1)

        # thickness = 2
        # Create some random colors
        color = np.random.randint(0, 255, (100, 3))
        cv2.line(out_img, (int(x1) + x_trans1, int(y1) + y_trans1), (int(x2) + cols1 + x_trans2, int(y2) + y_trans2), \
                 color[i].tolist(), 1)
        i += 1
    return out_img

def vis_matches(img1, kp1, x_trans1, y_trans1, img2, kp2, x_trans2, y_trans2, matches, show_start=0, show_end=50):
    # patch-image
    out_img = draw_Matches(img1, kp1, x_trans1, y_trans1, img2, kp2, matches[show_start:show_end])
    # patch-patch
    # out_img = draw_Matches_trans(img1, kp1, x_trans1, y_trans1, img2, kp2, x_trans2, y_trans2, matches[show_start:show_end])
    return out_img