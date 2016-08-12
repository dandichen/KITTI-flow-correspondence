import cv2
import numpy as np
import matplotlib.pyplot as plt

__author__ = 'Dandi Chen'

def flow2color(flow):
    """
        plot optical flow
        optical flow have 2 channel : u ,v indicate displacement
    """
    hsv = np.zeros(flow.shape[:2] + (3,)).astype(np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    # dilation = cv2.dilate(rgb, kernel)
    return rgb

    # plt.figure()
    # plt.imshow(rgb)
    # plt.title('optical flow')
    # plt.waitforbuttonpress()

def plot_flow_value(img1, img2, flow, flow_gt):
    plt.figure()

    plt.subplot(421)
    plt.imshow(img1)
    plt.title('frame1')

    plt.subplot(423)
    plt.imshow(flow2color(flow))
    plt.title('flow detected')

    plt.subplot(425)
    plt.imshow(flow.x_val)
    plt.title('x flow detected')

    plt.subplot(427)
    plt.imshow(flow.y_val)
    plt.title('y flow detected')

    plt.subplot(422)
    plt.imshow(img2)
    plt.title('frame2')

    plt.subplot(424)
    plt.imshow(flow_gt)
    plt.title('flow ground truth')

    plt.subplot(426)
    plt.imshow(flow_gt.x_val)
    plt.title('x flow ground truth')

    plt.subplot(428)
    plt.imshow(flow_gt.y_val)
    plt.title('y flow ground truth')

    # maximize figure window
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    plt.show()