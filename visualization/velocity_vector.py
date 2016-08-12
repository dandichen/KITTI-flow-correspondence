import cv2
import numpy as np

__author__ = 'Dandi Chen'

def plot_velocity_vector(opt_flow, step, trans = 0):

    if trans == 0:
        flow = opt_flow
    else:
        flow = opt_flow - trans
    img = np.ones(flow.shape[:2] + (3,))
    for i in range(0, img.shape[0] - step, step):
        for j in range(0, img.shape[1] - step, step):
            try:
                # opencv 3.1.0
                if flow.shape[-1] == 2:
                    cv2.arrowedLine(img, (j, i), (j + int(round(flow[i, j, 0])), i + int(round(flow[i, j, 1]))),
                                    (150, 0, 0), 2)
                else:
                    cv2.arrowedLine(img, (j, i), (j + int(round(flow[i, j, 0])), i), (150, 0, 0), 2)

            except AttributeError:
                # opencv 2.4.8
                if flow.shape[-1] == 2:
                    cv2.line(img, (j, i), (j + int(round(flow[i, j, 0])), i + int(round(flow[i, j, 1]))),
                             (150, 0, 0), 2)
                else:
                    cv2.line(img, pt1=(j, i), pt2=(j + int(round(flow[i, j])), i), color=(150, 0, 0), thickness=1)

    # plt.figure()
    # plt.imshow(img)
    # plt.title('velocity vector')
    # plt.waitforbuttonpress()

    return img

def plot_velocity_vector_mask(opt_flow, mask, step, trans = 0):

    if trans == 0:
        flow = opt_flow
    else:
        flow = opt_flow - trans

    img = np.ones(flow.shape[:2] + (3,))
    for i in range(0, img.shape[0] - step, step):
        for j in range(0, img.shape[1] - step, step):
            if mask[i, j] != 0:
                try:
                    # opencv 3.1.0
                    if flow.shape[-1] == 2:
                        cv2.arrowedLine(img, (j, i), (j + int(round(flow[i, j, 0])), i + int(round(flow[i, j, 1]))),
                                        (150, 0, 0), 2)
                    else:
                        cv2.arrowedLine(img, (j, i), (j + int(round(flow[i, j, 0])), i), (150, 0, 0), 2)

                except AttributeError:
                    # opencv 2.4.8
                    if flow.shape[-1] == 2:
                        cv2.line(img, (j, i), (j + int(round(flow[i, j, 0])), i + int(round(flow[i, j, 1]))),
                                 (150, 0, 0), 2)
                    else:
                        cv2.line(img, pt1=(j, i), pt2=(j + int(round(flow[i, j])), i), color=(150, 0, 0), thickness=1)

    # plt.figure()
    # plt.imshow(img)
    # plt.title('velocity vector')
    # plt.waitforbuttonpress()

    return img
