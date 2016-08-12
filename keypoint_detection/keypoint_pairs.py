import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import gaussian_filter1d

import matplotlib.pyplot as plt

__author__ = 'Dandi Chen'


def read_img_pair(img_path1, img_path2):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    return img1, img2


class KeypointList(object):
    def __init__(self, bbox, kp=None, pt_x=None, pt_y=None,
                 pos_x=None, pos_y=None, mask=None, length=0):
        self.bbox = bbox         # focus on only one bounding box in each frame
        self.kp = kp
        self.pt_x = pt_x
        self.pt_y = pt_y
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.mask = mask
        self.length = length

    def init_val(self, img_width, img_height, kp=None, pt_vec=None):
        self.kp = kp
        self.mask = np.zeros((img_height, img_width), dtype=bool)
        if pt_vec == None and kp != None:
            self.length = len(kp)
            self.pt_x = np.zeros(self.length)
            self.pt_y = np.zeros(self.length)
            self.pos_x = np.zeros(self.length)
            self.pos_y = np.zeros(self.length)
            for idx in range(self.length):
                (x, y) = self.kp[idx].pt
                self.pt_x[idx] = x + self.bbox.top_left_x
                self.pt_y[idx] = y + self.bbox.top_left_y
        else:   # kp == None and pt_vec != None
            self.length = len(pt_vec[0])
            self.pt_x = np.zeros(self.length)
            self.pt_y = np.zeros(self.length)
            self.pos_x = np.zeros(self.length)
            self.pos_y = np.zeros(self.length)
            self.pt_x = pt_vec[0] + self.bbox.top_left_x
            self.pt_y = pt_vec[1] + self.bbox.top_left_y

        idx_x = np.int0(self.pt_x).tolist()
        idx_y = np.int0(self.pt_y).tolist()
        self.mask[(idx_y, idx_x)] = True
        self.pos_x = (self.pt_x - self.bbox.top_left_x) / self.bbox.width
        self.pos_y = (self.pt_y - self.bbox.top_left_y) / self.bbox.height

    def set_val(self, idx):
        self.length = len(idx)
        tmp = []
        for i in np.asarray(idx)[0]:
            tmp.append(self.kp[i])
        self.kp = None
        self.kp = tmp

        self.pt_x = self.pt_x[idx]
        self.pt_y = self.pt_y[idx]
        self.pos_x = self.pos_x[idx]
        self.pos_y = self.pos_y[idx]
        self.pt[self.pt_y][self.pt_x] = True


class KeypointPairList(object):
    def __init__(self, list1, list2):
        self.list1 = list1
        self.list2 = list2
        self.distance = np.zeros(min(self.list1.length, self.list2.length),
                                 dtype=float)
        self.neighbor_vec = np.zeros(self.list1.length, dtype=int)
        self.neighbor_pre = 0.0
        self.neighbor_score = 0.0

    def init_val(self, kp1, kp2):
        self.list1.init_val(kp1)
        self.list2.init_val(kp2)

    def set_val(self, idx):
        self.list1.set_val(idx)
        self.list2.set_val(idx)
        self.distance = self.distance[idx]
        self.neighbor_vec = self.neighbor_vec[idx]
        self.neighbor_pre = np.sum(self.neighbor_vec) / float(self.list1.length)

    def get_euclidean_dis(self):
        pt1 = np.array([self.list1.pos_x, self.list1.pos_y]).transpose()
        pt2 = np.array([self.list2.pos_x, self.list2.pos_y]).transpose()
        self.distance = np.sqrt(np.sum((pt1 - pt2) ** 2, axis=1))

    def get_neighbor_score(self, threshold=0.02):
        pt1 = np.array([self.list1.pos_x, self.list1.pos_y]).transpose()
        pt2 = np.array([self.list2.pos_x, self.list2.pos_y]).transpose()
        pairwise_dis = cdist(pt1, pt2, 'euclidean')
        pairwise_dis = (pairwise_dis - np.amin(pairwise_dis)) / \
                       (np.max(pairwise_dis) - np.min(pairwise_dis))
        self.neighbor_vec = np.sum((pairwise_dis < threshold).astype(int),
                                   axis=1)
        self.neighbor_pre = len(self.neighbor_vec[self.neighbor_vec != 0]) / \
                            float(self.list1.length)
        self.neighbor_score = np.sum(pairwise_dis[np.where(pairwise_dis
                                                           < threshold)])

    def get_grid_neighbor_score(self, blk_size=5):
        if blk_size %  2 == 0:
            blk_size += 1
        delta = (blk_size - 1) / 2

        box_delta_x = self.list2.bbox.top_left_x - self.list1.bbox.top_left_x
        box_delta_y = self.list2.bbox.top_left_y - self.list1.bbox.top_left_y

        scaled_x = self.list2.bbox.width / self.list1.bbox.width
        scaled_y = self.list2.bbox.height / self.list2.bbox.height

        mask = self.list1.mask
        for idx in range(self.list1.length):
            x_start = np.int0(self.list1.pt_x[idx] * scaled_x + box_delta_x) - delta
            x_end = np.int0(self.list1.pt_x[idx] * scaled_x + box_delta_x) + delta + 1
            y_start = np.int0(self.list1.pt_y[idx] * scaled_y + box_delta_y) - delta
            y_end = np.int0(self.list1.pt_y[idx] * scaled_y + box_delta_y) + delta + 1

            mask[y_start:y_end, x_start:x_end] = True
            statisfied = np.logical_and(
                self.list2.mask[y_start:y_end, x_start:x_end],
                mask[y_start:y_end, x_start:x_end])
            self.neighbor_vec[idx] = len(statisfied[np.where(
                statisfied == True)])
        self.neighbor_score = np.sum(self.neighbor_vec)
        self.neighbor_pre = len(self.neighbor_vec[np.where(
            self.neighbor_vec != 0)]) / float(self.list1.length)

    def get_grid_kernel_neighbor_score(self, blk_size=5):
        if blk_size %  2 == 0:
            blk_size += 1
        delta = (blk_size - 1) / 2
        sigma = blk_size / 4

        box_delta_x = self.list2.bbox.top_left_x - self.list1.bbox.top_left_x
        box_delta_y = self.list2.bbox.top_left_y - self.list1.bbox.top_left_y

        scaled_x = self.list2.bbox.width / self.list1.bbox.width
        scaled_y = self.list2.bbox.height / self.list2.bbox.height

        mask = self.list1.mask
        for idx in range(self.list1.length):
            x_start = np.int0(self.list1.pt_x[idx] * scaled_x + box_delta_x) - delta
            x_end = np.int0(self.list1.pt_x[idx] * scaled_x + box_delta_x) + delta + 1
            y_start = np.int0(self.list1.pt_y[idx] * scaled_y + box_delta_y) - delta
            y_end = np.int0(self.list1.pt_y[idx] * scaled_y + box_delta_y) + delta + 1

            mask[y_start:y_end, x_start:x_end] = True
            statisfied = np.logical_and(
                self.list2.mask[y_start:y_end, x_start:x_end],
                mask[y_start:y_end, x_start:x_end])

            self.neighbor_vec[idx] = len(statisfied[np.where(
                statisfied == True)])

            self.neighbor_score += np.sum(gaussian_filter1d(np.ones(
                (blk_size, blk_size)) * statisfied * 1, sigma))

        self.neighbor_pre = len(self.neighbor_vec[np.where(
            self.neighbor_vec != 0)]) / float(self.list1.length)

    def vis_pt_pairs(self, img1, img2):
        shown_img1 = img1
        shown_img2 = img2
        cv2.rectangle(shown_img1, (
            int(self.list1.bbox.top_left_x), int(self.list1.bbox.top_left_y)),
                      (int(self.list1.bbox.bottom_right_x),
                       int(self.list1.bbox.bottom_right_y)),
                      (0, 255, 0), 4)

        cv2.rectangle(shown_img2, (
            int(self.list2.bbox.top_left_x), int(self.list2.bbox.top_left_y)),
                      (int(self.list2.bbox.bottom_right_x),
                       int(self.list2.bbox.bottom_right_y)),
                      (0, 255, 0), 4)

        if self.list1.bbox.top_left_x == 0 and self.list1.bbox.top_left_y == 0 \
                and self.list2.bbox.top_left_x == 0 and \
                        self.list2.bbox.top_left_y == 0:
            shown_img1 = cv2.drawKeypoints(img1, self.list1.kp, shown_img1,
                                           color=(0, 255, 0), flags=0)
            cv2.imshow('image1', shown_img1)
            cv2.waitKey(0)

            shown_img2 = cv2.drawKeypoints(img2, self.list2.kp, shown_img2,
                                           color=(0, 255, 0), flags=0)
            cv2.imshow('image2', shown_img2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            if self.list1.bbox.top_left_x != 0 or \
                            self.list1.bbox.top_left_y != 0:
                for idx in range(self.list1.length):
                    cv2.circle(shown_img1, (int(self.list1.pt_x[idx]),
                                            int(self.list1.pt_y[idx])), 3,
                               color=(0, 255, 0))
                cv2.imshow('image1', shown_img1)
                cv2.waitKey(0)

            if self.list2.bbox.top_left_x != 0 or \
                            self.list2.bbox.top_left_y != 0:
                for idx in range(self.list2.length):
                    cv2.circle(shown_img2, (int(self.list2.pt_x[idx]),
                                            int(self.list2.pt_y[idx])),
                               3, color=(0, 255, 0))
                cv2.imshow('image2', shown_img2)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def vis_mask(self):
        box_delta_x = self.list2.bbox.top_left_x - self.list1.bbox.top_left_x
        box_delta_y = self.list2.bbox.top_left_y - self.list1.bbox.top_left_y

        scaled_x = self.list2.bbox.width / self.list1.bbox.width
        scaled_y = self.list2.bbox.height / self.list2.bbox.height

        plt.figure()
        plt.scatter(self.list1.pt_x , self.list1.pt_y, label='bbox1')
        plt.scatter(self.list2.pt_x, self.list2.pt_y, c='r', label='bbox2')
        plt.legend()
        plt.title('pt_x/y')

        plt.figure()
        plt.scatter(self.list1.pt_x * scaled_x ,
                    self.list1.pt_y * scaled_y, label='bbox1')
        plt.scatter(self.list2.pt_x, self.list2.pt_y, c='r', label='bbox2')
        plt.legend()
        plt.title('pt_x/y * scale')

        plt.figure()
        plt.scatter(self.list1.pt_x * scaled_x + box_delta_x,
                    self.list1.pt_y * scaled_y + box_delta_y, label='bbox1')
        plt.scatter(self.list2.pt_x, self.list2.pt_y, c='r', label='bbox2')
        plt.legend()
        plt.title('pt_x/y * scale + bbox')

        plt.figure()
        plt.scatter(self.list1.pt_x + box_delta_x,
                    self.list1.pt_y + box_delta_y, label='bbox1')
        plt.scatter(self.list2.pt_x, self.list2.pt_y, c='r', label='bbox2')
        plt.legend()
        plt.title('pt_x/y + bbox')

        plt.figure()
        plt.scatter(self.list1.pos_x, self.list1.pos_y, label='bbox1')
        plt.scatter(self.list2.pos_x, self.list2.pos_y, c='r', label='bbox2')
        plt.legend()
        plt.title('pos_x/y')
        plt.show()

        plt.waitforbuttonpress()
        plt.close('all')

    def write_pt_pairs(self, img1, img2, kp_path1, kp_path2):
        shown_img1 = img1
        shown_img2 = img2

        if self.list1.bbox.top_left_x == 0 and self.list1.bbox.top_left_y == 0 \
                and self.list2.bbox.top_left_x == 0 and \
                        self.list2.bbox.top_left_y == 0:
            shown_img1 = cv2.drawKeypoints(img1, self.list1.kp, shown_img1,
                                           color=(0, 255, 0), flags=0)
            cv2.imwrite(kp_path1, shown_img1)

            shown_img2 = cv2.drawKeypoints(img2, self.list2.kp, shown_img2,
                                           color=(0, 255, 0), flags=0)
            cv2.imwrite(kp_path2, shown_img2)
        else:
            if self.list1.bbox.top_left_x != 0 or \
                            self.list1.bbox.top_left_y != 0:
                for idx in range(self.list1.length):
                    cv2.circle(shown_img1, (int(self.list1.pt_x[idx]),
                                            int(self.list1.pt_y[idx])), 3,
                               color=(0, 255, 0))
                cv2.imwrite(kp_path1, shown_img1)

            if self.list2.bbox.top_left_x != 0 or \
                            self.list2.bbox.top_left_y != 0:
                for idx in range(self.list2.length):
                    cv2.circle(shown_img2, (int(self.list2.pt_x[idx]),
                                            int(self.list2.pt_y[idx])), 3,
                               color=(0, 255, 0))
                cv2.imwrite(kp_path2, shown_img2)
