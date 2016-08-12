import cv2
import numpy as np
from keypoint_detection.keypoint_pairs import normalize_arr

__author__ = 'Dandi Chen'


class Flow(object):
    def __init__(self, matcher=None, width=1242, height=375):
        self.width = width
        self.height = height
        self.matcher = matcher
        self.val_x = np.zeros((height, width))
        self.val_y = np.zeros((height, width))
        self.mask = np.zeros((height, width), dtype=bool)

    def normalize_flow(self):
        val_x_nr = normalize_arr(self.val_x, np.amax(self.val_x), np.amin(self.val_x))
        val_y_nr = normalize_arr(self.val_y, np.amax(self.val_y), np.amin(self.val_y))
        return val_x_nr, val_y_nr

    def reshape_vec(self, width, height):
        flow_vec_x = np.reshape(self.val_x, width * height)
        flow_vec_y = np.reshape(self.val_y, width * height)
        return flow_vec_x, flow_vec_y

    def read(self, gt_path):
        flow_gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        self.matcher = None
        self.height = flow_gt.shape[0]
        self.width = flow_gt.shape[1]

        # KITTI definition
        self.val_x = (np.float_(flow_gt[:, :, 2]) - 2 ** 15) / 64.0  # [-512..+512]
        self.val_y = (np.float_(flow_gt[:, :, 1]) - 2 ** 15) / 64.0  # [-512..+512]
        self.mask = np.array(flow_gt[:, :, 0], dtype=bool)

    def compute(self, match_path):
        delta_x = self.matcher.key_pt_pair_list.list2.pt_x - self.matcher.key_pt_pair_list.list1.pt_x
        delta_y = self.matcher.key_pt_pair_list.list2.pt_y - self.matcher.key_pt_pair_list.list1.pt_y

        # flow has been defined in first frame
        for idx in range(self.matcher.key_pt_pair_list.list1.length):
            idx_x = self.matcher.key_pt_pair_list.list1.pt_x[idx]
            idx_y = self.matcher.key_pt_pair_list.list1.pt_y[idx]
            if idx_x < self.width and idx_y < self.height:
                self.val_x[int(idx_y)][int(idx_x)] = delta_x[idx]
                self.val_y[int(idx_y)][int(idx_x)] = delta_y[idx]
                self.mask[int(idx_y)][int(idx_x)] = True

        # np.savez(match_path, val_x=self.val_x, val_y=self.val_y, mask=self.mask)

    # visualization
    def write_flow2match_mask(self, img1, img2, width, height, vel_path, bbox1, bbox2, step=3):
        rows1, cols1, _ = img1.shape
        rows2, cols2, _ = img2.shape

        out_img = np.zeros((rows1 + rows2, max([cols1, cols2]), 3), dtype='uint8')
        out_img[0:rows1, :, :] = img1
        out_img[rows1:rows1 + rows2, :, :] = img2

        # draw bounding box
        cv2.rectangle(out_img,
                      (int(bbox1.top_left_x), int(bbox1.top_left_y)),
                      (int(bbox1.bottom_right_x), int(bbox1.bottom_right_y)),
                      (0, 255, 0), 4)
        cv2.rectangle(out_img,
                      (int(bbox2.top_left_x), int(bbox2.top_left_y) + rows1),
                      (int(bbox2.bottom_right_x), int(bbox2.bottom_right_y) + rows1),
                      (0, 255, 0), 4)

        for j in range(0, width - step, step):
            for i in range(0, height - step, step):
                if self.mask[i, j] == True:
                    cv2.circle(out_img, (j, i), 3, (255, 0, 0), 1)
                    cv2.circle(out_img,
                               (j + int(round(self.val_x[i, j])), i + int(round(self.val_y[i, j])) + rows1),
                               3, (255, 0, 0), 1)

                    color = np.random.randint(0, 255, (100, 3))
                    cv2.line(out_img, (j, i),
                             (j + int(round(self.val_x[i, j])), i + int(round(self.val_y[i, j])) + rows1),
                             color[np.mod(i + j, 100)].tolist(), 1)
        cv2.imwrite(vel_path, out_img)

    def write_flow2match_overlap_mask(self, img, width, height, vel_path, bbox1, bbox2, step=3):
        out_img = img.copy()

        # draw bounding box
        cv2.rectangle(out_img,
                      (int(bbox1.top_left_x), int(bbox1.top_left_y)),
                      (int(bbox1.bottom_right_x), int(bbox1.bottom_right_y)),
                      (0, 255, 0), 4)
        cv2.rectangle(out_img,
                      (int(bbox2.top_left_x), int(bbox2.top_left_y)),
                      (int(bbox2.bottom_right_x), int(bbox2.bottom_right_y)),
                      (0, 255, 0), 4)

        for j in range(0, width - step, step):
            for i in range(0, height - step, step):
                if self.mask[i, j] == True:
                    cv2.circle(out_img, (j, i), 3, (255, 0, 0), 1)
                    cv2.circle(out_img,
                               (j + int(round(self.val_x[i, j])), i + int(round(self.val_y[i, j]))),
                               3, (255, 0, 0), 1)

                    color = np.random.randint(0, 255, (100, 3))
                    cv2.line(out_img, (j, i), (j + int(round(self.val_x[i, j])), i + int(round(self.val_y[i, j]))),
                             color[np.mod(i + j, 100)].tolist(), 1)
        cv2.imwrite(vel_path, out_img)

    def write_velocity_vector_compare(self, img, flow_gt, vel_path, step1=10, step2=10):
        # white background
        vel_img = np.ones((self.height, self.width, 3), dtype=np.float64)*255
        for j in range(0, self.width - step1, step1):
            for i in range(0, self.height - step1, step1):
                cv2.arrowedLine(vel_img, (j, i),
                                (j + int(round(flow_gt.val_x[i, j])), i + int(round(flow_gt.val_y[i, j]))),
                                (0, 0, 150), 2)

            for i in range(0, self.height - step2, step2):
                cv2.arrowedLine(vel_img, (j, i),
                                (j + int(round(self.val_x[i, j])), i + int(round(self.val_y[i, j]))),
                                (255, 0, 0), 2)
        cv2.imwrite(vel_path, vel_img)

    def write_velocity_vector_compare_mask(self, flow_gt, vel_path, step1=10, step2=10):
        # white background
        vel_img = np.ones((self.height, self.width, 3), dtype=np.float64)*255
        for j in range(0, self.width - step1, step1):
            for i in range(0, self.height - step1, step1):
                    cv2.arrowedLine(vel_img, (j, i),
                                    (j + int(round(flow_gt.val_x[i, j])), i + int(round(flow_gt.val_y[i, j]))),
                                    (0, 0, 150), 2)

        for j in range(0, self.width - step2, step2):
            for i in range(0, self.height - step2, step2):
                if self.mask[i, j]:
                    cv2.arrowedLine(vel_img, (j, i),
                                    (j + int(round(self.val_x[i, j])), i + int(round(self.val_y[i, j]))),
                                    (255, 0, 0), 2)
        cv2.imwrite(vel_path, vel_img)