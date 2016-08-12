import cv2
import numpy as np

__author__ = 'Dandi Chen'


class MatcherList(object):
    def __init__(self, key_pt_pair_list, mList=None, distance=None, trainIdx=None, queryIdx=None,
                 imgIdx=None, mask=None, length=0):
        self.key_pt_pair_list = key_pt_pair_list
        self.mList = mList                                  # DMatch object list
        self.distance = distance                            # numpy matrix
        self.trainIdx = trainIdx                            # numpy matrix
        self.queryIdx = queryIdx                            # numpy matrix
        self.imgIdx = imgIdx                                # numpy matrix
        self.mask = mask                                    # numpy matrix(bool)
        self.length = length

    def init_val(self, matches):
        self.mList = matches
        self.length = len(matches)

        self.distance = np.zeros(self.length)
        self.trainIdx = np.zeros(self.length, dtype=int)
        self.queryIdx = np.zeros(self.length, dtype=int)
        self.imgIdx = np.zeros(self.length, dtype=int)
        self.mask = np.zeros(self.length, dtype=bool)

        for idx in range(self.length):
            self.distance[idx] = self.mList[idx].distance
            self.trainIdx[idx] = self.mList[idx].trainIdx
            self.queryIdx[idx] = self.mList[idx].queryIdx
            self.imgIdx[idx] = self.mList[idx].imgIdx

        # distance normalization
        max_dis = np.amax(self.distance)
        min_dis = np.amin(self.distance)
        if max_dis > 1:
            self.distance = (self.distance - min_dis) / float(max_dis - min_dis)

    def set_val(self, idx):
        self.key_pt_pair_list.set_val(idx)

        tmp = []
        for i in np.asarray(idx):
            tmp.append(self.mList[i])
        self.mList = None
        self.mList = tmp

        self.distance = self.distance[idx]
        self.trainIdx = self.trainIdx[idx]
        self.queryIdx = self.queryIdx[idx]
        self.imgIdx = self.imgIdx[idx]
        self.mask = self.mask[idx]
        self.length = len(np.asarray(idx))

    def vis_matches(self, img1, img2, ctr_x, ctr_y, trans_x, trans_y,
                    trans_x_gt=0, trans_y_gt=0, flag=0, show_start=0, show_end=50):
        if self.key_pt_pair_list.list1.bbox.top_left_x == 0 and self.key_pt_pair_list.list1.bbox.top_left_y == 0 and \
                self.key_pt_pair_list.list2.bbox.top_left_x == 0 and self.key_pt_pair_list.list2.bbox.top_left_y == 0:
            height, width, _ = img1.shape
            out = np.zeros((width * 2, height))
            out = cv2.drawMatches(img1, self.key_pt_pair_list.list1.kp, img2, self.key_pt_pair_list.list2.kp,
                                  self.mList[show_start:show_end], out, flags=2)
            cv2.imshow('ORB matches', out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            rows1, cols1, _ = img1.shape
            rows2, cols2, _ = img2.shape

            if flag == 0:  # horizontal visualization
                out_img = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
                out_img[:, 0:cols1, :] = img1
                out_img[:, cols1:cols1 + cols2, :] = img2

            else:  # vertical visualization
                out_img = np.zeros((rows1 + rows2, max([cols1, cols2]), 3), dtype='uint8')
                out_img[0:rows1, :, :] = img1
                out_img[rows1:rows1 + rows2, :, :] = img2

            for idx in range(show_end - show_start):
                if idx < self.key_pt_pair_list.list1.length:
                    cv2.circle(out_img,
                               (int(self.key_pt_pair_list.list1.pt_x[idx]),
                                int(self.key_pt_pair_list.list1.pt_y[idx])),
                               3, (255, 0, 0), 1)

                    color = np.random.randint(0, 255, (100, 3))
                    if flag == 0:  # horizontal visualization
                        cv2.circle(out_img,
                                   (int(self.key_pt_pair_list.list2.pt_x[
                                            idx]) + cols1,
                                    int(self.key_pt_pair_list.list2.pt_y[idx])),
                                   3, (255, 0, 0), 1)
                        cv2.line(out_img,
                                 (int(self.key_pt_pair_list.list1.pt_x[idx]),
                                  int(self.key_pt_pair_list.list1.pt_y[idx])),
                                 (
                                 int(self.key_pt_pair_list.list2.pt_x[idx]) + cols1,
                                 int(self.key_pt_pair_list.list2.pt_y[idx])),
                                 color[np.mod(idx, 100)].tolist(), 1)
                    else:  # vertical visualization
                        cv2.circle(out_img,
                                   (int(self.key_pt_pair_list.list2.pt_x[idx]),
                                    int(self.key_pt_pair_list.list2.pt_y[
                                            idx]) + rows1),
                                   3, (255, 0, 0), 1)
                        cv2.line(out_img,
                                 (int(self.key_pt_pair_list.list1.pt_x[idx]),
                                  int(self.key_pt_pair_list.list1.pt_y[idx])),
                                 (int(self.key_pt_pair_list.list2.pt_x[idx]),
                                  int(self.key_pt_pair_list.list2.pt_y[
                                          idx]) + rows1),
                                 color[np.mod(idx, 100)].tolist(), 1)

            # draw bounding box
            bbox1 = self.key_pt_pair_list.list1.bbox
            bbox2 = self.key_pt_pair_list.list2.bbox
            cv2.rectangle(out_img,
                          (int(bbox1.top_left_x),
                           int(bbox1.top_left_y)),
                          (int(bbox1.bottom_right_x),
                           int(bbox1.bottom_right_y)),
                          (0, 255, 0), 4)

            if flag == 0:  # horizontal visualization
                cv2.rectangle(out_img, (int(bbox2.top_left_x) + cols1,
                                        int(bbox2.top_left_y)),
                                        (int(bbox2.bottom_right_x) + cols1,
                                        int(bbox2.bottom_right_y)),
                              (0, 255, 0), 4)
            else:  # vertical visualization
                cv2.rectangle(out_img,
                              (int(bbox2.top_left_x),
                               int(bbox2.top_left_y) + rows1),
                              (int(bbox2.bottom_right_x),
                               int(bbox2.bottom_right_y) + rows1),
                              (0, 255, 0), 4)

            # # draw bounding box grid
            # x_blk_size = 32
            # y_blk_size = 32
            # _, x_trans1, y_trans1, _, _, x_num1, y_num1 = \
            #     grid.grid_img(img1[int(self.key_pt_pair_list.list1.bbox.top_left_y):int(
            #         self.key_pt_pair_list.list1.bbox.bottom_right_y),
            #                   int(self.key_pt_pair_list.list1.bbox.top_left_x):int(
            #                       self.key_pt_pair_list.list1.bbox.bottom_right_x)],
            #                   int(self.key_pt_pair_list.list1.bbox.width), int(self.key_pt_pair_list.list1.bbox.height),
            #                   x_blk_size, y_blk_size)
            # box = BoundingBox()
            # box.vis_box_grid(out_img, x_trans1, y_trans1, x_num1, y_num1, x_blk_size, y_blk_size,
            #                  int(self.key_pt_pair_list.list1.bbox.width), int(self.key_pt_pair_list.list1.bbox.height),
            #                  0, 0, int(self.key_pt_pair_list.list1.bbox.top_left_x),
            #                  int(self.key_pt_pair_list.list1.bbox.top_left_y))
            #
            # _, x_trans2, y_trans2, _, _, x_num2, y_num2 = \
            #     grid.grid_img(img2[int(self.key_pt_pair_list.list2.bbox.top_left_y):int(
            #         self.key_pt_pair_list.list2.bbox.bottom_right_y),
            #                   int(self.key_pt_pair_list.list2.bbox.top_left_x):int(
            #                       self.key_pt_pair_list.list2.bbox.bottom_right_x)],
            #                   int(self.key_pt_pair_list.list2.bbox.width), int(self.key_pt_pair_list.list2.bbox.height),
            #                   x_blk_size, y_blk_size)
            # if flag == 0:  # horizontal visualization
            #     box.vis_box_grid(out_img, x_trans2, y_trans2, x_num2, y_num2, x_blk_size, y_blk_size,
            #                      int(self.key_pt_pair_list.list2.bbox.width),
            #                      int(self.key_pt_pair_list.list2.bbox.height),
            #                      cols1, 0,
            #                      int(self.key_pt_pair_list.list2.bbox.top_left_x),
            #                      int(self.key_pt_pair_list.list2.bbox.top_left_y))
            # else:  # vertical visualization
            #     box.vis_box_grid(out_img, x_trans2, y_trans2, x_num2, y_num2, x_blk_size, y_blk_size,
            #                      int(self.key_pt_pair_list.list2.bbox.width),
            #                      int(self.key_pt_pair_list.list2.bbox.height),
            #                      0, rows1,
            #                      int(self.key_pt_pair_list.list2.bbox.top_left_x),
            #                      int(self.key_pt_pair_list.list2.bbox.top_left_y))

            # detected orientation: red
            cv2.arrowedLine(out_img, (int(ctr_x), int(ctr_y)),
                            (int(trans_x), int(trans_y) + rows1),
                            (0, 0, 255), 10)
            if trans_x_gt != 0 and trans_y_gt != 0:
                # ground truth: blue
                cv2.arrowedLine(out_img, (int(ctr_x), int(ctr_y)),
                                (int(trans_x_gt), int(trans_y_gt) + rows1),
                                (255, 0, 0), 10)

            cv2.imshow('matches', out_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def write_matches_overlap(self, img1, img2, match_path, ctr_x,ctr_y,
                              trans_x, trans_y, trans_x_gt=0, trans_y_gt=0,
                              flag=0, show_start=0, show_end=50):
        out_img = img1.copy()
        for idx in range(show_end - show_start):
            if idx < self.key_pt_pair_list.list1.length:
                cv2.circle(out_img,
                           (int(self.key_pt_pair_list.list1.pt_x[idx]),
                            int(self.key_pt_pair_list.list1.pt_y[idx])),
                           3, (255, 0, 0), 1)

                color = np.random.randint(0, 255, (100, 3))
                if flag == 0:  # horizontal visualization
                    cv2.circle(out_img,
                               (int(self.key_pt_pair_list.list2.pt_x[idx]),
                                int(self.key_pt_pair_list.list2.pt_y[idx])),
                               3, (255, 0, 0), 1)
                    cv2.line(out_img,
                             (int(self.key_pt_pair_list.list1.pt_x[idx]),
                              int(self.key_pt_pair_list.list1.pt_y[idx])),
                             (int(self.key_pt_pair_list.list2.pt_x[idx]),
                              int(self.key_pt_pair_list.list2.pt_y[idx])),
                             color[np.mod(idx, 100)].tolist(), 1)
                else:  # vertical visualization
                    cv2.circle(out_img,
                               (int(self.key_pt_pair_list.list2.pt_x[idx]),
                                int(self.key_pt_pair_list.list2.pt_y[idx])),
                               3, (255, 0, 0), 1)
                    cv2.line(out_img,
                             (int(self.key_pt_pair_list.list1.pt_x[idx]),
                              int(self.key_pt_pair_list.list1.pt_y[idx])),
                             (int(self.key_pt_pair_list.list2.pt_x[idx]),
                              int(self.key_pt_pair_list.list2.pt_y[idx])),
                             color[np.mod(idx, 100)].tolist(), 1)

        # draw bounding box
        bbox1 = self.key_pt_pair_list.list1.bbox
        bbox2 = self.key_pt_pair_list.list2.bbox
        cv2.rectangle(out_img,
                      (int(bbox1.top_left_x),
                       int(bbox1.top_left_y)),
                      (int(bbox1.bottom_right_x),
                       int(bbox1.bottom_right_y)),
                      (0, 255, 0), 4)

        cv2.rectangle(out_img,
                      (int(bbox2.top_left_x),
                       int(bbox2.top_left_y)),
                      (int(bbox2.bottom_right_x),
                       int(bbox2.bottom_right_y)),
                      (0, 255, 0), 4)

        # # draw bounding box grid
        # x_blk_size = 32
        # y_blk_size = 32
        # _, x_trans1, y_trans1, _, _, x_num1, y_num1 = \
        #     grid.grid_img(img1[int(self.key_pt_pair_list.list1.bbox.top_left_y):int(
        #         self.key_pt_pair_list.list1.bbox.bottom_right_y),
        #                   int(self.key_pt_pair_list.list1.bbox.top_left_x):int(
        #                       self.key_pt_pair_list.list1.bbox.bottom_right_x)],
        #                   int(self.key_pt_pair_list.list1.bbox.width), int(self.key_pt_pair_list.list1.bbox.height),
        #                   x_blk_size, y_blk_size)
        # box = BoundingBox()
        # box.vis_box_grid(out_img, x_trans1, y_trans1, x_num1, y_num1, x_blk_size, y_blk_size,
        #                  int(self.key_pt_pair_list.list1.bbox.width), int(self.key_pt_pair_list.list1.bbox.height),
        #                  0, 0, int(self.key_pt_pair_list.list1.bbox.top_left_x),
        #                  int(self.key_pt_pair_list.list1.bbox.top_left_y))
        #
        # _, x_trans2, y_trans2, _, _, x_num2, y_num2 = \
        #     grid.grid_img(img2[int(self.key_pt_pair_list.list2.bbox.top_left_y):int(
        #         self.key_pt_pair_list.list2.bbox.bottom_right_y),
        #                   int(self.key_pt_pair_list.list2.bbox.top_left_x):int(
        #                       self.key_pt_pair_list.list2.bbox.bottom_right_x)],
        #                   int(self.key_pt_pair_list.list2.bbox.width), int(self.key_pt_pair_list.list2.bbox.height),
        #                   x_blk_size, y_blk_size)
        # box.vis_box_grid(out_img, x_trans2, y_trans2, x_num2, y_num2, x_blk_size, y_blk_size,
        #                  int(self.key_pt_pair_list.list2.bbox.width), int(self.key_pt_pair_list.list2.bbox.height),
        #                  0, 0,
        #                  int(self.key_pt_pair_list.list2.bbox.top_left_x),
        #                  int(self.key_pt_pair_list.list2.bbox.top_left_y))

        # detected orientation: red
        cv2.arrowedLine(out_img, (int(ctr_x), int(ctr_y)),
                        (int(trans_x), int(trans_y)),
                        (0, 0, 255), 10)
        if trans_x_gt != 0 and trans_y_gt != 0:
            # ground truth: blue
            cv2.arrowedLine(out_img, (int(ctr_x), int(ctr_y)),
                            (int(trans_x_gt), int(trans_y_gt)),
                            (255, 0, 0), 10)

        cv2.imwrite(match_path, out_img)

    def write_matches(self, img1, img2, match_path, ctr_x,ctr_y,
                      trans_x, trans_y, trans_x_gt=0, trans_y_gt=0,
                      flag=0, show_start=0, show_end=50):
        if self.key_pt_pair_list.list1.bbox.top_left_x == 0 and self.key_pt_pair_list.list1.bbox.top_left_y == 0 and \
                        self.key_pt_pair_list.list2.bbox.top_left_x == 0 and self.key_pt_pair_list.list2.bbox.top_left_y == 0:
            height, width, _ = img1.shape
            out = np.zeros((width * 2, height))
            out = cv2.drawMatches(img1, self.key_pt_pair_list.list1.kp, img2,
                                  self.key_pt_pair_list.list2.kp,
                                  self.mList[show_start:show_end], out, flags=2)
            cv2.imshow('ORB matches', out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        else:
            rows1, cols1, _ = img1.shape
            rows2, cols2, _ = img2.shape

            if flag == 0:  # horizontal visualization
                out_img = np.zeros((max([rows1, rows2]), cols1 + cols2, 3),
                                   dtype='uint8')
                out_img[:, 0:cols1, :] = img1
                out_img[:, cols1:cols1 + cols2, :] = img2

            else:  # vertical visualization
                out_img = np.zeros((rows1 + rows2, max([cols1, cols2]), 3),
                                   dtype='uint8')
                out_img[0:rows1, :, :] = img1
                out_img[rows1:rows1 + rows2, :, :] = img2

            for idx in range(show_end - show_start):
                if idx < self.key_pt_pair_list.list1.length:
                    cv2.circle(out_img,
                               (int(self.key_pt_pair_list.list1.pt_x[idx]),
                                int(self.key_pt_pair_list.list1.pt_y[idx])),
                               3, (255, 0, 0), 1)

                    color = np.random.randint(0, 255, (100, 3))
                    if flag == 0:  # horizontal visualization
                        cv2.circle(out_img,
                                   (int(self.key_pt_pair_list.list2.pt_x[
                                            idx]) + cols1,
                                    int(self.key_pt_pair_list.list2.pt_y[idx])),
                                   3, (255, 0, 0), 1)
                        cv2.line(out_img,
                                 (int(self.key_pt_pair_list.list1.pt_x[idx]),
                                  int(self.key_pt_pair_list.list1.pt_y[idx])),
                                 (
                                 int(self.key_pt_pair_list.list2.pt_x[idx]) + cols1,
                                 int(self.key_pt_pair_list.list2.pt_y[idx])),
                                 color[np.mod(idx, 100)].tolist(), 1)
                    else:  # vertical visualization
                        cv2.circle(out_img,
                                   (int(self.key_pt_pair_list.list2.pt_x[idx]),
                                    int(self.key_pt_pair_list.list2.pt_y[
                                            idx]) + rows1),
                                   3, (255, 0, 0), 1)
                        cv2.line(out_img,
                                 (int(self.key_pt_pair_list.list1.pt_x[idx]),
                                  int(self.key_pt_pair_list.list1.pt_y[idx])),
                                 (int(self.key_pt_pair_list.list2.pt_x[idx]),
                                  int(self.key_pt_pair_list.list2.pt_y[
                                          idx]) + rows1),
                                 color[np.mod(idx, 100)].tolist(), 1)

            # draw bounding box
            bbox1 = self.key_pt_pair_list.list1.bbox
            bbox2 = self.key_pt_pair_list.list2.bbox
            cv2.rectangle(out_img,
                          (int(bbox1.top_left_x),
                           int(bbox1.top_left_y)),
                          (int(bbox1.bottom_right_x),
                           int(bbox1.bottom_right_y)),
                          (0, 255, 0), 4)

            if flag == 0:  # horizontal visualization
                cv2.rectangle(out_img, (int(bbox2.top_left_x) + cols1,
                                        int(bbox2.top_left_y)),
                                        (int(bbox2.bottom_right_x) + cols1,
                                        int(bbox2.bottom_right_y)),
                              (0, 255, 0), 4)
            else:  # vertical visualization
                cv2.rectangle(out_img,
                              (int(bbox2.top_left_x),
                               int(bbox2.top_left_y) + rows1),
                              (int(bbox2.bottom_right_x),
                               int(bbox2.bottom_right_y) + rows1),
                              (0, 255, 0), 4)

            # detected orientation: red
            cv2.arrowedLine(out_img, (int(ctr_x), int(ctr_y)),
                            (int(trans_x), int(trans_y) + rows1),
                            (0, 0, 255), 10)

            if trans_x_gt != 0 and trans_y_gt != 0:
                # ground truth: blue
                cv2.arrowedLine(out_img, (int(ctr_x), int(ctr_y)),
                                (int(trans_x_gt), int(trans_y_gt) + rows1),
                                (255, 0, 0), 10)

            cv2.imwrite(match_path, out_img)