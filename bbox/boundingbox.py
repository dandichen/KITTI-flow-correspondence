import cv2
import os
import requests
import base64
import json
import timeit
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from keypoint_detection.keypoint_pairs import normalize_arr

import matplotlib.pyplot as plt

__author__ = 'Dandi Chen'

# server = 'http://objectdetection.app.tusimple.sd/v1/analyzer/objdetect'
server = 'http://192.168.1.162:32771/v1/analyzer/objdetect' # exciting 12

class BoundingBox(object):
    def __init__(self, top_left_x=0, top_left_y=0,
                 bottom_right_x=0, bottom_right_y=0):
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.bottom_right_x = bottom_right_x
        self.bottom_right_y = bottom_right_y
        self.width = abs(bottom_right_x - top_left_x)
        self.height = abs(bottom_right_y - top_left_y)

    def __getitem__(self):
        return self

    def vis_box(self, out_img):
        cv2.rectangle(out_img, (int(self.top_left_x), int(self.top_left_y)),
                      (int(self.bottom_right_x), int(self.bottom_right_y)),
                      (0, 255, 0), 4)
        cv2.imshow('bounding box', out_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def write_box(self, out_img, box_path):
        cv2.rectangle(out_img, (int(self.top_left_x), int(self.top_left_y)),
                          (int(self.bottom_right_x), int(self.bottom_right_y)),
                      (0, 255, 0), 4)
        cv2.imwrite(box_path, out_img)

    def vis_box_grid(self, img, x_trans, y_trans, x_num, y_num,
                     x_blk_size, y_blk_size, width, height,
                     img_x_trans=0, img_y_trans=0,
                     box_x_trans=0, box_y_trans=0):
        for i in range(x_num - 1):
            cv2.line(img,
                     (x_trans[i] + img_x_trans + box_x_trans,
                      y_trans[0] + img_y_trans + box_y_trans),
                     (x_trans[i] + img_x_trans + box_x_trans,
                      height + img_y_trans + box_y_trans),
                     (0, 255, 0))
        cv2.line(img,
                 (x_trans[x_num - 2] + x_blk_size + img_x_trans + box_x_trans,
                  y_trans[0] + img_y_trans + box_y_trans),
                 (x_trans[x_num - 2] + x_blk_size + img_x_trans + box_x_trans,
                  height + img_y_trans + box_y_trans),
                 (0, 255, 0))

        for j in range(y_num - 1):
            cv2.line(img,
                     (x_trans[0] + img_x_trans + box_x_trans,
                      y_trans[j * (x_num - 1)] + img_y_trans + box_y_trans),
                     (width + img_x_trans + box_x_trans,
                      y_trans[j * (x_num - 1)] + img_y_trans + box_y_trans),
                     (0, 255, 0))
        cv2.line(img, (
            x_trans[0] + img_x_trans + box_x_trans,
            y_trans[(y_num - 2) * (x_num - 1)] + y_blk_size +
            img_y_trans + box_y_trans),
                 (width + img_x_trans + box_x_trans,
                  y_trans[(y_num - 2) * (x_num - 1)] + y_blk_size +
                  img_y_trans + box_y_trans), (0, 255, 0))


        # cv2.imshow('box grid', img)
        # cv2.waitKey(0)
        # cv2.destroyWindow('box grid')

    def write_box_grid(self, img, x_trans, y_trans, x_num, y_num,
                       x_blk_size, y_blk_size, width, height,
                       img_x_trans=0, img_y_trans=0,
                       box_x_trans=0, box_y_trans=0):
        for i in range(x_num - 1):
            cv2.line(img, (x_trans[i] + img_x_trans + box_x_trans,
                           y_trans[0] + img_y_trans + box_y_trans),
                     (x_trans[i] + img_x_trans + box_x_trans,
                      height + img_y_trans + box_y_trans),
                     (0, 255, 0))
        cv2.line(img,
                 (x_trans[x_num - 2] + x_blk_size + img_x_trans + box_x_trans,
                  y_trans[0] + img_y_trans + box_y_trans),
                 (x_trans[x_num - 2] + x_blk_size + img_x_trans + box_x_trans,
                  height + img_y_trans + box_y_trans), (0, 255, 0))

        for j in range(y_num - 1):
            cv2.line(img,
                     (x_trans[0] + img_x_trans + box_x_trans,
                      y_trans[j * (x_num - 1)] + img_y_trans + box_y_trans),
                     (width + img_x_trans + box_x_trans,
                      y_trans[j * (x_num - 1)] + img_y_trans + box_y_trans),
                     (0, 255, 0))
        cv2.line(img, (
        x_trans[0] + img_x_trans + box_x_trans,
        y_trans[(y_num - 2) * (x_num - 1)] + y_blk_size +
        img_y_trans + box_y_trans),
                 (width + img_x_trans + box_x_trans,
                  y_trans[(y_num - 2) * (x_num - 1)] + y_blk_size +
                  img_y_trans + box_y_trans),
                 (0, 255, 0))

class BoundingBoxList(object):
    def __init__(self, top_left_x=None, top_left_y=None,
                 bottom_right_x=None, bottom_right_y=None,
                 ctr_x=None, ctr_y=None, width=None, height=None,
                 B_his=None, G_his=None, R_his=None):
        self.num = 0
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.bottom_right_x = bottom_right_x
        self.bottom_right_y = bottom_right_y
        # self.ctr_x = ctr_x
        # self.ctr_y = ctr_y
        self.width = width
        self.height = height
        # self.B_his = B_his
        # self.G_his = G_his
        # self.R_his = R_his

    def __getitem__(self, item):
        return BoundingBox(self.top_left_x[item], self.top_left_y[item],
                           self.bottom_right_x[item], self.bottom_right_y[item])

    def init_val(self, num, bins=32):
        self.top_left_x = np.zeros(num)
        self.top_left_y = np.zeros(num)
        self.bottom_right_x = np.zeros(num)
        self.bottom_right_y = np.zeros(num)
        # self.ctr_x = np.zeros(num)
        # self.ctr_y = np.zeros(num)
        self.width = np.zeros(num)
        self.height = np.zeros(num)
        # self.B_his = np.zeros((num, bins), dtype=float)
        # self.G_his = np.zeros((num, bins), dtype=float)
        # self.R_his = np.zeros((num, bins), dtype=float)


    def set_val(self, img, box, width, height, num=10, bins=32):
        self.init_val(num)
        count = 0
        for idx in range(len(box)):
            if box[idx]['confidence'] > 0.9 and box[idx]['type'] == 'CAR':
                self.num += 1
                self.top_left_x[idx] = width * box[idx + count]['left']
                self.top_left_y[idx] = height * box[idx + count]['top']
                self.bottom_right_x[idx] = width * box[idx + count]['right']
                self.bottom_right_y[idx] = height * box[idx + count]['bottom']
                self.width[idx] = abs(self.bottom_right_x[idx + count] - self.top_left_x[idx + count])
                self.height[idx] = abs(self.bottom_right_y[idx + count] - self.top_left_y[idx + count])
                # self.ctr_x[idx] = self.top_left_x[idx + count] + self.width[idx + count] / 2
                # self.ctr_y[idx] = self.top_left_y[idx + count] + self.height[idx+ count] / 2
                # self.get_histogram(img, idx, bins)
                # print 'idx = ', idx, 'left, top, right, bottom', \
                #     self.top_left_x[idx], self.top_left_y[idx], \
                #     self.bottom_right_x[idx], self.bottom_right_y[idx]
            else:
                count += 1
                continue
                # self.num -= 1
                # self.top_left_x = np.delete(self.top_left_x, self.num)
                # self.top_left_y = np.delete(self.top_left_y, self.num)
                # self.bottom_right_x = np.delete(self.bottom_right_x, self.num)
                # self.bottom_right_y = np.delete(self.bottom_right_y, self.num)
                # self.ctr_x = np.delete(self.ctr_x, self.num)
                # self.ctr_y = np.delete(self.ctr_y, self.num)
                # self.width = np.delete(self.width, self.num)
                # self.height = np.delete(self.height, self.num)
                # self.B_his = np.delete(self.B_his, self.num, 0)
                # self.G_his = np.delete(self.G_his, self.num, 0)
                # self.R_his = np.delete(self.R_his, self.num, 0)
    def set_list(self, idx):
        self.num = len(idx)
        self.top_left_x = self.top_left_x[idx]
        self.top_left_y = self.top_left_y[idx]
        self.bottom_right_x = self.bottom_right_x[idx]
        self.bottom_right_y = self.bottom_right_y[idx]
        self.width = self.width[idx]
        self.height = self.height[idx]

    def get_box(self, img, width, height):
        im_compress = cv2.imencode('.png', img)[1]
        encoded_string = base64.b64encode(im_compress)
        payload = {'image_base64': encoded_string}
        response = requests.post(server, json=payload)
        result = json.loads(response.text)
        box = result['objs']
        self.set_val(img, box, width, height)

    def get_histogram(self, img, idx, bins=32):
        y1 = int(self.top_left_y[idx])
        y2 = int(self.bottom_right_y[idx])
        x1 = int(self.top_left_x[idx])
        x2 = int(self.bottom_right_x[idx])
        patch_B = img[y1:y2, x1:x2, 0]
        patch_G = img[y1:y2, x1:x2, 1]
        patch_R = img[y1:y2, x1:x2, 2]
        self.B_his[idx] = cv2.calcHist([patch_B], [0], None, [bins], [0, 256])[0]
        self.G_his[idx] = cv2.calcHist([patch_G], [0], None, [bins], [0, 256])[0]
        self.R_his[idx] = cv2.calcHist([patch_R], [0], None, [bins], [0, 256])[0]

    def bbox_matching(self, bboxList):
        pt1 = np.array([self.top_left_x[0:self.num], self.top_left_y[0:self.num],
                        self.bottom_right_x[0:self.num], self.bottom_right_y[0:self.num]]).transpose()
        pt2 = np.array([bboxList.top_left_x[0:bboxList.num], bboxList.top_left_y[0:bboxList.num],
                        bboxList.bottom_right_x[0:bboxList.num], bboxList.bottom_right_y[0:bboxList.num]]).transpose()

        if self.num <= bboxList.num:
            min_idx = np.argmin(cdist(pt1, pt2), axis=1)
            bboxList.set_list(min_idx)
        else:
            min_idx = np.argmin(cdist(pt2, pt1), axis=1)
            self.set_list(min_idx)

    def vis_box(self, out_img, color):
        for idx in range(self.num):
            print (int(self.top_left_x[idx]), int(self.top_left_y[idx])), \
                (int(self.bottom_right_x[idx]), int(self.bottom_right_y[idx]))
            cv2.rectangle(out_img, (int(self.top_left_x[idx]),
                                    int(self.top_left_y[idx])),
                          (int(self.bottom_right_x[idx]),
                           int(self.bottom_right_y[idx])),
                          color[np.mod(idx, 100)].tolist(), 4)
        cv2.imshow('bounding box', out_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def write_box(self, out_img, color, box_path):
        for idx in range(self.num):
            cv2.rectangle(out_img, (int(self.top_left_x[idx]),
                                    int(self.top_left_y[idx])),
                          (int(self.bottom_right_x[idx]),
                           int(self.bottom_right_y[idx])),
                          color[np.mod(idx, 100)].tolist(), 4)
        cv2.imwrite(box_path, out_img)


class BoundingBoxPairList(object):
    def __init__(self):
        self.list1 = BoundingBoxList()
        self.list2 = BoundingBoxList()

    def init_val(self, num):
        self.list1.init_val(num)
        self.list2.init_val(num)

    def set_val(self, img1, img2, bbox1, bbox2, width, height):
        self.list1.set_val(img1, bbox1, width, height)
        self.list2.set_val(img2, bbox2, width, height)

    def set(self, idx):
        self.length = len(idx)
        tmp = []
        for i in np.asarray(idx):
            tmp.append(self.kp[i])
        self.kp = None
        self.kp = tmp

        self.pt_x = self.pt_x[idx]
        self.pt_y = self.pt_y[idx]
        self.pos_x = self.pos_x[idx]
        self.pos_y = self.pos_y[idx]

        idx_x = np.int0(self.pt_x).tolist()
        idx_y = np.int0(self.pt_y).tolist()
        self.mask[(idx_y, idx_x)] = True

    def set_list(self, idx, num):
        matched = BoundingBoxPairList()
        matched.init_val(num)
        if self.list1.num <= self.list2.num:
            matched.list1 = self.list1
            matched.list2.num = self.list1.num
            matched.list2.top_left_x = self.list2.top_left_x[idx]
            matched.list2.top_left_y = self.list2.top_left_y[idx]
            matched.list2.bottom_right_x = self.list2.bottom_right_x[idx]
            matched.list2.bottom_right_y = self.list2.bottom_right_y[idx]
            matched.list2.ctr_x = self.list2.ctr_x[idx]
            matched.list2.ctr_y = self.list2.ctr_y[idx]
            matched.list2.width = self.list2.width[idx]
            matched.list2.height = self.list2.height[idx]
            matched.list2.B_his = self.list2.B_his[idx]
            matched.list2.G_his = self.list2.G_his[idx]
            matched.list2.R_his = self.list2.R_his[idx]
        else:
            matched.list2 = self.list2
            matched.list1.num = self.list2.num
            matched.list1.top_left_x = self.list1.top_left_x[idx]
            matched.list1.top_left_y = self.list1.top_left_x[idx]
            matched.list1.bottom_right_x = self.list1.bottom_right_x[idx]
            matched.list1.bottom_right_y = self.list1.bottom_right_y[idx]
            matched.list1.ctr_x = self.list1.ctr_x[idx]
            matched.list1.ctr_y = self.list1.ctr_y[idx]
            matched.list1.width = self.list1.width[idx]
            matched.list1.height = self.list1.height[idx]
            matched.list1.B_his = self.list1.B_his[idx]
            matched.list1.G_his = self.list1.G_his[idx]
            matched.list1.R_his = self.list1.R_his[idx]
        return matched

    def get_box_pair(self, img1, img2, width, height):
        print 'img1'
        self.list1.get_box(img1, width, height)
        print 'img2'
        self.list2.get_box(img2, width, height)

    # def vis_box_pair(self, img1, img2):
    #     color = np.random.randint(0, 255, (100, 3))
    #     for idx in range(self.list1.num):
    #         self.list1.vis_box(img1, idx, color)
    #         self.list2.vis_box(img2, idx, color)
    #
    #     cv2.imshow('img1 bounding box', img1)
    #     cv2.waitKey(0)
    #
    #     cv2.imshow('img2 bounding box', img2)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    def write_box_pair(self, out_img1, out_img2, box_path1, box_path2):
        color = np.random.randint(0, 255, (100, 3))
        for idx in range(min(self.list1.num, self.list2.num)):
            self.list1.write_box(out_img1, idx, box_path1, color)
            self.list2.write_box(out_img2, idx, box_path2, color)

        cv2.imwrite(box_path, out_img1)
        cv2.imwrite(box_path, out_img2)

    def box_matching(self, width, height):
        # Euclidean distance between center points of bounding boxes
        pt1 = np.array([self.list1.top_left_x[0 : self.list1.num],
                        self.list1.top_left_y[0 : self.list1.num],
                        self.list1.bottom_right_x[0 : self.list1.num],
                        self.list1.bottom_right_y[0: self.list1.num]]).transpose()
        pt2 = np.array([self.list2.top_left_x[0: self.list2.num],
                        self.list2.top_left_y[0: self.list2.num],
                        self.list2.bottom_right_x[0: self.list2.num],
                        self.list2.bottom_right_y[0: self.list2.num]]).transpose()
        min_idx = np.zeros(self.list1.num, dtype=int)
        for i in range(self.list1.num):
            tmp = []
            for j in range(self.list2.num):
                tmp.append(np.mean(abs(pt2[j] - pt1[i])))
            min_idx[i] = np.argmin(tmp)
        print min_idx
        matched = self.set_list(min_idx, self.list2.num)


        # ctr_dis = cdist(pt1, pt2)
        #
        # # color histogram of bounding boxes
        # B_his_dis = cdist(self.list1.B_his[0 : self.list1.num], self.list2.B_his[0 : self.list2.num])
        # G_his_dis = cdist(self.list1.G_his[0 : self.list1.num], self.list2.G_his[0 : self.list2.num])
        # R_his_dis = cdist(self.list1.R_his[0 : self.list1.num], self.list2.R_his[0 : self.list2.num])
        #
        # # width/height ratio between bounding boxes
        # diag1 = np.sqrt(self.list1.width[0 : self.list1.num]**2 + self.list1.height[0 : self.list1.num]**2)
        # diag2 = np.sqrt(self.list2.width[0 : self.list2.num]**2 + self.list2.height[0 : self.list2.num]**2)
        # vec1 = np.array([self.list1.height[0 : self.list1.num] / diag1, self.list1.width[0 : self.list1.num] / diag1]).transpose()
        # vec2 = np.array([self.list2.height[0 : self.list2.num] / diag2, self.list2.width[0 : self.list2.num] / diag2]).transpose()
        # diag_dis = cdist(vec1, vec2)
        #
        # if self.list1.num <= self.list2.num:
        #     min_idx = np.zeros(self.list1.num, dtype=int)
        #     distance = np.zeros((self.list1.num, self.list2.num))
        #     for idx in range(self.list1.num):
        #         # # Euclidean distance between center points of bounding boxes
        #         ctr_dis[idx, :] = normalize_arr(ctr_dis[idx, :], np.amax(ctr_dis[idx, :]), np.amin(ctr_dis[idx, :]))
        #         #
        #         # # color histogram of bounding boxes
        #         B_his_dis[idx, :] = normalize_arr(B_his_dis[idx, :], np.amax(B_his_dis[idx, :]), np.amin(B_his_dis[idx, :]))
        #         G_his_dis[idx, :] = normalize_arr(G_his_dis[idx, :], np.amax(G_his_dis[idx, :]), np.amin(G_his_dis[idx, :]))
        #         R_his_dis[idx, :] = normalize_arr(R_his_dis[idx, :], np.amax(R_his_dis[idx, :]), np.amin(R_his_dis[idx, :]))
        #
        #         his_dis = (B_his_dis + G_his_dis + R_his_dis) / 3.0
        #         his_dis[idx, :] = normalize_arr(his_dis[idx, :], np.amax(his_dis[idx, :]), np.amin(his_dis[idx, :]))
        #         #
        #         # # width/height ratio between bounding boxes
        #         diag_dis[idx, :] = normalize_arr(diag_dis[idx, :], np.amax(diag_dis[idx, :]), np.amin(diag_dis[idx, :]))
        #
        #         # distance fusion
        #         print np.mean(ctr_dis), np.mean(his_dis), np.mean(diag_dis)
        #         # dis = (ctr_dis + his_dis + diag_dis) / 3.0
        #         dis = (0.07 * ctr_dis + 0.9 * his_dis + 0.03 * diag_dis) / 3.0
        #
        #         # distance[idx, :] = normalize_arr(dis[idx, :], np.amax(dis[idx, :]), np.amin(dis[idx, :]))
        #         min_idx = pd.unique(np.argmin(dis, axis=1))
        #         print min_idx
        #         matched = self.set_list(min_idx, self.list1.num)
        # else:
        #     min_idx = np.zeros(self.list1.num, dtype=int)
        #     distance = np.zeros((self.list2.num, self.list1.num))
        #     for idx in range(self.list2.num):
        #         # # Euclidean distance between center points of bounding boxes
        #         # ctr_dis[idx, :] = normalize_arr(ctr_dis[idx, :], np.amax(ctr_dis[idx, :]), np.amin(ctr_dis[idx, :]))
        #         #
        #         # # color histogram of bounding boxes
        #         # B_his_dis[idx, :] = normalize_arr(B_his_dis[idx, :], np.amax(B_his_dis[idx, :]), np.amin(B_his_dis[idx, :]))
        #         # G_his_dis[idx, :] = normalize_arr(G_his_dis[idx, :], np.amax(G_his_dis[idx, :]), np.amin(G_his_dis[idx, :]))
        #         # R_his_dis[idx, :] = normalize_arr(R_his_dis[idx, :], np.amax(R_his_dis[idx, :]), np.amin(R_his_dis[idx, :]))
        #         #
        #         his_dis = (B_his_dis + G_his_dis + R_his_dis) / 3.0
        #         # his_dis[idx, :] = normalize_arr(his_dis[idx, :], np.amax(his_dis[idx, :]), np.amin(his_dis[idx, :]))
        #         #
        #         # # width/height ratio between bounding boxes
        #         # diag_dis[idx, :] = normalize_arr(diag_dis[idx, :], np.amax(diag_dis[idx, :]), np.amin(diag_dis[idx, :]))
        #
        #         # distance fusion
        #         dis = (ctr_dis + his_dis + diag_dis) / 3.0
        #
        #         # distance[idx, :] = normalize_arr(dis[:, idx], np.amax(dis[:, idx]), np.amin(dis[:, idx]))
        #         min_idx = pd.unique(np.argmin(dis, axis=1))
        #         print min_idx
        #         matched = self.set_list(min_idx, self.list2.num)
        return matched


# path = '/mnt/scratch/haoyiliang/KITTI/SceneFLow2015/data_scene_flow/training/image_2/'
# box_path = '/mnt/scratch/DandiChen/KITTI/bbox/fill/test/'
# # bboxPairList = BoundingBoxPairList()
#
# fix_height = 1080
# fix_width = 1920
#
# for img in range(1):
#     print ''
#     print 'img number: ', img
#
#     img_path1 = os.path.join(path, str(img).zfill(6) + '_10.png')
#     img_path2 = os.path.join(path, str(img).zfill(6) + '_11.png')
#
#     box_path1 = os.path.join(box_path, str(img).zfill(6) + '_10.png')
#     box_path2 = os.path.join(box_path, str(img).zfill(6) + '_11.png')
#
#     img1 = cv2.imread(img_path1)
#     img2 = cv2.imread(img_path2)
#     height, width, _ = img1.shape
#
#     res1 = np.ones((fix_height, fix_width, 3), dtype=np.uint8) * 255
#     res2 = np.ones((fix_height, fix_width, 3), dtype=np.uint8) * 255
#
#     res1[0: height, 0: width, :] = img1
#     res2[0: height, 0: width, :] = img2
#
#     bboxList1 = BoundingBoxList()
#     bboxList2 = BoundingBoxList()
#
#     bboxList1.get_box(res1, fix_width, fix_height)
#     bboxList2.get_box(res2, fix_width, fix_height)
#
#     color = np.random.randint(0, 255, (100, 3))
#     # bboxList1.vis_box(res1, color)
#     # bboxList2.vis_box(res2, color)
#     # bboxList1.write_box(res1, os.path.join(box_path, str(img).zfill(6) + '_10.png'))
#     # bboxList2.write_box(res2, os.path.join(box_path, str(img).zfill(6) + '_11.png'))
#
#     bboxList1.bbox_matching(bboxList2)
#     # bboxList1.vis_box(res1, color)
#     # bboxList2.vis_box(res2, color)
#     bboxList1.write_box(res1, os.path.join(box_path, str(img).zfill(6) + '_10.png'))
#     bboxList2.write_box(res2, os.path.join(box_path, str(img).zfill(6) + '_11.png'))
#
#     # bboxPairList.get_box_pair(res1, res2, fix_width, fix_height)
#     # # bboxPairList.vis_box_pair(res1, res2)
#     #
#     # matched = bboxPairList.box_matching(width, height)
#     #
#     # matched.vis_box_pair(res1, res2)
#     # # color = np.random.randint(0, 255, (100, 3))
#     # # matched.list1.vis_box(res1, 0, color)
#     # # matched.list2.vis_box(res2, 0, color)


