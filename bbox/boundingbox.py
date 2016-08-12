import cv2
# import requests
import base64
import json
import numpy as np
from scipy.spatial.distance import cdist

__author__ = 'Dandi Chen'

server = "http://detection.app.tusimple.sd/v1/analyzer/objdetect"


class BoundingBox(object):
    def __init__(self, top_left_x=0, top_left_y=0,
                 bottom_right_x=0, bottom_right_y=0):
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.bottom_right_x = bottom_right_x
        self.bottom_right_y = bottom_right_y
        self.width = self.bottom_right_x - self.top_left_x
        self.height = self.bottom_right_y - self.top_left_y
        self.ctr_x = self.top_left_x + self.width / 2
        self.ctr_y = self.top_left_y + self.height / 2

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
                 ctr_x=None, ctr_y=None, width=None, height=None):
        self.num = 0
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.bottom_right_x = bottom_right_x
        self.bottom_right_y = bottom_right_y
        self.ctr_x = ctr_x
        self.ctr_y = ctr_y
        self.width = width
        self.height = height

    def __getitem__(self, item):
        return BoundingBox(self.top_left_x[item], self.top_left_y[item],
                           self.bottom_right_x[item], self.bottom_right_y[item])

    def init_val(self, num):
        self.num = num
        self.top_left_x = np.zeros(self.num)
        self.top_left_y = np.zeros(self.num)
        self.bottom_right_x = np.zeros(self.num)
        self.bottom_right_y = np.zeros(self.num)
        self.ctr_x = np.zeros(self.num)
        self.ctr_y = np.zeros(self.num)
        self.width = np.zeros(self.num)
        self.height = np.zeros(self.num)

    def set_val(self, box, width, height):
        self.num = len(box)
        for idx in range(self.num):
            self.top_left_x[idx] = width * box[idx]['left']
            self.top_left_y[idx] = height * box[idx]['top']
            self.bottom_right_x[idx] = width * box[idx]['right']
            self.bottom_right_y[idx] = height * box[idx]['bottom']
        self.width = self.bottom_right_x - self.top_left_x
        self.height = self.bottom_right_y - self.top_left_y
        self.ctr_x = self.top_left_x + self.width / 2
        self.ctr_y = self.top_left_y + self.height / 2

    def get_box(self, img):
        height, width, _ = img.shape

        binary = cv2.imencode('.png', img)[1].tostring()
        encoded_string = base64.b64encode(binary)
        payload = {'image_base64': encoded_string, 'trim_detect': 0.8}
        response = requests.post(server, json=payload)
        result = json.loads(response.text)
        print result
        box = result['objs']
        self.set_val(box, width, height)

    def vis_box(self, out_img):
        for idx in range(self.num):
            cv2.rectangle(out_img, (int(self.top_left_x[idx]),
                                    int(self.top_left_y[idx])),
                          (int(self.bottom_right_x[idx]),
                           int(self.bottom_right_y[idx])),
                          (0, 255, 0), 4)
        cv2.imshow('bounding box', out_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def write_box(self, out_img, box_path):
        for idx in range(self.num):
            cv2.rectangle(out_img, (int(self.top_left_x[idx]),
                                    int(self.top_left_y[idx])),
                          (int(self.bottom_right_x[idx]),
                           int(self.bottom_right_y[idx])),
                          (0, 255, 0), 4)
        cv2.imwrite(box_path, out_img)


class BoundingBoxPairList(object):
    def __init__(self):
        self.list1 = BoundingBoxList()
        self.list2 = BoundingBoxList()
        self.num = min(self.list1.num, self.list2.num)

    def init_val(self, num):
        self.list1.init_val(num)
        self.list2.init_val(num)

    def set_val(self, bbox1, bbox2, width, height):
        self.list1.set_val(bbox1, width, height)
        self.list2.set_val(bbox2, width, height)

    def get_box_pair(self, img1, img2):
        self.list1.get_box(img1)
        self.list2.get_box(img2)

    def vis_box_pair(self, img1, img2):
        self.list1.vis_box(img1)
        self.list2.vis_box(img2)

    def box_matching(self, width, height):
        pt1 = np.array([self.list1.ctr_x, self.list1.ctr_y]).transpose()
        pt2 = np.array([self.list2.ctr_x, self.list2.ctr_y]).transpose()
        pairwise_dis = cdist(pt1, pt2, 'euclidean')
        match_idx = np.amin(pairwise_dis, axis=1)
        self.list2.set_val(self.list2[match_idx], width, height)
































