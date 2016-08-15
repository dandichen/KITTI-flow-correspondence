import cv2
import requests
import base64
import json
import numpy as np
from scipy.spatial.distance import cdist

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

class BoundingBoxList(object):
    def __init__(self, top_left_x=None, top_left_y=None,
                 bottom_right_x=None, bottom_right_y=None,
                 width=None, height=None):
        self.num = 0
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.bottom_right_x = bottom_right_x
        self.bottom_right_y = bottom_right_y
        self.width = width
        self.height = height

    def __getitem__(self, item):
        return BoundingBox(self.top_left_x[item], self.top_left_y[item],
                           self.bottom_right_x[item], self.bottom_right_y[item])

    def init_val(self, num):
        self.top_left_x = np.zeros(num)
        self.top_left_y = np.zeros(num)
        self.bottom_right_x = np.zeros(num)
        self.bottom_right_y = np.zeros(num)
        self.width = np.zeros(num)
        self.height = np.zeros(num)

    def set_val(self, img, box, width, height, num=15):
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
            else:
                count += 1
                continue

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

