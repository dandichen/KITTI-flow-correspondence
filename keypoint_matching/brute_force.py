import cv2
import numpy as np


from matcher import MatcherList

__author__ = 'Dandi Chen'


def get_matcher_1v1(des1, des2):
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)
    return matches

def vis_matches(out_img, kp1, kp2, bbox1, bbox2, matches, show_start=0, show_end=50):
    for idx in range(show_end - show_start):
        match = matches[show_start + idx]
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx

        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        (x1, y1) = (int(x1 + bbox1.top_left_x), int(y1 + bbox1.top_left_y))
        (x2, y2) = (int(x2 + bbox2.top_left_x), int(y2 + bbox2.top_left_y))
        print '(x1, y1)  =', (x1, y1), '(x2, y2) = ', (x2, y2)

        color = np.random.randint(0, 255, (100, 3))

        cv2.circle(out_img, (x1, y1), 3, (255, 0, 0), 1)
        cv2.circle(out_img, (x2, y2), 3, (255, 0, 0), 1)
        cv2.line(out_img, (x1, y1), (x2, y2),
                 color[np.mod(idx, 100)].tolist(), 1)
    return out_img

def vis_matchesList(out_img, kp1, kp2, bbox1, bbox2, matches, show_start=0, show_end=50):
    for idx in range(show_start, show_end):
        img1_idx = matches[show_start + idx].queryIdx
        img2_idx = matches[show_start + idx].trainIdx

        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        (x1, y1) = (int(x1 + bbox1.top_left_x), int(y1 + bbox1.top_left_y))
        (x2, y2) = (int(x2 + bbox2.top_left_x), int(y2 + bbox2.top_left_y))
        print '(x1, y1)  =', (x1, y1), '(x2, y2) = ', (x2, y2)

        color = np.random.randint(0, 255, (100, 3))

        cv2.circle(out_img, (x1, y1), 3, (255, 0, 0), 1)
        cv2.circle(out_img, (x2, y2), 3, (255, 0, 0), 1)
        cv2.line(out_img, (x1, y1), (x2, y2),
                 color[np.mod(idx, 100)].tolist(), 1)
    return out_img


class BruteForceMatcherList(MatcherList):
    def __init__(self, key_pt_pair_list, mList=None, distance=None, trainIdx=None, queryIdx=None,
                 imgIdx=None, mask=None, length=0):
        MatcherList.__init__(self, key_pt_pair_list, mList, distance, trainIdx, queryIdx, imgIdx, mask, length)

    def init_val(self, matches):
        MatcherList.init_val(self, matches)

    def set_val(self, idx):
        MatcherList.set_val(self, idx)

    def get_matcher(self):
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        matches = bf.match(self.key_pt_pair_list.list1.des, self.key_pt_pair_list.list2.des)
        matches = sorted(matches, key=lambda x: x.distance)
        self.init_val(matches)

    def get_good_matcher(self, threshold=0.7):
        idx = np.where(self.distance < threshold * np.amax(self.distance))
        self.set_val(np.array(idx).tolist()[0])

    def get_wgt_dis_matcher(self, weight=0.5):
        self.key_pt_pair_list.get_euclidean_dis()
        self.distance = (1 - weight) * self.distance + weight * self.key_pt_pair_list.distance

    def get_homography(self, src=None, min_match_count=10, threshold=3.0):
        if self.length > min_match_count:
            src_pts = np.array([self.key_pt_pair_list.list1.pt_x,
                                self.key_pt_pair_list.list1.pt_y]).reshape(-1, 1, 2)
            dst_pts = np.array([self.key_pt_pair_list.list2.pt_x,
                                self.key_pt_pair_list.list2.pt_y]).reshape(-1, 1, 2)

            Mat, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)
            self.mask = np.array(mask.ravel(), dtype=bool)

            if src != None:
                dst = cv2.perspectiveTransform(src, Mat)
            else:
                dst = None
        else:
            print "Not enough matches are found - %d/%d" % (self.length, min_match_count)
            dst = None
        return Mat, dst

    def orientationVoting(self, threshold=164):
        angle =  np.arctan2(self.key_pt_pair_list.list2.pt_y -
                            self.key_pt_pair_list.list1.pt_y,
                            self.key_pt_pair_list.list2.pt_x -
                            self.key_pt_pair_list.list1.pt_x) * 180 / np.pi
        idx = np.array(np.where(abs(angle) < threshold)[0])
        self.set_val(idx)



















