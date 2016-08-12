import numpy as np

__author__ = "Dandi Chen"

def get_overlap_per(flow_mask, flow_mask_gt):
    ol = np.logical_and(flow_mask, flow_mask_gt)
    ol_num = len((np.where(ol == True))[0])
    gt_num = len((np.where(flow_mask_gt == True))[0])
    if gt_num != 0:
        per = ol_num / float(gt_num)

    else:
        per = 0
    return per, ol_num, gt_num
