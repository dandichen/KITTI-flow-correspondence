import numpy as np

import correlation as corr
import angular_error as ang

__author__ = 'Dandi Chen'


def get_endpoint_err(flow_X, flow_Y, flow_X_gt, flow_Y_gt, flow_mask, flow_mask_gt, width, height):
    # [IJCV 2011] A Database and Evaluation Methodology for Optical Flow (section 4.1)
    flow_X_re, flow_Y_re, flow_X_gt_re, flow_Y_gt_re = corr.reshape_flow(flow_X, flow_Y, flow_X_gt, flow_Y_gt,
                                                                         width, height)
    flow_X_nr, flow_Y_nr, flow_X_gt_nr, flow_Y_gt_nr = ang.normalize_flow(flow_X_re, flow_Y_re, flow_X_gt_re, flow_Y_gt_re)

    valid_idx = np.logical_and(flow_mask, flow_mask_gt)
    valid_idx_re = corr.reshape(valid_idx, width, height)

    delta_X = flow_X_nr[valid_idx_re] - flow_X_gt_nr[valid_idx_re]
    delta_Y = flow_Y_nr[valid_idx_re] - flow_Y_gt_nr[valid_idx_re]

    err_amp = (delta_X**2 + delta_Y**2)**0.5

    return np.mean(err_amp)