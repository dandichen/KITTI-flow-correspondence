import numpy as np

__author__ = 'Dandi Chen'

def get_outlier_err(flow_X, flow_Y, flow_X_gt, flow_Y_gt, flow_mask, flow_mask_gt, tau=3, tau_per=0.05):    # KITTI evaluation criteria: tau
    valid_ind = np.logical_and(flow_mask, flow_mask_gt)
    gt_amp = (flow_X_gt[valid_ind]**2 + flow_Y_gt[valid_ind]**2)**0.5

    delta_X = flow_X[valid_ind] - flow_X_gt[valid_ind]
    delta_Y = flow_Y[valid_ind] - flow_Y_gt[valid_ind]
    err_amp = (delta_X**2 + delta_Y**2)**0.5

    valid_mask = flow_mask_gt[valid_ind]
    err_num = len(np.logical_and(valid_mask, err_amp > tau, err_amp/gt_amp > tau_per))
    err_ratio = err_num/len(valid_mask)
    return err_ratio