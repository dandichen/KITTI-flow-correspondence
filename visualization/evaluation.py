import numpy as np
import matplotlib.pyplot as plt

__author__ = "Dandi Chen"

def vis_evaluation(pair_num, corr_X_mat, corr_Y_mat, err_ratio_mat, ang_err_mat, end_pt_err_mat):
    plt.figure()
    plt.plot(np.arange(pair_num), corr_X_mat, 'ro', label='flow x')
    plt.plot(np.arange(pair_num), corr_Y_mat, 'bo', label='flow y')
    plt.legend()
    plt.title('correlation coefficient')
    plt.show()

    plt.figure()
    plt.plot(np.arange(pair_num), err_ratio_mat, 'bo')
    plt.title('KITTI error ratio')
    plt.show()

    plt.figure()
    plt.plot(np.arange(pair_num), ang_err_mat, 'bo')
    plt.title('angle error')
    plt.show()

    plt.figure()
    plt.plot(np.arange(pair_num), end_pt_err_mat, 'bo')
    plt.title('endpoint error')
    plt.show()