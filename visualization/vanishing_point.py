import cv2
import matplotlib.pyplot as plt

__author__ = 'Dandi Chen'

def plot_vanishing_point(x, y, img):
    img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig = plt.figure()
    plt.imshow(img_show)
    plt.scatter([x], [y], s=100)          # blue dot, flowstereo(mxnet) detected vanishing point

    return fig