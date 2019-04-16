# -*- coding:utf-8 -*-
"""
create by gezhipeng
create on 18-12-28 下午5:58
func: 
"""
import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def plotCM(classes, matrix, savname):
    """classes: a list of class names"""
    # Normalize by row
    matrix = matrix.T
    matrix = matrix.astype(np.float)
    linesum = matrix.sum(0)
    linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
    matrix /= linesum
    # plot
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, cmap=plt.cm.GnBu)
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(i, j, str('%.2f' % (matrix[i, j] * 100)), va='center', ha='center')
    ax.set_xticklabels([''] + classes, rotation=45)
    ax.set_yticklabels([''] + classes)
    # save
    ax.set_title("predict")
    ax.set_ylabel("ground truth")
    plt.savefig(savname)
    # ax.imshow()
    # plt.show()
if __name__ == '__main__':
    classes = [str(i).zfill(5) for i in range(2)]
    matrix = np.array([
        [0.6, 0.4],
        [0.2, 0.8]
    ])
    plotCM(classes, matrix, 'res.png')