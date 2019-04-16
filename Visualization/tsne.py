# -*- coding:utf-8 -*-
"""
create by gezhipeng
create on 19-1-2 下午5:51
func: 
"""
import numpy
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize_by_tsne(datas, labels, save_path, n_components=2, learning_rate=100):
    """
    TSNE visualization, save the result
    :param datas: numpy array
    :param labels: numpy array
    :param save_path: the path to save the figure
    :param dims: dimension of results
    :return:
    """
    tsne = TSNE(n_components=n_components, learning_rate=learning_rate, n_iter=1000)
    res = tsne.fit_transform(datas, labels)
    if n_components == 2:
        xs = res[:, 0]
        ys = res[:, 1]
        plt.scatter(xs, ys, c=labels)
        plt.savefig(save_path)
        plt.show()

    elif n_components == 3:
        fig = plt.figure()
        xs = res[:, 0]
        ys = res[:, 1]
        zs = res[:, 2]
        ax = Axes3D(fig)
        ax.scatter(xs, ys, zs, c=labels)
        plt.savefig(save_path)
        plt.show()


if __name__ == '__main__':
    # 加载数据集
    iris = load_iris()
    # 共有150个sample, 3类数据的类型是numpy.ndarray
    print(iris.data.shape)
    print(numpy.unique(iris.target))
    datas = iris.data
    labels = iris.target
    visualize_by_tsne(datas, labels, 'res.png', 2)

