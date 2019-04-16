# -*- coding:utf-8 -*-
"""
create by gezhipeng
create on 18-12-24 下午2:48
func: 
"""
import sys
sys.path.append('../')
import os


def multi_process(func, *args, **kwargs):
    """
    :param func: function of process
    :param args: 需要repeat参数
    :param kwargs: 不需要进行repeat的参数
    :return:
    """
    # print(args, list(kwargs.values()))
    from itertools import repeat
    from multiprocessing import Pool, freeze_support
    freeze_support()
    params = zip(*tuple([repeat(arg) for arg in args] + list(kwargs.values())))
    # for i, j in params:
    #     print(i, j)
    # params = zip(*tuple([repeat(arg) for arg in args] + list(kwargs.values())))
    with Pool(processes=None) as pool:
        pool.starmap(func, params)
        # pool.starmap(func, zip(repeat('hello'), ['tom', 'jack']))
    exit()

def func(common, name):
    print(common, name, "current pid:{}".format(os.getpid()))

def test():
    multi_process(func, 'hello', names=['tom', 'jack'])

if __name__ == '__main__':
    test()


