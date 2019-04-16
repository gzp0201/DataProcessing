# -*- coding:utf-8 -*-
"""
create by gezhipeng
create on 18-12-24 下午6:12
func: 
"""
import sys
sys.path.append('../')
import math

def view_bar(message, num, total):
    rate = num / float(total)
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()

